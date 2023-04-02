from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from typing import List

import click
import fsspec
import numpy as np
import requests
from tqdm import tqdm

from tmx_pipeline.utils import (
    KafkaConfig,
    SentPair,
    create_consumer_from_topic,
    init_logging,
)
from tmx_pipeline.workers import (
    clean_pairs_and_write_to_kafka,
    parse_tmx_and_write_to_kafka,
)

logger = logging.getLogger(__name__)


@click.group()
@click.option("--kafka-hosts", default="localhost:9092", help="Kafka hosts")
@click.option(
    "--unprocessed-topic",
    default="unprocessed_topic",
    help="Kafka topic with unprocessed pairs stream",
)
@click.option(
    "--filtered-topic",
    default="filtered_topic",
    help="Kafka topic with filtered pairs stream",
)
@click.option("--group-name", default="default", help="Kafka group name")
@click.pass_context
def cli(
    ctx, kafka_hosts: str, unprocessed_topic: str, filtered_topic: str, group_name: str
):
    ctx.ensure_object(dict)
    ctx.obj["KAFKA"] = KafkaConfig(
        kafka_hosts, group_name, unprocessed_topic, filtered_topic
    )


@cli.command()
@click.option(
    "--num-proc", default=os.cpu_count(), type=click.INT, help="Number of processes"
)
@click.argument("files", nargs=-1, required=True)
@click.pass_context
def parse_tmx(ctx, num_proc: int, files: List[str]) -> None:
    parse_tmx_and_write_to_kafka(files, ctx.obj["KAFKA"], num_proc)


@cli.command()
@click.option(
    "--num-proc", default=os.cpu_count(), type=click.INT, help="Number of processes"
)
@click.pass_context
def clean_pairs(ctx, num_proc: int) -> None:
    clean_pairs_and_write_to_kafka(ctx.obj["KAFKA"], num_proc)


@cli.command()
@click.argument("output_folder", nargs=1)
@click.pass_context
def write_to_file(ctx, output_folder: str) -> None:
    output_file = os.path.join(
        output_folder, f"filtered_pairs_{time.strftime('%Y%m%d-%H%M%S')}.jl"
    )
    kafka_config = ctx.obj["KAFKA"]

    consumer = create_consumer_from_topic(
        kafka_config.servers,
        kafka_config.filtered_topic,
        kafka_config.group_name,
        timeout=60,
        exit_on_timeout=True,
    )

    logger.info(f"Writing data to {output_file}")
    with fsspec.open(output_file, "wb") as f, consumer as c:
        entries = 0
        for pair in c:
            f.write(pair)
            f.write(b"\n")
            entries += 1
        logger.info(f"Finshed: {entries} lines")


def call_vectorizer(endpoint: str, lines: List[str], lang: str = "en") -> np.array:
    resp = requests.post(endpoint, json={"lines": lines, "lang": lang})
    resp.raise_for_status()

    vect = np.asarray(json.loads(resp.content)["vectors"])
    return vect / np.sqrt((vect**2).sum(axis=1)).reshape((-1, 1))


@cli.command()
@click.argument("files", nargs=-1, required=True)
@click.argument("output_path", nargs=1, required=True)
@click.option("--endpoint", default="http://localhost:8282/vectorize")
def filter_misalignment(files: List[str], output_path: str, endpoint: str) -> None:
    files = fsspec.open_files(files)
    filtered, total = 0, 0

    for file in tqdm(files):
        lang_pair_to_sents = defaultdict(list)
        f_name = os.path.basename(file.path)

        with file as inp_f, fsspec.open(
            os.path.join(output_path, f_name), "w"
        ) as out_f:
            for line in inp_f:
                sp = SentPair.from_json(json.loads(line))
                lang_pair_to_sents[(sp.lang_from, sp.lang_to)].append(
                    (sp.sent_from, sp.sent_to)
                )

            for (from_lang, to_lang), pairs in lang_pair_to_sents.items():
                vects_from = call_vectorizer(
                    endpoint, [f for (f, t) in pairs], from_lang[:2]
                )
                vects_to = call_vectorizer(
                    endpoint, [t for (f, t) in pairs], to_lang[:2]
                )

                cos_sim = (vects_from * vects_to).sum(axis=1)
                filtered_idxs = np.argwhere(cos_sim > 0.9).reshape(-1) # TODO: chose the threshhold based on training data
                for idx in filtered_idxs:
                    sp = SentPair(from_lang, to_lang, pairs[idx][0], pairs[idx][1])
                    out_f.write(json.dumps(sp.to_json()))
                    out_f.write("\n")

                total += len(pairs)
                filtered += len(filtered_idxs)

        logger.info(f"Filtered: {filtered}, total: {total}")


def main() -> None:
    init_logging(logger)
    cli()


if __name__ == "__main__":
    main()

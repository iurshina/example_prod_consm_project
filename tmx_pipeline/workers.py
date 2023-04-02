from __future__ import annotations

import json
import logging
from dataclasses import asdict
from multiprocessing import Pool
from typing import List

import fsspec
from tqdm import tqdm

from tmx_pipeline.utils import (
    KafkaConfig,
    SentPair,
    clean_pair,
    create_consumer_from_topic,
    create_producer,
    parse_tmx_file,
)

logger = logging.getLogger(__name__)


def _parse_tmx_pool_worker(of: fsspec.core.OpenFile, kafka_config: KafkaConfig) -> int:
    """Writes entries extracted from a parsed tmx file into Kafka by a producer (into unprocessed_topic)

    Args:
        of (fsspec.core.OpenFile): tmx file to parse
        kafka_config (KafkaConfig)

    Returns:
        int: number of entries extracted from the file.
    """
    with of as f, create_producer(kafka_config.servers) as p:
        entries = 0
        for pair in parse_tmx_file(f):
            p.produce(
                topic=kafka_config.unprocessed_topic,
                key=None,
                value=json.dumps(asdict(pair)),
            )
            entries += 1

        return entries


def parse_tmx_and_write_to_kafka(
    input_globs: List[str], kafka_config: KafkaConfig, num_proc: int = 4
) -> None:
    """Goes through tmx file(s), calls _parse_tmx_pool_worker in a separate process (creates as many processes as
    files or num_proc if the number of files is greater than num_proc)

    Args:
        input_globs (List[str]): a dir with tmx file(s) to parse
        kafka_config (KafkaConfig)
        num_proc (int, optional): max number of process. Defaults to 4.
    """
    files = fsspec.open_files(input_globs)

    logger.info(
        f"Staring processing, number of files: {len(files)}, files: '{';'.join(input_globs)}'"
    )

    with Pool(min(num_proc, len(files))) as pool:
        futures = []
        for of in tqdm(files):
            fut = pool.apply_async(_parse_tmx_pool_worker, args=(of, kafka_config))
            futures.append(fut)

        logger.info(
            f"Process finished, number of entries: {sum(fut.get() for fut in futures)}"
        )


def _clean_pairs_pool_worker(kafka_config: KafkaConfig) -> int:
    """ Consumes extracted sentence pairs from Kafka (unprocessed_topic),
    cleans them and writes them back to Kafka (filtered_topic).

    Args:
        kafka_config (KafkaConfig)

    Returns:
        int: number of cleaned pairs.
    """
    total = 0

    producer = create_producer(kafka_config.servers)
    consumer = create_consumer_from_topic(
        kafka_config.servers,
        kafka_config.unprocessed_topic,
        kafka_config.group_name,
        timeout=10,
        exit_on_timeout=True,
    )
    with consumer as c, producer as p:
        for pair in c:
            pair = SentPair.from_json(json.loads(pair))
            pair = clean_pair(pair)

            p.produce(
                topic=kafka_config.filtered_topic,
                key=None,
                value=json.dumps(asdict(pair)),
            )

        total += 1

    return total


def clean_pairs_and_write_to_kafka(
    kafka_config: KafkaConfig, num_proc: int = 4
) -> None:
    """Creates num_proc worker processes to clean sentence pairs.

    Args:
        kafka_config (KafkaConfig)
        num_proc (int, optional): number of processes. Defaults to 4.
    """
    with Pool(num_proc) as pool:
        futures = []
        for _ in range(num_proc):
            fut = pool.apply_async(_clean_pairs_pool_worker, args=(kafka_config,))
            futures.append(fut)

        _ = [fut.get() for fut in futures]

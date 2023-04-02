from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from typing import IO, ContextManager, Dict, Iterator, List

import lxml.etree
import numpy as np
import requests
from confluent_kafka import Consumer, KafkaException, Producer

TAG_RE = re.compile(r"<[^>]+>")

logger = logging.getLogger(__name__)


def init_logging(logger: logging.Logger) -> None:
    """Initializes logger.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)


@dataclass
class KafkaConfig:
    servers: str
    group_name: str
    unprocessed_topic: str
    filtered_topic: str


@dataclass
class SentPair:
    lang_from: str
    lang_to: str
    sent_from: str
    sent_to: str

    def to_json(self) -> Dict[str, str]:
        return asdict(self)

    @staticmethod
    def from_json(data: Dict[str, str]) -> SentPair:
        return SentPair(**data)


def clean_sentence(sent: str) -> str:
    """Removes xml/html tags from the sentence.

    Args:
        sent (str): a sentence to clean.

    Returns:
        str: the sentence with tags removed.
    """
    return TAG_RE.sub("", sent)


def clean_pair(sp: SentPair) -> SentPair:
    return replace(
        sp, sent_to=clean_sentence(sp.sent_to), sent_from=clean_sentence(sp.sent_from)
    )


@contextmanager
def create_producer(servers: str) -> ContextManager[Producer]:
    """Creates a Kafka producer.

    Args:
        servers (str): Kafka hosts.
    """
    c = Producer({"bootstrap.servers": servers})
    try:
        yield c
    finally:
        c.flush()


@contextmanager
def create_consumer_from_topic(
    servers: str,
    topic_name: str,
    group_name: str,
    timeout: bool = 1.0,
    exit_on_timeout=False,
) -> ContextManager[Iterator[bytes]]:
    """

    Args:
        servers (str): Kafka hosts.
        topic_name (str): a topic to subscribe to.
        group_name (str): the group this consumer belongs to.
        timeout (bool, optional): timeout to wait to get the message. Defaults to 1.0.
        exit_on_timeout (bool, optional): if True, exits if the timeout exceeded. Defaults to False.

    Raises:
        KafkaException

    Returns:
        ContextManager[Iterator[bytes]]: iterator over messages the consumer got from Kafka.
    """
    c = Consumer(
        {
            "bootstrap.servers": servers,
            "group.id": group_name,
            "auto.offset.reset": "earliest",
            "enable.auto.offset.store": False,
        },
        logger=logger,
    )
    c.subscribe([topic_name])

    def _it():
        while True:
            msg = c.poll(timeout=timeout)
            if msg is None:
                if exit_on_timeout:
                    return
            elif msg.error():
                raise KafkaException(msg.error())
            else:
                c.store_offsets(msg)

                yield msg.value()

    try:
        yield _it()
    finally:
        c.close()


def parse_tmx_file(source: IO[bytes]) -> Iterator[SentPair]:
    """Parses .tmx file and extracts sentence pairs with the corresponding languages.

    Args:
        source (IO[bytes]): input to parse (e.g. file)

    Yields:
        Iterator[SentPair]: iterator of sentence pairs extracted from the source.
    """
    for _, el in lxml.etree.iterparse(source, tag="tu"):
        texts = el.xpath(".//seg/text()")
        langs = el.xpath(".//@xml:lang")

        if len(texts) == len(langs) == 2:
            yield SentPair(
                lang_from=langs[0],
                lang_to=langs[1],
                sent_from=texts[0],
                sent_to=texts[1],
            )
        else:
            logger.warn(
                f"Probably wrong tmx file, unexpected number of `tuv` tags: {lxml.etree.tostring(el, pretty_print=True)}"
            )

        # iterparse builds a tree in memory, delete the element after it was processed
        el.clear()


def call_vectorizer(endpoint: str, lines: List[str], lang: str = "en") -> np.array:
    """Calls the passed endpoint to to obtain embeddings of the passed strings.

    Args:
        endpoint (str): endpoint to call.
        lines (List[str]): lines to embed.
        lang (str, optional): the language of the lines. Defaults to "en".

    Returns:
        np.array: Normalized embeddings of the passed lines.
    """
    resp = requests.post(endpoint, json={"lines": lines, "lang": lang})
    resp.raise_for_status()

    vect = np.asarray(json.loads(resp.content)["vectors"])
    
    return vect / np.sqrt((vect**2).sum(axis=1)).reshape((-1, 1))

# code adopted from https://github.com/facebookresearch/LASER/tree/main/docker

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response
from LASER.source.embed import *
from LASER.source.lib.text_processing import BPEfastApply, Token

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


app = FastAPI()
encoder_path = model_dir / "bilstm.93langs.2018-12-26.pt"
encoder = SentenceEncoder(
    encoder_path,
    max_sentences=None,
    max_tokens=12000,
    sort_kind="mergesort",
    cpu=True,
)


def use_LASER(lines: List[str], lang: str) -> np.ndarray:
    model_dir = Path(__file__).parent / "LASER" / "models"
    bpe_codes_path = model_dir / "93langs.fcodes"
   
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        ifname = tmpdir / "content.txt"
        bpe_fname = tmpdir / "bpe"
        bpe_oname = tmpdir / "out.raw"
        with ifname.open("w") as f:
            for line in lines:
                f.write(line.strip())
                f.write("\n")

        if lang != "--":
            tok_fname = tmpdir / "tok"
            Token(
                str(ifname),
                str(tok_fname),
                lang=lang,
                romanize=True if lang == "el" else False,
                lower_case=True,
                gzip=False,
                verbose=True,
                over_write=False,
            )
            ifname = tok_fname
        BPEfastApply(
            str(ifname),
            str(bpe_fname),
            str(bpe_codes_path),
            verbose=True,
            over_write=False,
        )
        ifname = bpe_fname
        EncodeFile(
            encoder,
            str(ifname),
            str(bpe_oname),
            verbose=True,
            over_write=False,
            buffer_size=10000,
        )
        
        X = np.fromfile(str(bpe_oname), dtype=np.float32)

        return X.reshape((len(lines), -1))


@app.post("/vectorize")
async def vectorize(req: Request):
    req = await req.json()
    lines = req["lines"]
    lang = req["lang"]

    loop = asyncio.get_running_loop()
    emb = await loop.run_in_executor(None, use_LASER, lines, lang)

    return Response(
        content=json.dumps({"vectors": emb.tolist()}), media_type="application/json"
    )


def main() -> None:
    logging.info(f"Running server")

    uvicorn.run(app, host="0.0.0.0", port=80, log_level="info")


if __name__ == "__main__":
    main()

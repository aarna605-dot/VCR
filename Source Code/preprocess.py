"""
preprocess.py

Downloads the VCR split(s) from the MMInstruction/M3IT dataset on Hugging Face,
parses the M3IT instruction-formatted rows into structured fields, and decodes
the base64-encoded images (which already ship with color-coded bounding-box
overlays drawn around referenced objects).

Usage:
    from preprocess import load_split, parse_m3it_row, decode_image

    rows = load_split("val")          # or "train" / "test"
    parsed = [parse_m3it_row(r) for r in rows]
"""

import re
import json
import base64
from io import BytesIO

from PIL import Image
from huggingface_hub import hf_hub_download

DATASET_REPO_ID = "MMInstruction/M3IT"
VCR_FILES = {
    "train": "data/reasoning/vcr/train.jsonl",
    "val": "data/reasoning/vcr/val.jsonl",
    "test": "data/reasoning/vcr/test.jsonl",
}


def load_split(split: str = "val") -> list[dict]:
    """Download and load a raw VCR split ('train', 'val', or 'test') as a list of dicts."""
    if split not in VCR_FILES:
        raise ValueError(f"split must be one of {list(VCR_FILES)}, got {split!r}")

    path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=VCR_FILES[split],
        repo_type="dataset",
    )
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def decode_image(b64_str: str) -> Image.Image:
    """Decode a base64-encoded image string (already box-overlaid) into a PIL image."""
    return Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB")


def parse_m3it_row(row: dict) -> dict:
    """
    Parse a single M3IT VCR row into structured fields.

    M3IT bundles the question, answer choices, and rationale choices into a
    single free-text `inputs` field, and the gold answer+rationale into a
    single free-text `outputs` field ("<answer text> Because <rationale text>").
    This function extracts the structured pieces and recovers the gold label
    indices by matching the free text back against the parsed choice lists.
    """
    text = row["inputs"]

    q_match = re.search(r"Question:\s*\n(.+?)\nAnswer Choices:", text, re.DOTALL)
    question = q_match.group(1).strip() if q_match else None

    answer_choices = re.findall(r"Answer \([A-D]\)\s*(.+?)\s*\n", text)
    answer_choices = [a.strip() for a in answer_choices]

    rationale_choices = re.findall(r"Rationale \([A-D]\)\s*(.+?)\s*\n", text)
    rationale_choices = [r.strip() for r in rationale_choices]

    outputs = row["outputs"].strip()
    if " Because " in outputs:
        ans_text, rat_text = outputs.split(" Because ", 1)
    else:
        ans_text, rat_text = outputs, None
    ans_text = ans_text.strip()
    rat_text = rat_text.strip() if rat_text else None

    answer_label = next(
        (i for i, c in enumerate(answer_choices) if c.strip(" .") == ans_text.strip(" .")),
        None,
    )
    rationale_label = (
        next(
            (i for i, c in enumerate(rationale_choices) if c.strip(" .") == rat_text.strip(" .")),
            None,
        )
        if rat_text
        else None
    )

    return {
        "question": question,
        "answer_choices": answer_choices,
        "rationale_choices": rationale_choices,
        "answer_label": answer_label,
        "rationale_label": rationale_label,
        "image_base64_str": row["image_base64_str"],
        "img_path": row["meta"].get("img_path"),
    }


def make_held_out_split(parsed: list[dict], held_out_size: int = 175, seed: int = 0):
    """Reproduce the fixed held-out / train-pool split used throughout this project."""
    import random

    parsed = list(parsed)
    random.seed(seed)
    random.shuffle(parsed)
    held_out = parsed[:held_out_size]
    train_pool = parsed[held_out_size:]
    return held_out, train_pool


if __name__ == "__main__":
    rows = load_split("val")
    parsed = [parse_m3it_row(r) for r in rows]
    missing = sum(1 for p in parsed if p["answer_label"] is None or p["rationale_label"] is None)
    print(f"{len(parsed)} rows parsed, {missing} failed label recovery")

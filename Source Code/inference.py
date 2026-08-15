"""
inference.py

Single entry point: image + question + candidates -> (A*, R*).

Usage (as a library):
    from model_utils import load_finetuned_model
    from inference import predict

    model, processor = load_finetuned_model("./lora_final_adapter")
    result = predict(model, processor, image, question, answer_choices, rationale_choices)
    print(result)  # {'answer_idx': 1, 'answer': '...', 'rationale_idx': 2, 'rationale': '...'}

Usage (from the command line, on one VCR val row by index):
    python inference.py --adapter ./lora_final_adapter --row_index 0
"""

import argparse

from preprocess import load_split, parse_m3it_row, decode_image
from score import score_candidates, score_rationale
from model_utils import load_finetuned_model


def predict(model, processor, image, question: str, answer_choices: list[str], rationale_choices: list[str]) -> dict:
    """Run the full joint pipeline on a single example and return the predicted answer + rationale."""
    a_idx, a_scores = score_candidates(model, processor, image, question, answer_choices)
    pred_answer_text = answer_choices[a_idx]

    r_idx, r_scores = score_rationale(model, processor, image, question, pred_answer_text, rationale_choices)
    pred_rationale_text = rationale_choices[r_idx]

    return {
        "answer_idx": a_idx,
        "answer": pred_answer_text,
        "answer_scores": a_scores,
        "rationale_idx": r_idx,
        "rationale": pred_rationale_text,
        "rationale_scores": r_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Run joint VCR inference on one example.")
    parser.add_argument("--adapter", type=str, required=True, help="Path to the local LoRA adapter directory.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--row_index", type=int, default=0, help="Index into the parsed split to run inference on.")
    args = parser.parse_args()

    print("Loading model...")
    model, processor = load_finetuned_model(args.adapter)

    print(f"Loading {args.split} split...")
    rows = load_split(args.split)
    row = parse_m3it_row(rows[args.row_index])
    image = decode_image(row["image_base64_str"])

    result = predict(model, processor, image, row["question"], row["answer_choices"], row["rationale_choices"])

    print(f"\nQuestion: {row['question']}")
    print(f"Predicted answer:    {result['answer']}")
    print(f"Predicted rationale: {result['rationale']}")
    if row["answer_label"] is not None:
        print(f"\nGold answer:    {row['answer_choices'][row['answer_label']]}")
        print(f"Gold rationale: {row['rationale_choices'][row['rationale_label']]}")


if __name__ == "__main__":
    main()

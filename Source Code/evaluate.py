"""
evaluate.py

Computes the three official metrics (Q->A, QA->R, Q->AR) on the fixed
175-question held-out slice, and optionally produces test-split predictions
in submission format.

Usage:
    # Evaluate the fine-tuned model on the held-out slice
    python evaluate.py --adapter ./lora_final_adapter --mode held_out

    # Produce test-set predictions
    python evaluate.py --adapter ./lora_final_adapter --mode test --test_sample_size 200
"""

import argparse
import json

from tqdm import tqdm

from preprocess import load_split, parse_m3it_row, decode_image, make_held_out_split
from score import score_candidates, score_rationale
from model_utils import load_finetuned_model


def evaluate_held_out(model, processor, held_out: list[dict], out_path: str = "finetuned_predictions.json"):
    results = []
    correct_a = 0
    correct_ar = 0

    for row in tqdm(held_out):
        img = decode_image(row["image_base64_str"])
        a_idx, a_scores = score_candidates(model, processor, img, row["question"], row["answer_choices"])
        pred_answer_text = row["answer_choices"][a_idx]
        r_idx, r_scores = score_rationale(model, processor, img, row["question"], pred_answer_text, row["rationale_choices"])

        a_correct = a_idx == row["answer_label"]
        r_correct = r_idx == row["rationale_label"]

        results.append({
            "question": row["question"],
            "pred_answer": a_idx,
            "pred_rationale": r_idx,
            "answer_label": row["answer_label"],
            "rationale_label": row["rationale_label"],
        })
        if a_correct:
            correct_a += 1
            if r_correct:
                correct_ar += 1

    n = len(held_out)
    qa_acc = correct_a / n
    qar_acc = correct_ar / n
    qa_r_acc = correct_ar / correct_a if correct_a else 0.0

    print(f"Q->A:   {qa_acc:.4f}")
    print(f"QA->R:  {qa_r_acc:.4f}")
    print(f"Q->AR:  {qar_acc:.4f}")

    json.dump(results, open(out_path, "w"))
    print(f"Saved predictions to {out_path}")
    return {"Q->A": qa_acc, "QA->R": qa_r_acc, "Q->AR": qar_acc}


def run_test_predictions(model, processor, test_sample: list[dict], out_path: str = "test_predictions.json"):
    predictions = []
    for i, row in enumerate(tqdm(test_sample)):
        img = decode_image(row["image_base64_str"])
        a_idx, _ = score_candidates(model, processor, img, row["question"], row["answer_choices"])
        pred_answer_text = row["answer_choices"][a_idx]
        r_idx, _ = score_rationale(model, processor, img, row["question"], pred_answer_text, row["rationale_choices"])

        predictions.append({"id": i, "pred_answer": a_idx, "pred_rationale": r_idx})

        if (i + 1) % 50 == 0:
            json.dump(predictions, open(out_path.replace(".json", "_partial.json"), "w"))

    json.dump(predictions, open(out_path, "w"))
    print(f"Saved {len(predictions)} test predictions to {out_path}")
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["held_out", "test"], default="held_out")
    parser.add_argument("--test_sample_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()

    print("Loading model...")
    model, processor = load_finetuned_model(args.adapter)

    if args.mode == "held_out":
        rows = load_split("val")
        parsed = [parse_m3it_row(r) for r in rows]
        held_out, _train_pool = make_held_out_split(parsed, held_out_size=175, seed=0)
        evaluate_held_out(model, processor, held_out)
    else:
        import random
        rows = load_split("test")
        parsed = [parse_m3it_row(r) for r in rows]
        random.seed(args.seed)
        sample = random.sample(parsed, min(args.test_sample_size, len(parsed)))
        run_test_predictions(model, processor, sample)


if __name__ == "__main__":
    main()

"""
train_lora.py

QLoRA fine-tuning on VCR, using the exact same Yes/No candidate-scoring
prompt format at train time as at inference time (see score.py), so train
and inference distributions match.

Each VCR question expands into 8 binary training examples: 4 for the answer
candidates, 4 for the rationale candidates (rationale examples conditioned
on the GOLD answer during training, even though inference conditions the
rationale scorer on the model's own PREDICTED answer -- see score.py).

Usage:
    python train_lora.py --max_steps 370 --train_size 4500 --output_dir ./lora_final_adapter
"""

import argparse
import random

import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from preprocess import load_split, parse_m3it_row, decode_image, make_held_out_split
from model_utils import load_base_model_and_processor


def build_training_examples(row: dict) -> list[dict]:
    """Expand one parsed VCR row into 8 binary Yes/No training examples."""
    examples = []
    for i, cand in enumerate(row["answer_choices"]):
        label = "Yes" if i == row["answer_label"] else "No"
        prompt = (
            f"Question: {row['question']}\n"
            f"Proposed answer: {cand}\n"
            "Is this proposed answer correct? Answer strictly Yes or No."
        )
        examples.append({"image_base64_str": row["image_base64_str"], "prompt": prompt, "label": label})

    gold_answer_text = row["answer_choices"][row["answer_label"]]
    for i, rat in enumerate(row["rationale_choices"]):
        label = "Yes" if i == row["rationale_label"] else "No"
        prompt = (
            f"Question: {row['question']}\n"
            f"Answer: {gold_answer_text}\n"
            f"Proposed rationale: {rat}\n"
            "Does this rationale correctly explain the answer? Answer strictly Yes or No."
        )
        examples.append({"image_base64_str": row["image_base64_str"], "prompt": prompt, "label": label})

    return examples


class VCRTrainingDataset(Dataset):
    """
    Wraps a list of {image_base64_str, prompt, label} examples for the Trainer.

    Note: pixel_values / image_grid_thw are NOT batch-dimensioned the same way
    text tensors are -- Qwen2.5-VL's processor returns them as flattened
    patches with no leading batch dim. __getitem__ and collate_fn below must
    NOT add/strip a batch dimension on those two keys, or the vision tower's
    internal reshape logic breaks (this was the source of a real bug during
    development -- see README for details).
    """

    def __init__(self, examples: list[dict], processor):
        self.examples = examples
        self.processor = processor

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        img = decode_image(ex["image_base64_str"])

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": ex["prompt"]},
        ]}]
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + ex["label"]

        inputs = self.processor(text=[full_text], images=[img], return_tensors="pt")
        prompt_only = self.processor(text=[prompt_text], images=[img], return_tensors="pt")
        prompt_len = prompt_only["input_ids"].shape[1]

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask the prompt, train only on the Yes/No completion token

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "labels": labels,
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }


def collate_fn(batch):
    """Batch size is always 1 here; only unsqueeze the text-side tensors (see class docstring)."""
    item = batch[0]
    out = {}
    for k, v in item.items():
        if k in ("pixel_values", "image_grid_thw"):
            out[k] = v
        else:
            out[k] = v.unsqueeze(0)
    return out


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune on VCR.")
    parser.add_argument("--train_size", type=int, default=4500, help="Number of Yes/No examples to sample for training.")
    parser.add_argument("--max_steps", type=int, default=370, help="Hard cap on optimizer steps.")
    parser.add_argument("--grad_accum", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./lora_final_adapter")
    parser.add_argument("--checkpoint_dir", type=str, default="./lora_real_checkpoints")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    print("Loading base model...")
    model, processor = load_base_model_and_processor()

    print("Applying LoRA...")
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()  # required to fit a single T4's memory budget

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading and parsing training data...")
    rows = load_split("val")  # the "val" split was used as the held-in/held-out source throughout this project
    parsed = [parse_m3it_row(r) for r in rows]
    _held_out, train_pool = make_held_out_split(parsed, held_out_size=175, seed=0)

    training_examples = []
    for row in train_pool:
        training_examples.extend(build_training_examples(row))
    print(f"{len(training_examples)} total training examples available from {len(train_pool)} questions")

    random.seed(args.seed)
    sample = random.sample(training_examples, min(args.train_size, len(training_examples)))
    dataset = VCRTrainingDataset(sample, processor)
    print(f"Training on {len(sample)} examples")

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=5,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()

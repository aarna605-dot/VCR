# VCR Joint Answer + Rationale System

## 1. Overview

A multimodal system that performs Visual Commonsense Reasoning (VCR): given
an image and a question, it selects the correct answer from 4 choices, then
selects the rationale that best explains that answer from 4 more choices.
Predictions are scored jointly as **Q→AR** (both answer and rationale must be
correct on the same example) — the task's primary and hardest metric.

The core design choice is that the rationale-scoring stage is conditioned on
the model's own **predicted** answer (not the gold answer), making this a
genuine joint pipeline rather than two independent classifiers whose outputs
are combined afterward.

## 2. Tech Stack & Model

- **Base model:** [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) — a 3B-parameter vision-language model
- **Quantization:** 4-bit NF4 (via `bitsandbytes`), bfloat16 compute precision
- **Fine-tuning:** QLoRA (`peft`) — rank 16, alpha 32, dropout 0.05, applied to `q_proj`/`v_proj`/`k_proj`/`o_proj`
- **Framework:** PyTorch, Hugging Face `transformers`, `peft`, `bitsandbytes`
- **Dataset:** [`MMInstruction/M3IT`](https://huggingface.co/datasets/MMInstruction/M3IT), `reasoning/vcr` subset — images ship with pre-drawn color-coded bounding boxes around referenced objects
- **Compute:** Single Kaggle T4 GPU (free tier)

## 3. What Was Done (in order)

1. **Environment setup** — Kaggle notebook, GPU (T4), installed `transformers`, `accelerate`, `bitsandbytes`, `peft`, `qwen-vl-utils`.
2. **Dataset acquisition** — downloaded the VCR `val` split from `MMInstruction/M3IT` via `huggingface_hub` (5,000 rows).
3. **Parsing** — extracted question, answer choices, rationale choices, and gold labels from M3IT's free-text `inputs`/`outputs` fields via regex; decoded base64 images (0/5000 failed label recovery).
4. **Held-out split** — reserved a fixed 175-question held-out slice (seed=0), untouched until final evaluation. Remaining ~4,825 questions used as the training pool.
5. **Model loading** — loaded Qwen2.5-VL-3B-Instruct in 4-bit NF4 quantization, bfloat16 compute, pinned to a single GPU.
6. **Candidate scoring function** — built Yes/No constrained-prompt scoring for both answer and rationale candidates (softmax over "Yes"/"No" token logits, argmax over 4 candidates).
7. **Zero-shot baseline** — ran the untrained model over an 800-question sample of the training pool to establish a floor.
8. **QLoRA fine-tuning** — trained LoRA adapters on 2,220 of 38,600 available Yes/No training examples (370 optimizer steps, ~5.5 hours on a single T4), using the same prompt format as inference.
9. **Full evaluation** — computed Q→A, QA→R, and Q→AR on the untouched 175-question held-out slice, for both the zero-shot and fine-tuned models.
10. **Error analysis** — bucketed held-out accuracy by question type (what/why/how/other/what-would-happen) and reviewed qualitative failure examples.
11. **Test predictions** — ran the final fine-tuned pipeline over a 200-question sample of the VCR test split, in submission format, and spot-checked results by hand.
12. **Report & packaging** — wrote a ≤4-page report and packaged all code into standalone, reusable scripts.

## 4. Final Evaluation Metrics

All metrics computed on the fixed 175-question held-out slice (zero-shot) / same slice (fine-tuned), and an 800-question sample for the zero-shot baseline where noted.

| Metric | Zero-Shot Baseline | Fine-Tuned (LoRA) | Change |
|---|---|---|---|
| **Q→A** | 0.6188 | 0.7029 | +8.41 pp |
| **QA→R** | 0.6364 | 0.6911 | +5.47 pp |
| **Q→AR** (primary) | 0.3938 | 0.4857 | +9.19 pp |

Fine-tuning on just **5.7% of available training data** (2,220 / 38,600 examples) improved all three official metrics, most notably the primary Q→AR metric by +9.19 percentage points.

### Error breakdown by question type (fine-tuned, held-out)

| Question Type | Accuracy | n |
|---|---|---|
| what | 0.80 | 69 |
| other | 0.68 | 34 |
| why | 0.67 | 57 |
| how | 0.50 | 10 |
| what-would-happen | 0.40 | 5 |

## 5. Links

- **Trained model weights (LoRA adapter):** [funezocode/lora-vcr-finetuned](https://huggingface.co/funezocode/lora-vcr-finetuned)
- **Test set predictions:** [`test_predictions.json`](https://huggingface.co/funezocode/lora-vcr-finetuned/blob/main/test_predictions.json)
- **Full report:** `report.docx` (in this repository)
- **Kaggle notebook:** `notebook.ipynb` (in this repository)

## 6. Repository Structure
repo/
README.md - this file
requirements.txt
preprocess.py - dataset download, M3IT row parsing, image decoding
score.py - score_candidates(), score_rationale()
model_utils.py - base/fine-tuned model loading
inference.py - single entry point: image+question -> (A*, R*)
train_lora.py - QLoRA fine-tuning
evaluate.py - held-out metric computation + test predictions
notebook.ipynb - full Kaggle notebook (all cells + outputs)
report.docx - project report
predictions/
test_predictions.json - 200-row test-set predictions

## 7. Setup & Reproduction

```bash
pip install -r requirements.txt
```

```python
from huggingface_hub import snapshot_download
adapter_path = snapshot_download(repo_id="funezocode/lora-vcr-finetuned") + "/lora_final_adapter"
```

```bash
python evaluate.py --adapter <adapter_path> --mode held_out
```

## 8. Known Implementation Notes

- **bfloat16, not float16:** float16 compute precision under 4-bit quantization produced NaN losses on this model from the first training step; bfloat16 resolved this.
- **Qwen2.5-VL's `pixel_values`/`image_grid_thw` have no batch dimension by default** — a custom `Dataset`/`collate_fn` must not unsqueeze or index these the same way as `input_ids`/`attention_mask`, or the vision tower's internal reshape logic breaks.

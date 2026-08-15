# VCR Joint Answer + Rationale System

A multimodal system that answers a question about an image (4-way choice), then
selects the rationale that best explains that answer (4-way choice), scored
jointly as Q→AR.

## Model

- **Backbone:** `Qwen/Qwen2.5-VL-3B-Instruct`, loaded in 4-bit (NF4) quantization, bfloat16 compute.
- **Fine-tuning:** QLoRA (r=16, alpha=32, dropout=0.05) on the attention projections (`q_proj`, `v_proj`, `k_proj`, `o_proj`). 7,372,800 trainable parameters (0.196% of the 3.76B total).
- **Dataset:** [`MMInstruction/M3IT`](https://huggingface.co/datasets/MMInstruction/M3IT), `reasoning/vcr` subset — a pre-parsed instruction-tuning version of VCR whose images already have color-coded bounding boxes drawn around referenced objects, so no custom box-overlay code was needed.
- **Trained adapter weights:** [`funezocode/lora-vcr-finetuned`](https://huggingface.co/funezocode/lora-vcr-finetuned) (`lora_final_adapter/`).

## Data volume

Fine-tuned on **2,220 of 38,600 available training examples** (370 optimizer
steps at effective batch size 6) — capped by Kaggle's free-tier GPU time
budget, not by design. See the report (`report.pdf`) for the full discussion.

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU with at least ~8GB VRAM (developed and tested on a single Kaggle T4).

## Reproducing the reported numbers

**1. Download the fine-tuned adapter:**

```python
from huggingface_hub import snapshot_download
adapter_path = snapshot_download(repo_id="funezocode/lora-vcr-finetuned") + "/lora_final_adapter"
```

**2. Run inference on a single example:**

```bash
python inference.py --adapter <adapter_path> --split val --row_index 0
```

**3. Reproduce the held-out evaluation (Q→A, QA→R, Q→AR):**

```bash
python evaluate.py --adapter <adapter_path> --mode held_out
```

Expected output on the fixed 175-question held-out slice (seed=0):

| Metric | Zero-shot | Fine-tuned |
|---|---|---|
| Q→A | 0.6200 | 0.7029 |
| QA→R | — | 0.6911 |
| Q→AR | — | 0.4857 |

**4. Reproduce test-set predictions:**

```bash
python evaluate.py --adapter <adapter_path> --mode test --test_sample_size 200
```

## Re-running training from scratch

```bash
python train_lora.py --train_size 4500 --max_steps 370 --output_dir ./lora_final_adapter
```

On a single T4, this takes roughly 5.5 hours at the observed throughput of
~0.22 examples/sec (batch size 1, gradient accumulation 6). Checkpoints save
every 50 steps to `--checkpoint_dir`, so an interrupted run can be resumed
from the last checkpoint via `PeftModel.from_pretrained(base_model, checkpoint_path)`.

## Repository structure

```
repo/
  README.md                    - this file
  requirements.txt
  preprocess.py                 - dataset download, M3IT row parsing, image decoding
  score.py                      - score_candidates(), score_rationale()
  model_utils.py                - shared base/fine-tuned model loading
  inference.py                  - single entry point: image+question -> (A*, R*)
  train_lora.py                 - QLoRA fine-tuning
  evaluate.py                   - held-out metric computation + test predictions
  predictions/
    test_predictions.json       - 200-row test-set predictions
  report.pdf / report.docx      - project report
```

## Known implementation notes

- **bfloat16, not float16:** float16 compute precision combined with 4-bit
  NF4 quantization produced NaN losses on this model from the first training
  step. Switching to bfloat16 resolved this; see `model_utils.py`.
- **Qwen2.5-VL's image tensors have no batch dimension by default** —
  `pixel_values` and `image_grid_thw` are flattened-patch tensors and must
  **not** be unsqueezed/indexed the same way as `input_ids`/`attention_mask`
  in a custom `Dataset`/`collate_fn`. Doing so breaks the vision tower's
  internal patch-merging reshape. See the comments in `train_lora.py`.
- No hardcoded absolute paths (e.g. `/kaggle/input/...`) are used anywhere in
  this repository; all paths are passed as arguments or resolved via
  `huggingface_hub`.

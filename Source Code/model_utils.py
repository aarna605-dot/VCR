"""
model_utils.py

Shared model-loading utility for the VCR joint answer + rationale system.
Loads Qwen2.5-VL-3B-Instruct in 4-bit (NF4) quantization, pinned to a single
GPU, with bfloat16 compute precision (required for numerical stability under
4-bit quantization -- float16 produces NaN losses on this model).
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_base_model_and_processor(max_pixels: int = 512 * 28 * 28, min_pixels: int = 256 * 28 * 28):
    """Load the quantized base model + processor. Does not attach a LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    return model, processor


def load_finetuned_model(adapter_path: str, **kwargs):
    """Load the base model and attach the fine-tuned LoRA adapter from a local path."""
    from peft import PeftModel

    base_model, processor = load_base_model_and_processor(**kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, processor

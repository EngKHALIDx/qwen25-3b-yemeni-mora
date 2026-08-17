#!/usr/bin/env python3
"""Run a chat prompt with Qwen2.5-3B-Instruct and a trained GL-log-MoRA adapter.

The script assumes the local PEFT fork is installed with `pip install -e ./peft-mora`.
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True, help="Path to the saved PEFT adapter")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model identifier or local path",
    )
    parser.add_argument("--prompt", required=True, help="User prompt")
    parser.add_argument("--system", default="أجب باللغة العربية بوضوح ودقة.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)

    model_kwargs = {"device_map": "auto"}
    if not args.no_4bit and torch.cuda.is_available():
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    base = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(outputs[0][inputs.shape[-1] :], skip_special_tokens=True)
    print(answer.strip())


if __name__ == "__main__":
    main()

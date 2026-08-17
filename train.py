#!/usr/bin/env python3
"""
Train Qwen2.5-3B-Instruct with the local MoRA-enabled PEFT fork.

This script is intentionally independent from the original DeepSpeed/LLaMA
training entry point. It reads JSONL/JSONL.GZ records whose main field is
``messages`` and applies the Qwen chat template before supervised fine-tuning.
Only assistant turns contribute to the loss; system and user turns remain in
the context and are masked with -100.

The script is designed for one-GPU Google Colab sessions. QLoRA is enabled by
default because a 4-bit NF4 backbone plus MoRA adapters is the most practical
configuration for a free Colab T4/V100. The local ``peft-mora`` package must
be installed before this file is run.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


LOGGER = logging.getLogger("qwen_mora")
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def configure_logging() -> None:
    """Use a readable log format in both notebooks and normal terminals."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA/MoRA supervised fine-tuning for Qwen2.5-3B-Instruct."
    )

    # Model and data.
    parser.add_argument(
        "--model_name_or_path",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Hugging Face model identifier or a local model directory.",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        help="Path to qwen25_3b_semantic_colab.jsonl or .jsonl.gz. Can also be set in --config.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON file whose keys match this parser's arguments; command-line values override its defaults.",
    )
    parser.add_argument(
        "--output_dir",
        default="./qwen25_3b_mora_adapter",
        help="Directory for the adapter, tokenizer, checkpoints, and metadata.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length. The supplied audited dataset is <=1024 tokens.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=-1,
        help="Optional smoke-test limit; -1 trains on every record.",
    )
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.0,
        help="Optional deterministic validation fraction, e.g. 0.01; default trains on all records.",
    )

    # PEFT/MoRA.
    parser.add_argument("--r", type=int, default=64, help="MoRA/LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=DEFAULT_TARGET_MODULES,
        help="Qwen projection modules to adapt.",
    )
    parser.add_argument(
        "--use_mora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the MoRA extension from the local PEFT fork.",
    )
    parser.add_argument(
        "--mora_type",
        type=int,
        default=6,
        choices=[1, 2, 3, 4, 6],
        help="MoRA implementation variant. Type 6 is the RoPE-based small-rank option.",
    )
    parser.add_argument(
        "--use_gl_log_mora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the GL-log-MoRA Q transform and log-determinant regularizer.",
    )
    parser.add_argument(
        "--gl_log_lambda",
        type=float,
        default=0.01,
        help="Lambda in -lambda*logdet(Q^T Q + delta I).",
    )
    parser.add_argument(
        "--gl_log_delta",
        type=float,
        default=1e-3,
        help="Positive numerical stabilizer delta for the GL-log-MoRA regularizer.",
    )
    parser.add_argument(
        "--gl_log_q_lr",
        type=float,
        default=3e-4,
        help="Separate learning rate for GL-log-MoRA Q parameters.",
    )

    # Colab memory and numerical settings.
    parser.add_argument(
        "--use_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the frozen backbone in 4-bit NF4 (recommended on free Colab).",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use bfloat16 when the GPU supports it; T4 normally uses fp16 instead.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="sdpa",
        help="Attention backend. Use flash_attention_2 only after installing flash-attn.",
    )

    # Trainer hyperparameters.
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report_to",
        default="none",
        help="none, wandb, tensorboard, or a comma-separated Transformers value.",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Usually unnecessary for Qwen2.5; enable only for a custom model directory.",
    )

    # Load JSON defaults before the final parse, so explicit CLI arguments win.
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", default=None)
    config_probe_args, _ = config_probe.parse_known_args()
    if config_probe_args.config:
        with open(config_probe_args.config, "r", encoding="utf-8") as handle:
            config_values = json.load(handle)
        if not isinstance(config_values, dict):
            parser.error("--config must contain a JSON object")
        valid_dests = {action.dest for action in parser._actions}
        unknown = sorted(set(config_values) - valid_dests)
        if unknown:
            parser.error(f"Unknown keys in --config: {unknown}")
        parser.set_defaults(**config_values)

    args = parser.parse_args()
    if not args.data_path:
        parser.error("--data_path is required, either on the command line or inside --config")
    if args.max_seq_length <= 0:
        parser.error("--max_seq_length must be positive")
    if args.validation_size < 0 or args.validation_size >= 1:
        parser.error("--validation_size must be in [0, 1)")
    if args.use_4bit and not torch.cuda.is_available():
        LOGGER.warning("4-bit loading normally requires CUDA; a CPU-only run will fail when loading the model.")
    return args


def _as_token_list(value: Any) -> List[int]:
    """Normalize tokenizer output to a plain Python list of token IDs."""

    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(x) for x in value]


def _prefix_matches(full_ids: Sequence[int], prefix_ids: Sequence[int]) -> bool:
    return len(prefix_ids) <= len(full_ids) and list(full_ids[: len(prefix_ids)]) == list(prefix_ids)


def assistant_label_spans(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    full_ids: Sequence[int],
) -> List[tuple[int, int]]:
    """Return token spans belonging to assistant messages.

    Qwen's chat template is deterministic. Tokenizing the conversation up to
    each assistant turn gives a prefix of the full conversation, which lets us
    mask system/user tokens without depending on hard-coded special-token IDs.
    """

    spans: List[tuple[int, int]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        before = _as_token_list(
            tokenizer.apply_chat_template(
                list(messages[:index]),
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        through = _as_token_list(
            tokenizer.apply_chat_template(
                list(messages[: index + 1]),
                tokenize=True,
                add_generation_prompt=False,
            )
        )

        if _prefix_matches(full_ids, before) and _prefix_matches(full_ids, through):
            start, end = len(before), len(through)
            if end > start:
                spans.append((start, end))
            continue

        # Some custom templates add a generation marker when an assistant turn
        # is opened. Try the explicit generation-prompt prefix as a fallback.
        generation_prefix = _as_token_list(
            tokenizer.apply_chat_template(
                list(messages[:index]),
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        if _prefix_matches(full_ids, generation_prefix):
            end = len(through) if _prefix_matches(full_ids, through) else len(full_ids)
            if end > len(generation_prefix):
                spans.append((len(generation_prefix), end))

    return spans


def preprocess_record(
    record: Dict[str, Any], tokenizer: Any, max_seq_length: int
) -> Dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Each training record must contain a non-empty messages list")

    normalized: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Every messages item must be an object")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported chat role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Every chat message must have non-empty string content")
        normalized.append({"role": role, "content": content})

    if not any(m["role"] == "assistant" for m in normalized):
        raise ValueError("Every training record must contain an assistant message")

    input_ids = _as_token_list(
        tokenizer.apply_chat_template(
            normalized,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if len(input_ids) > max_seq_length:
        raise ValueError(
            "A record exceeds max_seq_length="
            f"{max_seq_length}; refusing to truncate because truncation could remove legal meaning. "
            "Use the audited 1024-token dataset or increase --max_seq_length."
        )

    labels = [-100] * len(input_ids)
    spans = assistant_label_spans(tokenizer, normalized, input_ids)
    if not spans:
        raise ValueError(
            "Could not locate the assistant span in the tokenizer chat template; "
            "refusing to train with an incorrect loss mask."
        )
    for start, end in spans:
        labels[start:end] = input_ids[start:end]

    if not any(label != -100 for label in labels):
        raise ValueError("The record produced no supervised assistant tokens")

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "length": len(input_ids),
    }


def load_and_tokenize_dataset(
    data_path: str,
    tokenizer: Any,
    max_seq_length: int,
    seed: int,
    max_train_samples: int,
    validation_size: float,
) -> tuple[Dataset, Dataset | None]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Training data was not found: {path}")
    if path.suffix not in {".gz", ".jsonl", ".json"} and not path.name.endswith(".jsonl.gz"):
        LOGGER.warning("The data path does not have a usual JSONL/JSON.GZ suffix: %s", path)

    LOGGER.info("Loading JSONL dataset from %s", path)
    dataset = load_dataset("json", data_files={"train": str(path)}, split="train")
    LOGGER.info("Loaded %d raw records", len(dataset))

    # Keep every row unless the caller explicitly requests a smoke-test limit.
    if max_train_samples > 0:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_train_samples, len(dataset))))
        LOGGER.info("Smoke-test mode: selected %d records", len(dataset))

    dataset = dataset.map(
        lambda record: preprocess_record(record, tokenizer, max_seq_length),
        remove_columns=dataset.column_names,
        desc="Applying Qwen chat template and assistant-only loss mask",
    )

    if validation_size > 0:
        split = dataset.train_test_split(test_size=validation_size, seed=seed)
        train_dataset, eval_dataset = split["train"], split["test"]
        LOGGER.info("Tokenized train=%d, validation=%d", len(train_dataset), len(eval_dataset))
        return train_dataset, eval_dataset

    LOGGER.info("Tokenized train=%d (all records)", len(dataset))
    return dataset, None


class ChatDataCollator:
    """Pad model inputs while keeping ``length`` only for the Trainer sampler."""

    def __init__(self, tokenizer: Any, model: Any):
        self._collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
            return_tensors="pt",
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        model_features = [
            {key: value for key, value in feature.items() if key != "length"}
            for feature in features
        ]
        return self._collator(model_features)


def resolve_dtype(requested_bf16: bool) -> tuple[torch.dtype, bool]:
    if requested_bf16:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16, True
        LOGGER.warning("BF16 was requested but this GPU does not report BF16 support; falling back to FP16.")
    return torch.float16, False


def model_loading_kwargs(args: argparse.Namespace, dtype: torch.dtype) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation != "auto":
        kwargs["attn_implementation"] = args.attn_implementation

    if args.use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else "auto"
    return kwargs


def build_training_args(
    args: argparse.Namespace,
    use_bf16: bool,
    has_eval: bool,
) -> TrainingArguments:
    report_to = [] if args.report_to.lower() == "none" else args.report_to.split(",")
    values: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "logging_strategy": "steps",
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "evaluation_strategy": "steps" if has_eval else "no",
        "eval_steps": args.save_steps if has_eval else None,
        "fp16": not use_bf16,
        "bf16": use_bf16,
        "optim": "paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        "gradient_checkpointing": args.gradient_checkpointing,
        "group_by_length": True,
        "length_column_name": "length",
        "dataloader_num_workers": args.dataloader_num_workers,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "report_to": report_to,
        "run_name": args.run_name,
        "seed": args.seed,
        "max_grad_norm": 0.3 if args.use_4bit else 1.0,
    }

    # Transformers renamed evaluation_strategy to eval_strategy in newer
    # releases. Keep the script usable across standard Colab image versions.
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters and "evaluation_strategy" not in parameters:
        values["eval_strategy"] = values.pop("evaluation_strategy")
    if "gradient_checkpointing_kwargs" in parameters and args.gradient_checkpointing:
        values["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if "tf32" in parameters:
        values["tf32"] = bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)

    return TrainingArguments(**values)


class GLLogMoRATrainer(Trainer):
    """Trainer that adds the GL-log-MoRA regularizer and separates Q learning rate."""

    def __init__(self, *args: Any, gl_log_q_lr: float = 3e-4, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.gl_log_q_lr = float(gl_log_q_lr)

    def _gl_log_mora_regularization(self, model: Any) -> torch.Tensor | None:
        terms: List[torch.Tensor] = []
        for module in model.modules():
            regularizer = getattr(module, "gl_log_mora_regularizer", None)
            if not callable(regularizer):
                continue
            active_adapters = getattr(module, "active_adapters", [])
            if callable(active_adapters):
                active_adapters = active_adapters()
            flags = getattr(module, "use_gl_log_mora", {})
            for adapter_name in active_adapters:
                if isinstance(flags, dict) and flags.get(adapter_name, False):
                    terms.append(regularizer(adapter_name).float())
        if not terms:
            return None
        return torch.stack(terms).mean()

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        outputs = model(**inputs)
        if hasattr(outputs, "loss"):
            task_loss = outputs.loss
        elif isinstance(outputs, dict):
            task_loss = outputs["loss"]
        else:
            task_loss = outputs[0]
        regularizer = self._gl_log_mora_regularization(model)
        loss = task_loss + regularizer.to(task_loss.dtype) if regularizer is not None else task_loss
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is not None:
            return
        q_parameters = []
        base_parameters = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "lora_Q" in name:
                q_parameters.append(parameter)
            else:
                base_parameters.append((name, parameter))

        if not q_parameters or self.gl_log_q_lr <= 0:
            return super().create_optimizer()

        decay_parameters = []
        no_decay_parameters = []
        for name, parameter in base_parameters:
            if name.endswith("bias") or "norm" in name.lower():
                no_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)

        optimizer_grouped_parameters = [
            {"params": decay_parameters, "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate},
            {"params": no_decay_parameters, "weight_decay": 0.0, "lr": self.args.learning_rate},
            {"params": q_parameters, "weight_decay": 0.0, "lr": self.gl_log_q_lr},
        ]
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        LOGGER.info("GL-log-MoRA Q optimizer group: %d parameters, lr=%s", len(q_parameters), self.gl_log_q_lr)


def main() -> None:
    configure_logging()
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Fail early if the normal PEFT package was imported instead of the local
    # MoRA fork; silently training ordinary LoRA would produce the wrong artifact.
    if args.use_mora and not hasattr(LoraConfig, "__dataclass_fields__"):
        raise RuntimeError("Could not inspect the local MoRA-enabled LoraConfig")
    probe_fields = getattr(LoraConfig, "__dataclass_fields__", {})
    if args.use_mora and "use_mora" not in probe_fields:
        raise RuntimeError(
            "The imported PEFT package does not expose use_mora. Install the local fork with: "
            "pip install -e ./peft-mora"
        )

    dtype, use_bf16 = resolve_dtype(args.bf16)
    LOGGER.info("Loading tokenizer: %s", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.chat_template is None:
        raise ValueError("The tokenizer has no chat_template; Qwen2.5-Instruct is required for this script.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    LOGGER.info("Loading model with %s and %s attention", dtype, args.attn_implementation)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_loading_kwargs(args, dtype),
    )
    model.config.use_cache = False

    if args.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing:
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        use_mora=args.use_mora,
        mora_type=args.mora_type,
        use_gl_log_mora=args.use_gl_log_mora,
        gl_log_lambda=args.gl_log_lambda,
        gl_log_delta=args.gl_log_delta,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset, eval_dataset = load_and_tokenize_dataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        validation_size=args.validation_size,
    )

    data_collator = ChatDataCollator(tokenizer=tokenizer, model=model)
    training_args = build_training_args(args, use_bf16=use_bf16, has_eval=eval_dataset is not None)
    trainer = GLLogMoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        gl_log_q_lr=args.gl_log_q_lr,
    )

    LOGGER.info("Starting training: %d records", len(train_dataset))
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    metadata = {
        "base_model": args.model_name_or_path,
        "data_path": str(Path(args.data_path).resolve()),
        "max_seq_length": args.max_seq_length,
        "use_4bit": args.use_4bit,
        "use_mora": args.use_mora,
        "mora_type": args.mora_type,
        "use_gl_log_mora": args.use_gl_log_mora,
        "gl_log_lambda": args.gl_log_lambda,
        "gl_log_delta": args.gl_log_delta,
        "gl_log_q_lr": args.gl_log_q_lr,
        "rank": args.r,
        "target_modules": args.target_modules,
        "assistant_only_loss": True,
        "train_records": len(train_dataset),
        "validation_records": len(eval_dataset) if eval_dataset is not None else 0,
        "seed": args.seed,
    }
    with open(Path(args.output_dir) / "training_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Adapter and tokenizer saved to %s", args.output_dir)


if __name__ == "__main__":
    main()

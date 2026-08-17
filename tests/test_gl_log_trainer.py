import sys
from pathlib import Path

import torch
from transformers import TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "peft-mora" / "src"))
sys.path.insert(0, str(ROOT))

from test_gl_log_integration import TinyConfig, TinyCausalLM
from peft import LoraConfig, get_peft_model
from train import GLLogMoRATrainer


def main():
    torch.manual_seed(17)
    model = get_peft_model(
        TinyCausalLM(TinyConfig()),
        LoraConfig(
            r=2,
            lora_alpha=2,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            use_mora=True,
            use_gl_log_mora=True,
            gl_log_lambda=0.01,
            gl_log_delta=1e-3,
        ),
    )
    args = TrainingArguments(
        output_dir=str(ROOT / "tests" / "_trainer_output"),
        per_device_train_batch_size=1,
        report_to=[],
        max_steps=1,
        learning_rate=2e-4,
        remove_unused_columns=False,
    )
    trainer = GLLogMoRATrainer(model=model, args=args, gl_log_q_lr=3e-4)
    reg = trainer._gl_log_mora_regularization(model)
    assert reg is not None and torch.isfinite(reg)
    trainer.create_optimizer()
    optimizer = trainer.optimizer
    q_params = {id(p) for n, p in model.named_parameters() if "lora_Q" in n}
    q_groups = [group for group in optimizer.param_groups if any(id(p) in q_params for p in group["params"])]
    assert len(q_groups) == 1
    assert abs(q_groups[0]["lr"] - 3e-4) < 1e-10
    print("trainer path passed: regularizer finite and Q lr group is 3e-4")


if __name__ == "__main__":
    main()

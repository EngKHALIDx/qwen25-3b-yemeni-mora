from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
issues: list[tuple[str, str, str]] = []

# Syntax and JSON validation for project-owned execution files.
for path in [ROOT / "train.py", ROOT / "train_qwen_mora.py", ROOT / "training_utils.py", ROOT / "verify_bundle.py", ROOT / "train_legacy_deepspeed.py"]:
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        print(f"SYNTAX PASS {path.relative_to(ROOT)}")
    except Exception as exc:
        issues.append(("CRITICAL", str(path.relative_to(ROOT)), f"syntax: {exc}"))
        print(f"SYNTAX FAIL {path.relative_to(ROOT)}: {exc}")

notebook = ROOT / "colab_train_qwen25_3b_mora.ipynb"
try:
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    print(f"NOTEBOOK JSON PASS cells={len(nb.get('cells', []))}")
except Exception as exc:
    issues.append(("CRITICAL", notebook.name, f"invalid JSON: {exc}"))
    nb = {"cells": []}

cfg_path = ROOT / "configs" / "qwen25_3b_mora.json"
config = json.loads(cfg_path.read_text(encoding="utf-8"))
print("CONFIG keys:", ", ".join(sorted(config)))

train_text = (ROOT / "train_qwen_mora.py").read_text(encoding="utf-8")
train_lines = train_text.splitlines()

# Verify CLI/config correspondence.
parser = ast.parse(train_text)
cli_names: set[str] = set()
for node in ast.walk(parser):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                cli_names.add(arg.value[2:].replace("-", "_"))
for key in config:
    if key not in cli_names:
        issues.append(("HIGH", "configs/qwen25_3b_mora.json", f"config key {key!r} has no matching CLI argument"))
for name in sorted(cli_names):
    if name not in config and name not in {"config", "data_path", "output_dir", "resume_from_checkpoint", "max_train_samples", "max_steps", "run_name", "report_to", "bf16", "trust_remote_code", "attn_implementation", "validation_size", "seed"}:
        print(f"INFO CLI-only argument {name}")

# Inspect fragile integration points.
checks = {
    "local_peft_import_guard": '"use_mora" not in probe_fields',
    "editable_install_hint": "pip install -e ./peft-mora",
    "assistant_only_loss": "labels[start:end] = input_ids[start:end]",
    "length_column": '"length"',
    "remove_unused_columns_false": '"remove_unused_columns": False',
    "save_model": "trainer.save_model",
    "tokenizer_save": "tokenizer.save_pretrained",
    "resume": "resume_from_checkpoint=args.resume_from_checkpoint",
}
for name, needle in checks.items():
    found = needle in train_text
    print(f"CHECK {name}: {'PASS' if found else 'FAIL'}")
    if not found:
        issues.append(("HIGH", "train_qwen_mora.py", f"missing integration point: {needle}"))

# Notebook command consistency and execution-state risks.
nb_text = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))
for needle in ["pip install -e ./peft-mora", "sys.path", "train.py", "download", "zip"]:
    print(f"NOTEBOOK CHECK {needle}: {'PASS' if needle in nb_text else 'MISSING'}")
if "pip install -e ./peft-mora" in nb_text and "sys.path.insert" not in nb_text and "sys.path.append" not in nb_text:
    issues.append(("HIGH", "colab_train_qwen25_3b_mora.ipynb", "editable peft install is not followed by a path refresh or kernel restart"))

# Detect accidental duplicated or conflicting entry points.
for path in [ROOT / "train.py", ROOT / "train_qwen_mora.py"]:
    text = path.read_text(encoding="utf-8")
    print(f"ENTRYPOINT {path.name}: main_guard={'if __name__ == \"__main__\"' in text}, lines={len(text.splitlines())}")

# Check metadata claims against actual behavior.
if '"assistant_only_loss": True' in train_text and "assistant_only_loss" not in config:
    print("INFO assistant-only loss is hard-coded, not configurable")

print("\nISSUES")
for severity, where, detail in issues:
    print(f"{severity}\t{where}\t{detail}")
print(f"TOTAL_ISSUES={len(issues)}")

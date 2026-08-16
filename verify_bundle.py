#!/usr/bin/env python3
"""Lightweight, dependency-free preflight check for the bundled legal dataset."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

EXPECTED_RECORDS = 165_303
EXPECTED_MAX_TOKENS = 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./qwen25_3b_semantic_colab.jsonl.gz")
    args = parser.parse_args()
    path = Path(args.data)
    if not path.exists():
        raise FileNotFoundError(path)

    count = 0
    max_tokens = 0
    role_errors = 0
    empty_assistant = 0
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        for block in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(block)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            count += 1
            messages = record.get("messages")
            if not isinstance(messages, list):
                role_errors += 1
                continue
            roles = {item.get("role") for item in messages if isinstance(item, dict)}
            if not roles.issubset({"system", "user", "assistant"}):
                role_errors += 1
            if not any(item.get("role") == "assistant" and item.get("content", "").strip() for item in messages):
                empty_assistant += 1
            value = record.get("token_count_qwen25_chat", 0)
            if not isinstance(value, int):
                role_errors += 1
            else:
                max_tokens = max(max_tokens, value)

    if count != EXPECTED_RECORDS:
        raise AssertionError(f"record count {count} != expected {EXPECTED_RECORDS}")
    if max_tokens > EXPECTED_MAX_TOKENS:
        raise AssertionError(f"max token_count {max_tokens} > {EXPECTED_MAX_TOKENS}")
    if role_errors or empty_assistant:
        raise AssertionError(f"role/schema errors={role_errors}, empty assistant records={empty_assistant}")

    print(json.dumps({
        "status": "PASS",
        "path": str(path.resolve()),
        "records": count,
        "max_token_count_qwen25_chat": max_tokens,
        "sha256_gzip": digest.hexdigest(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

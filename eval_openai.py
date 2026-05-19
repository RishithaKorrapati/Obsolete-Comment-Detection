"""
Zero-shot and few-shot comment–code consistency evaluation with the OpenAI API.

Same prompts and label semantics as test.py / eval_gemini.py:
  label 1 = inconsistent, label 0 = consistent
  prediction 1 = model said inconsistent

Few-shot demos are sampled from Data/{Summary|Param|Return}/train.json (never test.json).

Set CHATGPT_API_KEY in the environment or in a .env file next to this script.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import fire
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm

# Shared prompts and few-shot builders (no Gemini HTTP calls on import).
from eval_gemini import (
    ANSWER_SUFFIX,
    POST_INSTRUCTION,
    ROOT,
    _maybe_truncate,
    build_few_shot_prompt,
    build_few_shot_prompt_code_diff,
    format_code_diff_case,
    prediction_from_text,
)

load_dotenv(ROOT / ".env")


def _compute_metrics(pred: list[int], label: list[int]):
    acc = accuracy_score(label, pred)
    precision = precision_score(label, pred, zero_division=0)
    recall = recall_score(label, pred, zero_division=0)
    denom = precision + recall
    f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
    return acc, precision, recall, f1


def openai_list_models(api_key: str) -> list[str]:
    """List model ids available to this API key (subset of /v1/models)."""
    url = "https://api.openai.com/v1/models"
    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    ids = [m.get("id", "") for m in payload.get("data", []) if m.get("id")]
    return sorted(set(ids))


def openai_chat_complete(
    api_key: str,
    model_name: str,
    prompt: str,
    max_retries: int = 6,
    initial_backoff: float = 2.0,
) -> str:
    """POST /v1/chat/completions (stdlib only)."""
    url = "https://api.openai.com/v1/chat/completions"
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 128,
    }
    data = json.dumps(body).encode("utf-8")
    last_err: Exception | None = None
    backoff = initial_backoff

    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            choices = payload.get("choices") or []
            if not choices:
                return ""
            msg = (choices[0].get("message") or {})
            return (msg.get("content") or "").strip()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            last_err = RuntimeError(f"OpenAI HTTP {e.code}: {detail}")
            if e.code in (400, 401, 403):
                raise last_err from None
            if attempt < max_retries - 1 and e.code in (429, 500, 502, 503):
                print(
                    f"OpenAI HTTP {e.code}: sleeping {backoff:.1f}s before retry "
                    f"({attempt + 1}/{max_retries})…",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(backoff + random.uniform(0.0, 1.0))
                backoff = min(backoff * 1.8, 60.0)
                continue
            raise last_err from None

    raise last_err or RuntimeError("OpenAI request failed")


def main(
    model_name: str = "gpt-4o-mini",
    out_path: str = "Data/OpenAITestResult.xlsx",
    result_log: str = "result_openai.txt",
    sleep_seconds: float = 0.25,
    max_examples_per_split: int = 0,
    max_total_examples: int = 0,
    list_models: bool = False,
    categories: str = "Summary,Param,Return",
    few_shot_k: int = 0,
    few_shot_seed: int = 42,
    max_demo_chars: int = 4000,
    max_target_chars: int = 0,
    prompt_style: str = "new_code_comment",
    api_max_retries: int = 6,
    api_initial_backoff: float = 2.0,
):
    """
    :param model_name: OpenAI chat model id (e.g. gpt-4o-mini, gpt-4o).
    :param max_examples_per_split: If > 0, only first N examples per category.
    :param max_total_examples: If > 0, cap total examples across all categories.
    :param list_models: If True, print model ids for this key, then exit.
    :param max_target_chars: If > 0, truncate each text block in the prompt (saves tokens; 0 = full text).
    :param prompt_style: "new_code_comment" (new code + old comment) or "old_new_code_comment" (old + new code + old comment).
    """
    api_key = os.environ.get("CHATGPT_API_KEY")
    if not api_key:
        print(
            "Missing CHATGPT_API_KEY. Add it to .env or your environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    if prompt_style not in ("new_code_comment", "old_new_code_comment"):
        print(
            f"Invalid prompt_style={prompt_style!r}. Use new_code_comment or old_new_code_comment.",
            file=sys.stderr,
        )
        sys.exit(1)

    if list_models:
        try:
            ids = openai_list_models(api_key)
        except Exception as e:
            print(f"list_models failed: {e}", file=sys.stderr)
            sys.exit(1)
        print("Models visible to this key (use with --model_name=...):")
        for mid in ids[:200]:
            print(f"  {mid}")
        if len(ids) > 200:
            print(f"  ... and {len(ids) - 200} more")
        sys.exit(0)

    # Fire may pass --categories=A,B,C as a tuple instead of one string.
    if isinstance(categories, (list, tuple)):
        cats = [str(c).strip() for c in categories if str(c).strip()]
    else:
        cats = [c.strip() for c in str(categories).split(",") if c.strip()]
    results = {
        "category": [],
        "old_comment_raw": [],
        "new_code_raw": [],
        "new_comment_raw": [],
        "label": [],
        "raw_output": [],
        "flag": [],
    }
    flags: list[int] = []
    labels: list[int] = []
    skipped = 0
    examples_done = 0

    for class_ in cats:
        path = ROOT / f"Data/{class_}/test.json"
        if not path.exists():
            print(f"Skip missing file: {path}", file=sys.stderr)
            continue
        train_path = ROOT / f"Data/{class_}/train.json"
        train_rows: list | None = None
        if few_shot_k > 0:
            if not train_path.exists():
                print(f"Few-shot needs {train_path}; skipping category {class_}.", file=sys.stderr)
                continue
            with open(train_path, encoding="utf-8") as f:
                train_rows = json.load(f)
            if len(train_rows) < few_shot_k:
                print(
                    f"Warning: {class_} train has only {len(train_rows)} rows; using all as pool.",
                    file=sys.stderr,
                )

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if max_examples_per_split > 0:
            data = data[: max_examples_per_split]
        if max_total_examples > 0:
            remaining = max_total_examples - examples_done
            if remaining <= 0:
                break
            data = data[: min(len(data), remaining)]

        style_tag = "-diff" if prompt_style == "old_new_code_comment" else ""
        desc = f"OpenAI/{class_}{style_tag}" + (f"-{few_shot_k}shot" if few_shot_k else "-0shot")
        pbar = tqdm(data, desc=desc, mininterval=0.3, smoothing=0.05)
        for example in pbar:
            pbar.set_postfix_str("API…", refresh=True)
            if few_shot_k > 0 and train_rows is not None:
                if prompt_style == "old_new_code_comment":
                    prompt = build_few_shot_prompt_code_diff(
                        class_,
                        train_rows,
                        example["old_code_raw"],
                        example["new_code_raw"],
                        example["old_comment_raw"],
                        few_shot_k,
                        few_shot_seed,
                        max_demo_chars,
                        max_target_chars=max_target_chars,
                    )
                else:
                    prompt = build_few_shot_prompt(
                        class_,
                        train_rows,
                        example["new_code_raw"],
                        example["old_comment_raw"],
                        few_shot_k,
                        few_shot_seed,
                        max_demo_chars,
                        max_target_chars=max_target_chars,
                    )
            else:
                if prompt_style == "old_new_code_comment":
                    prompt = (
                        format_code_diff_case(
                            class_,
                            example["old_code_raw"],
                            example["new_code_raw"],
                            example["old_comment_raw"],
                            max_target_chars,
                        )
                        + ANSWER_SUFFIX
                    )
                else:
                    code_t = _maybe_truncate(example["new_code_raw"], max_target_chars)
                    comment_t = _maybe_truncate(example["old_comment_raw"], max_target_chars)
                    user_part = POST_INSTRUCTION.format(
                        class_.lower(),
                        code_t,
                        class_.lower(),
                        comment_t,
                    )
                    prompt = user_part + ANSWER_SUFFIX

            try:
                text = openai_chat_complete(
                    api_key,
                    model_name,
                    prompt,
                    max_retries=api_max_retries,
                    initial_backoff=api_initial_backoff,
                )
            except Exception as e:
                print(f"{class_} example error: {e}", file=sys.stderr)
                text = ""
            finally:
                pbar.set_postfix_str("", refresh=True)

            pred = prediction_from_text(text)
            if pred is None:
                skipped += 1
                pred = 0

            results["category"].append(class_)
            results["old_comment_raw"].append(example["old_comment_raw"])
            results["new_code_raw"].append(example["new_code_raw"])
            results["new_comment_raw"].append(example.get("new_comment_raw", ""))
            results["label"].append(example["label"])
            results["raw_output"].append(text)
            results["flag"].append(pred)
            flags.append(pred)
            labels.append(example["label"])

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            examples_done += 1
            if max_total_examples > 0 and examples_done >= max_total_examples:
                break

        if max_total_examples > 0 and examples_done >= max_total_examples:
            break

    acc, precision, recall, f1 = _compute_metrics(flags, labels)

    out_abs = ROOT / out_path
    out_abs.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_excel(out_abs, index=False)

    log_line = (
        f"openai_model: {model_name}, prompt_style: {prompt_style}, few_shot_k: {few_shot_k}, seed: {few_shot_seed}, "
        f"n_examples: {len(labels)}, max_per_split: {max_examples_per_split}, max_total: {max_total_examples}, "
        f"max_target_chars: {max_target_chars}, "
        f"acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, "
        f"skipped_unparsed: {skipped}\n"
    )
    with open(ROOT / result_log, "a", encoding="utf-8") as f:
        f.write(log_line)

    print(log_line.strip())
    if skipped:
        print(
            f"Note: {skipped} outputs could not be parsed; those were counted as CONSISTENT (0)."
        )


if __name__ == "__main__":
    fire.Fire(main)

"""
Zero-shot and few-shot comment–code consistency evaluation with the Gemini API.

Uses the same prompts and label semantics as test.py:
  label 1 = inconsistent, label 0 = consistent
  prediction 1 = model said inconsistent

Few-shot demos are sampled from Data/{Summary|Param|Return}/train.json (never test.json).

Set GOOGLE_GEMINI_API_KEY in the environment or in a .env file next to this script.
Use python eval_gemini.py --list_models=True to print model ids your key can call (avoids 404 on deprecated names).

For OpenAI (ChatGPT API) with the same task and prompts, use eval_openai.py and CHATGPT_API_KEY in .env.
Use --prompt_style=old_new_code_comment for old+new code + old comment (change-aware); default is new code + old comment only.
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
from urllib.parse import quote

import fire
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

POST_INSTRUCTION = """Is the given code consistent with the corresponding {}?
```code
{}
```
```{}
{}
```
"""

# Tight instruction so parsing is reliable; keeps parity with test.py’s binary task.
ANSWER_SUFFIX = """

Answer with exactly one word, either CONSISTENT or INCONSISTENT, and nothing else.
"""

FEW_SHOT_PREAMBLE = """Below are labeled examples from training data. Each ends with the correct answer (CONSISTENT or INCONSISTENT).

"""

FEW_SHOT_FINAL_INTRO = """Now classify this new case. Do not reuse the examples above; answer for this case only.
Reply with exactly one word: CONSISTENT or INCONSISTENT.

"""

# Second approach: show code before/after change + the (pre-change) comment.
POST_INSTRUCTION_CODE_DIFF = """Given the previous code and the updated code, is the corresponding {comment_kind} comment still consistent with the updated code?

```old_code
{old_code}
```
```new_code
{new_code}
```
```{comment_kind}
{comment}
```
"""


def _label_to_answer_word(label: int) -> str:
    return "INCONSISTENT" if int(label) == 1 else "CONSISTENT"


def _maybe_truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 24)] + "\n... [truncated] ..."


def format_single_case(class_: str, new_code_raw: str, old_comment_raw: str, max_chars: int) -> str:
    c = class_.lower()
    code = _maybe_truncate(new_code_raw, max_chars)
    comment = _maybe_truncate(old_comment_raw, max_chars)
    return POST_INSTRUCTION.format(c, code, c, comment)


def format_code_diff_case(
    class_: str,
    old_code_raw: str,
    new_code_raw: str,
    old_comment_raw: str,
    max_chars: int,
) -> str:
    """Old + new code and the comment (uses old_comment_raw from the dataset)."""
    kind = class_.lower()
    old_c = _maybe_truncate(old_code_raw, max_chars)
    new_c = _maybe_truncate(new_code_raw, max_chars)
    com = _maybe_truncate(old_comment_raw, max_chars)
    return POST_INSTRUCTION_CODE_DIFF.format(
        comment_kind=kind,
        old_code=old_c,
        new_code=new_c,
        comment=com,
    )


def select_few_shot_demonstrations(
    train_data: list,
    k: int,
    seed: int,
) -> list:
    """Pick up to k train rows, balanced between label 0 and 1 when possible."""
    if k <= 0:
        return []
    rng = random.Random(seed)
    zeros = [x for x in train_data if int(x["label"]) == 0]
    ones = [x for x in train_data if int(x["label"]) == 1]
    rng.shuffle(zeros)
    rng.shuffle(ones)
    n1 = k // 2
    n0 = k - n1
    out = zeros[:n0] + ones[:n1]
    if len(out) < k:
        pool = zeros[n0:] + ones[n1:]
        rng.shuffle(pool)
        for row in pool:
            if len(out) >= k:
                break
            out.append(row)
    rng.shuffle(out)
    return out[:k]


def build_few_shot_prompt(
    class_: str,
    train_rows: list,
    target_new_code: str,
    target_old_comment: str,
    k: int,
    seed: int,
    max_demo_chars: int,
    max_target_chars: int = 0,
) -> str:
    demos = select_few_shot_demonstrations(train_rows, k, seed)
    parts = [FEW_SHOT_PREAMBLE]
    for i, ex in enumerate(demos, start=1):
        block = format_single_case(
            class_,
            ex["new_code_raw"],
            ex["old_comment_raw"],
            max_demo_chars,
        )
        parts.append(f"Example {i}:\n{block}\nAnswer: {_label_to_answer_word(ex['label'])}\n\n")
    target_block = format_single_case(
        class_, target_new_code, target_old_comment, max_target_chars
    )
    parts.append(FEW_SHOT_FINAL_INTRO)
    parts.append(target_block)
    parts.append(ANSWER_SUFFIX)
    return "".join(parts)


def build_few_shot_prompt_code_diff(
    class_: str,
    train_rows: list,
    target_old_code: str,
    target_new_code: str,
    target_old_comment: str,
    k: int,
    seed: int,
    max_demo_chars: int,
    max_target_chars: int = 0,
) -> str:
    """Few-shot using old_code_raw, new_code_raw, old_comment_raw per row."""
    demos = select_few_shot_demonstrations(train_rows, k, seed)
    parts = [FEW_SHOT_PREAMBLE]
    for i, ex in enumerate(demos, start=1):
        block = format_code_diff_case(
            class_,
            ex["old_code_raw"],
            ex["new_code_raw"],
            ex["old_comment_raw"],
            max_demo_chars,
        )
        parts.append(f"Example {i}:\n{block}\nAnswer: {_label_to_answer_word(ex['label'])}\n\n")
    target_block = format_code_diff_case(
        class_,
        target_old_code,
        target_new_code,
        target_old_comment,
        max_target_chars,
    )
    parts.append(FEW_SHOT_FINAL_INTRO)
    parts.append(target_block)
    parts.append(ANSWER_SUFFIX)
    return "".join(parts)


def _gemini_error_json(detail: str) -> dict | None:
    try:
        return json.loads(detail)
    except json.JSONDecodeError:
        return None


def _gemini_429_daily_quota_exhausted(detail: str) -> bool:
    """True when 429 is from per-day free tier (waiting and retrying the same day won't help)."""
    obj = _gemini_error_json(detail)
    if not obj:
        return False
    err = obj.get("error") or {}
    for d in err.get("details") or []:
        if d.get("@type") != "type.googleapis.com/google.rpc.QuotaFailure":
            continue
        for v in d.get("violations") or []:
            qid = (v.get("quotaId") or "").lower()
            if "perday" in qid or "dayperproject" in qid:
                return True
    return False


def _gemini_429_retry_delay_seconds(detail: str) -> float | None:
    """Parse google.rpc.RetryInfo retryDelay if present (e.g. '23s' or '23.2s')."""
    obj = _gemini_error_json(detail)
    if not obj:
        return None
    err = obj.get("error") or {}
    for d in err.get("details") or []:
        if d.get("@type") != "type.googleapis.com/google.rpc.RetryInfo":
            continue
        rd = d.get("retryDelay")
        if rd is None:
            continue
        if isinstance(rd, (int, float)):
            return float(rd)
        if isinstance(rd, str):
            s = rd.strip().lower().rstrip("s")
            try:
                return float(s)
            except ValueError:
                return None
    return None


def gemini_generate_text(
    api_key: str,
    model_name: str,
    prompt: str,
    max_retries: int = 8,
    initial_backoff: float = 12.0,
) -> str:
    """Call Gemini REST API (stdlib only — no google-generativeai / heavy crypto stack)."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={quote(api_key, safe='')}"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 64},
    }
    data = json.dumps(body).encode("utf-8")
    last_err: Exception | None = None
    backoff = initial_backoff

    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            candidates = payload.get("candidates") or []
            if not candidates:
                return ""
            parts = (candidates[0].get("content") or {}).get("parts") or []
            return "".join((p.get("text") or "") for p in parts).strip()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            last_err = RuntimeError(f"Gemini HTTP {e.code}: {detail}")
            if attempt < max_retries - 1 and e.code in (429, 502, 503):
                if e.code == 429:
                    if _gemini_429_daily_quota_exhausted(detail):
                        print(
                            "Gemini HTTP 429: daily free-tier quota for this model is exhausted "
                            "(more retries today will not help; use another model, fewer examples, "
                            "or billing — see https://ai.google.dev/gemini-api/docs/rate-limits ).",
                            file=sys.stderr,
                            flush=True,
                        )
                        raise last_err from None
                    api_delay = _gemini_429_retry_delay_seconds(detail)
                    wait429 = api_delay if api_delay is not None else backoff
                    print(
                        f"Gemini HTTP 429: sleeping {wait429:.1f}s before retry "
                        f"({attempt + 1}/{max_retries})…",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(wait429)
                    if api_delay is None:
                        backoff = min(backoff * 1.5, 120.0)
                else:
                    # 502/503: capacity / overload — honor Retry-After if present, else backoff + jitter
                    wait: float
                    hdrs = getattr(e, "headers", None)
                    ra = hdrs.get("Retry-After") if hdrs else None
                    if ra is not None:
                        try:
                            wait = float(ra)
                        except ValueError:
                            wait = min(10.0 * (1.55**attempt), 120.0)
                    else:
                        wait = min(10.0 * (1.55**attempt), 120.0)
                    jitter = random.uniform(0.0, 2.5)
                    print(
                        f"Gemini HTTP {e.code}: sleeping {wait + jitter:.1f}s before retry "
                        f"({attempt + 1}/{max_retries})…",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(wait + jitter)
                continue
            raise last_err from None

    raise last_err or RuntimeError("Gemini request failed")


def gemini_list_generate_content_models(api_key: str) -> list[str]:
    """Return model ids (no 'models/' prefix) that list generateContent in supportedGenerationMethods."""
    out: list[str] = []
    page_token: str | None = None
    while True:
        q = f"pageSize=100&key={quote(api_key, safe='')}"
        if page_token:
            q += f"&pageToken={quote(page_token, safe='')}"
        url = f"https://generativelanguage.googleapis.com/v1beta/models?{q}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        for m in payload.get("models") or []:
            methods = m.get("supportedGenerationMethods") or []
            if "generateContent" not in methods:
                continue
            name = m.get("name") or ""
            if name.startswith("models/"):
                out.append(name[len("models/") :])
            elif name:
                out.append(name)
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return sorted(set(out))


def _compute_metrics(pred: list[int], label: list[int]):
    acc = accuracy_score(label, pred)
    precision = precision_score(label, pred, zero_division=0)
    recall = recall_score(label, pred, zero_division=0)
    denom = precision + recall
    f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
    return acc, precision, recall, f1


def prediction_from_text(text: str) -> int | None:
    """
    Return 1 if inconsistent, 0 if consistent, None if unclear.
    Mirrors test.py heuristic: any mention of 'inconsisten' -> inconsistent (1).
    """
    if not text:
        return None
    raw = text.strip().upper()
    # Prefer explicit one-word answers
    if re.search(r"\bINCONSISTENT\b", raw):
        return 1
    if re.search(r"\bCONSISTENT\b", raw):
        return 0
    lower = text.lower()
    if "inconsisten" in lower:
        return 1
    if "consistent" in lower and "inconsistent" not in lower:
        return 0
    return None


def main(
    model_name: str = "gemini-2.0-flash",
    out_path: str = "Data/GeminiTestResult.xlsx",
    result_log: str = "result_gemini.txt",
    sleep_seconds: float = 0.4,
    max_examples_per_split: int = 0,
    max_total_examples: int = 0,
    list_models: bool = False,
    categories: str = "Summary,Param,Return",
    few_shot_k: int = 0,
    few_shot_seed: int = 42,
    max_demo_chars: int = 4000,
    max_target_chars: int = 0,
    prompt_style: str = "new_code_comment",
    gemini_max_retries: int = 8,
    gemini_initial_backoff: float = 12.0,
):
    """
    :param model_name: Gemini model id for the URL path (e.g. gemini-2.0-flash). Bare
        names like gemini-1.5-flash often 404; use list_models=True or a versioned id from the API list.
    :param max_examples_per_split: If > 0, only first N examples per category (debug / quota).
    :param max_total_examples: If > 0, stop after this many examples total across all categories
        (smoke test; combines with max_examples_per_split if both set).
    :param list_models: If True, print models that support generateContent for this API key, then exit.
    :param categories: Comma-separated: Summary, Param, Return
    :param few_shot_k: Number of train-set demonstrations prepended to each prompt (0 = zero-shot).
    :param few_shot_seed: RNG seed for picking balanced demos from train.json (reproducible).
    :param max_demo_chars: Max chars per code / comment block in demos only (0 = no limit).
    :param max_target_chars: If > 0, truncate each text block in the test case to this many chars (saves tokens).
    :param prompt_style: "new_code_comment" (default: new_code + old_comment) or
        "old_new_code_comment" (old_code + new_code + old_comment for change-aware check).
    :param gemini_max_retries: HTTP attempts per example for 429/502/503 (each waits with backoff between tries).
    :param gemini_initial_backoff: First sleep (seconds) after HTTP 429 before retry; grows ×1.5 each 429.
    """
    api_key = os.environ.get("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        print(
            "Missing GOOGLE_GEMINI_API_KEY. Add it to .env or your environment.",
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
            ids = gemini_list_generate_content_models(api_key)
        except Exception as e:
            print(f"list_models failed: {e}", file=sys.stderr)
            sys.exit(1)
        print("Models supporting generateContent (use the id with --model_name=...):")
        for mid in ids:
            print(f"  {mid}")
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
        desc = f"Gemini/{class_}{style_tag}" + (f"-{few_shot_k}shot" if few_shot_k else "-0shot")
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
                text = gemini_generate_text(
                    api_key,
                    model_name,
                    prompt,
                    max_retries=gemini_max_retries,
                    initial_backoff=gemini_initial_backoff,
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
        f"gemini_model: {model_name}, prompt_style: {prompt_style}, few_shot_k: {few_shot_k}, seed: {few_shot_seed}, "
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

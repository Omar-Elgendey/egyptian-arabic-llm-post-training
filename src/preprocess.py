"""Egyptian-Arabic preprocessing: clean -> Egyptian filter (SFT) -> exact dedup -> 90/10 split -> validate -> save.

Reads raw SFT/DPO with streaming, writes clean JSONL for training by someone else.
No model training, no tokenizer, no max_length decision, no LLaMA-Factory config.

Usage:
    uv run python -m src.preprocess --limit-sft 2000 --limit-dpo 500   # smoke test
    uv run python -m src.preprocess                                    # full run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import unicodedata
from collections import Counter

from datasets import load_dataset

SFT_DATASET_ID = "MBZUAI-Paris/Egyptian-SFT-Mixture"
DPO_DATASET_ID = "HeshamHaroon/arabic-dialect-dpo"
DPO_CONFIG = "egyptian"
SPLIT = "train"

SEED = 42
TRAIN_RATIO = 0.90
OUTPUT_DIR = "data/processed"

VALID_ROLES = ("system", "user", "assistant")

# Small transparent word lists (heuristics only, not a classifier).
EGYPTIAN_WORDS = [
    "ايه", "ازاي", "ازاى", "عايز", "عاوز", "كده", "كدا",
    "دلوقتي", "دلوقتى", "علشان", "عشان", "برضه", "كمان",
    "يعني", "يعنى", "مصر", "القاهرة", "جنيه", "خلاص", "فين", "ليه",
]
# Strong non-Egyptian markers. Only used to remove when NO Egyptian signal exists.
NON_EGYPTIAN_WORDS = ["وشلون", "شلون", "شخبار", "وايد", "هايدا", "هيك", "برشا"]
# A few common Franco words. Franco = Latin letters + digits like 3/7/2.
FRANCO_WORDS = {
    "ana", "enta", "enti", "eih", "ezay", "3ayez", "3awz",
    "keda", "awi", "mosh", "mesh", "el", "fel", "3ashan", "masr",
}
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
FRANCO_DIGIT_RE = re.compile(r"[a-zA-Z][23758]|[23758][a-zA-Z]")


def norm_text(t: str) -> str:
    """Normalize for dedup keys only. Original text is preserved in outputs."""
    t = unicodedata.normalize("NFKC", t or "")
    return " ".join(t.strip().lower().split())


def clean_sft_messages(raw) -> tuple[list[dict] | None, str | None]:
    """Keep valid system/user/assistant turns. id/length_tokens are ignored."""
    if not isinstance(raw, list) or not raw:
        return None, "invalid_messages"
    cleaned: list[dict] = []
    for m in raw:
        if not isinstance(m, dict):
            return None, "invalid_messages"
        if m.get("role") not in VALID_ROLES:
            return None, "invalid_messages"
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            return None, "empty_message"
        cleaned.append({"role": m["role"], "content": content.strip()})
    roles = [m["role"] for m in cleaned]
    if "user" not in roles or "assistant" not in roles:
        return None, "invalid_messages"
    return cleaned, None


def clean_dpo(prompt, chosen, rejected) -> tuple[dict | None, str | None]:
    if not isinstance(prompt, str) or not prompt.strip():
        return None, "empty_prompt"
    if not isinstance(chosen, str) or not chosen.strip():
        return None, "empty_chosen"
    if not isinstance(rejected, str) or not rejected.strip():
        return None, "empty_rejected"
    if chosen.strip() == rejected.strip():
        return None, "chosen_equals_rejected"
    return {
        "prompt": prompt.strip(),
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
    }, None


def is_egyptian_relevant(messages: list[dict]) -> tuple[bool, str]:
    """Simple content check. Uncertain cases return True (kept, not deleted)."""
    text = " ".join(m["content"] for m in messages)
    n_arabic = len(ARABIC_RE.findall(text))
    n_latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    low = text.lower()
    has_franco = bool(FRANCO_DIGIT_RE.search(low)) or len(
        set(re.findall(r"[a-zA-Z0-9]+", low)) & FRANCO_WORDS
    ) >= 2
    has_egyptian = any(w in text for w in EGYPTIAN_WORDS)
    has_non_egyptian = any(w in text for w in NON_EGYPTIAN_WORDS)

    if has_egyptian or has_franco:
        return True, "egyptian_or_franco"
    if n_arabic == 0 and not has_franco and n_latin >= 20:
        return False, "pure_english"
    if "```" in text and n_arabic == 0 and not has_franco:
        return False, "pure_code_no_arabic"
    if has_non_egyptian and not has_egyptian and n_arabic < 10 and not has_franco:
        return False, "non_egyptian"
    return True, "uncertain_kept"


def sft_key(messages: list[dict]) -> str:
    joined = "\n".join(f"{m['role']}:{norm_text(m['content'])}" for m in messages)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def dpo_key(prompt: str, chosen: str, rejected: str) -> str:
    joined = "\n".join([norm_text(prompt), norm_text(chosen), norm_text(rejected)])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def split_train_eval(records: list, train_ratio: float = TRAIN_RATIO, seed: int = SEED):
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n_train = int(len(records) * train_ratio)
    return [records[i] for i in idx[:n_train]], [records[i] for i in idx[n_train:]]


def write_jsonl(path: str, rows: list[dict], flush_every: int = 5000) -> None:
    """Write rows progressively, one line at a time.

    Writes to path + ".tmp" and renames only on success, so an interrupted
    run never leaves a truncated file looking like the final output.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for i, r in enumerate(rows, 1):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                if i % flush_every == 0:
                    f.flush()
        os.replace(tmp_path, path)
    except BaseException:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def write_text_atomic(path: str, text: str) -> None:
    """Same temp+rename guarantee for small files like stats.json."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean SFT/DPO -> JSONL + stats.json")
    parser.add_argument("--limit-sft", type=int, default=None)
    parser.add_argument("--limit-dpo", type=int, default=None)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    sft_ds = load_dataset(SFT_DATASET_ID, split=SPLIT, streaming=True)
    dpo_ds = load_dataset(DPO_DATASET_ID, DPO_CONFIG, split=SPLIT, streaming=True)
    if args.limit_sft is not None:
        sft_ds = sft_ds.take(args.limit_sft)
    if args.limit_dpo is not None:
        dpo_ds = dpo_ds.take(args.limit_dpo)

    # ---- SFT: clean + Egyptian filter ----
    sft_raw = 0
    sft_removed: Counter = Counter()
    sft_records: list[dict] = []
    for ex in sft_ds:
        sft_raw += 1
        messages, reason = clean_sft_messages(ex.get("messages"))
        if reason:
            sft_removed[reason] += 1
            continue
        keep, _ = is_egyptian_relevant(messages)
        if not keep:
            sft_removed["not_egyptian_relevant"] += 1
            continue
        sft_records.append({"messages": messages})

    # ---- SFT: exact dedup (normalized full conversation incl. roles) ----
    seen: set[str] = set()
    sft_deduped: list[dict] = []
    for r in sft_records:
        h = sft_key(r["messages"])
        if h in seen:
            continue
        seen.add(h)
        sft_deduped.append(r)
    sft_dedup_removed = len(sft_records) - len(sft_deduped)
    sft_train, sft_eval = split_train_eval(sft_deduped)

    # ---- DPO: clean only (generic/short rejected are kept) ----
    dpo_raw = 0
    dpo_removed: Counter = Counter()
    dpo_records: list[dict] = []
    for ex in dpo_ds:
        dpo_raw += 1
        cleaned, reason = clean_dpo(ex.get("prompt"), ex.get("chosen"), ex.get("rejected"))
        if reason:
            dpo_removed[reason] += 1
            continue
        dpo_records.append(cleaned)

    seen = set()
    dpo_deduped: list[dict] = []
    for r in dpo_records:
        h = dpo_key(r["prompt"], r["chosen"], r["rejected"])
        if h in seen:
            continue
        seen.add(h)
        dpo_deduped.append(r)
    dpo_dedup_removed = len(dpo_records) - len(dpo_deduped)
    dpo_train, dpo_eval = split_train_eval(dpo_deduped)

    # ---- Validate + save (temp files renamed only on success) ----
    for r in sft_train + sft_eval:
        assert isinstance(r.get("messages"), list) and r["messages"]
    for r in dpo_train + dpo_eval:
        assert all(isinstance(r[k], str) and r[k].strip() for k in ("prompt", "chosen", "rejected"))
        assert r["chosen"].strip() != r["rejected"].strip()

    outputs = [
        (os.path.join(args.output_dir, "sft_train.jsonl"), sft_train),
        (os.path.join(args.output_dir, "sft_eval.jsonl"), sft_eval),
        (os.path.join(args.output_dir, "dpo_train.jsonl"), dpo_train),
        (os.path.join(args.output_dir, "dpo_eval.jsonl"), dpo_eval),
    ]
    try:
        for path, rows in outputs:
            print(f"writing {path} ({len(rows)} rows)...", flush=True)
            write_jsonl(path, rows)
            print(f"done {path}", flush=True)
    except (KeyboardInterrupt, Exception):
        print("interrupted during JSONL writing; partial .tmp files removed, final outputs untouched.", flush=True)
        raise

    stats = {
        "sft": {
            "raw": sft_raw,
            "removed_by_reason": dict(sft_removed),
            "dedup_removed": sft_dedup_removed,
            "final": len(sft_deduped),
            "train": len(sft_train),
            "eval": len(sft_eval),
        },
        "dpo": {
            "raw": dpo_raw,
            "removed_by_reason": dict(dpo_removed),
            "dedup_removed": dpo_dedup_removed,
            "final": len(dpo_deduped),
            "train": len(dpo_train),
            "eval": len(dpo_eval),
        },
    }
    os.makedirs(args.output_dir, exist_ok=True)
    write_text_atomic(
        os.path.join(args.output_dir, "stats.json"),
        json.dumps(stats, ensure_ascii=False, indent=2),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

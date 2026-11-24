from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from build_threads import speaker

LOSS_BIT = np.uint32(1 << 31)
VAL_PER_10K = 20


def is_val(match_id: int) -> bool:
    return (int(match_id) * 2654435761) % 2**32 % 10_000 < VAL_PER_10K


def label(who: int) -> str:
    return speaker(who) + ":"


def body(text: str) -> str:
    return f" {text}\n"


_tok = None


def load_tokenizer(name: str):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = Path(name) / "tokenizer.json"
    return Tokenizer.from_file(str(path if path.exists() else hf_hub_download(name, "tokenizer.json")))


def _init(tokenizer_name: str) -> None:
    global _tok
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _tok = load_tokenizer(tokenizer_name)


def _encode(tok, texts: list[str]) -> list[list[int]]:
    if hasattr(tok, "encode_batch"):
        return [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]
    return tok(texts, add_special_tokens=False)["input_ids"]


def encode_chunks(tok, chunks: list[dict]) -> list[np.ndarray]:
    labels = {}
    bodies = [body(t) for ch in chunks for t in ch["text"]]
    body_ids = _encode(tok, bodies)
    docs, k = [], 0
    for ch in chunks:
        loss = set(ch["loss"])
        parts = []
        for i, who in enumerate(ch["who"]):
            lab = labels.get(who)
            if lab is None:
                lab = labels[who] = np.asarray(_encode(tok, [label(who)])[0], np.uint32)
            b = np.asarray(body_ids[k], np.uint32)
            k += 1
            if i in loss:
                b = b | LOSS_BIT
            parts += [lab, b]
        docs.append(np.concatenate(parts))
    return docs


def tokenize_shard(job):
    src, out_dir, force = Path(job[0]), Path(job[1]), job[2]
    import zstandard
    from build_dataset import iter_records

    idx = src.name.split("-")[1].split(".")[0]
    stats_path = out_dir / "stats" / f"{idx}.json"
    if not force and stats_path.exists():
        return idx, json.loads(stats_path.read_text()), True

    started = time.time()
    stats = {"docs_train": 0, "docs_val": 0, "tok_train": 0, "tok_val": 0, "loss_train": 0, "loss_val": 0}
    buf = {"train": [], "val": []}
    for rec in iter_records(src):
        buf["val" if is_val(rec["m"]) else "train"].append(rec)
    for split, chunks in buf.items():
        tokens, ends, pos = [], [], 0
        for i in range(0, len(chunks), 4096):
            for d in encode_chunks(_tok, chunks[i : i + 4096]):
                tokens.append(d)
                pos += len(d)
                ends.append(pos)
        arr = np.concatenate(tokens) if tokens else np.zeros(0, np.uint32)
        for ext, data in (("bin", arr), ("idx", np.asarray(ends, np.int64))):
            p = out_dir / f"{split}-{idx}.{ext}"
            data.tofile(p.with_suffix(".part"))
            os.replace(p.with_suffix(".part"), p)
        stats[f"docs_{split}"] = len(ends)
        stats[f"tok_{split}"] = int(pos)
        stats[f"loss_{split}"] = int((arr >> 31).sum())
    stats["seconds"] = round(time.time() - started, 1)
    stats_path.write_text(json.dumps(stats))
    return idx, stats, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("threads"))
    ap.add_argument("--out", type=Path, default=Path("tokens"))
    ap.add_argument("--tokenizer", default="unsloth/Qwen3.5-4B-Base")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shards", type=int, nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    (args.out / "stats").mkdir(parents=True, exist_ok=True)
    srcs = sorted(args.src.glob("threads-*.jsonl.zst"))
    if args.shards:
        wanted = {f"{s:05d}" for s in args.shards}
        srcs = [p for p in srcs if p.name.split("-")[1].split(".")[0] in wanted]
    if not srcs:
        raise SystemExit(f"no thread shards in {args.src}")

    totals: dict[str, float] = {}
    started = time.time()
    jobs = [(str(p), str(args.out), args.force) for p in srcs]
    with Pool(min(args.workers, len(jobs)), initializer=_init, initargs=(args.tokenizer,)) as pool:
        for done, (idx, stats, cached) in enumerate(pool.imap_unordered(tokenize_shard, jobs), 1):
            for k, v in stats.items():
                totals[k] = totals.get(k, 0) + v
            print(
                f"[{done}/{len(jobs)}] shard {idx} {'(cached)' if cached else f'{stats['seconds']}s'}"
                f"  tokens={stats['tok_train'] + stats['tok_val']:,}"
                f"  | total={totals['tok_train'] + totals['tok_val']:,}  {time.time() - started:.0f}s",
                flush=True,
            )
    totals.pop("seconds", None)
    totals["tokenizer"] = args.tokenizer
    (args.out / "summary.json").write_text(json.dumps(totals, indent=2))
    for split in ("train", "val"):
        t, l, d = totals[f"tok_{split}"], totals[f"loss_{split}"], totals[f"docs_{split}"]
        print(f"{split:5}  {d:,} docs  {t:,} tokens  loss on {l:,} ({l / max(t, 1):.0%})  avg {t / max(d, 1):.0f} tok/doc")


if __name__ == "__main__":
    main()

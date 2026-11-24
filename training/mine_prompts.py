from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import zstandard

from pretokenize import is_val
from toxic_words import toxic


def norm(s: str) -> str:
    return " ".join(s.split())


def matches(path: Path):
    with open(path, "rb") as f:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True), encoding="utf-8"):
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=Path, default=Path("threads"))
    ap.add_argument("--shards", default="0-59")
    ap.add_argument("--n-prompts", type=int, default=20000)
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=Path("prompts.jsonl"))
    ap.add_argument("--prefix", default="p")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    a, b = map(int, args.shards.split("-"))
    used = set()
    for p in args.exclude:
        for l in Path(p).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                used.add(norm(r.get("insult") or r.get("text") or "").lower())
    rng = random.Random(args.seed)
    prompts, seen = [], set()
    n_matches = n_val = 0
    for k in range(a, b + 1):
        p = args.threads / f"threads-{k:05d}.jsonl.zst"
        if not p.exists():
            continue
        for m in matches(p):
            n_matches += 1
            if is_val(m["m"]):
                n_val += 1
                continue
            who, text = m["who"], [norm(t) for t in m["text"]]
            for i in range(len(text) - 1):
                t = text[i]
                if who[i + 1] != who[i] and 2 <= len(t.split()) <= 25 and toxic(t):
                    key = t.lower()
                    if key not in used and key not in seen and rng.random() < args.rate:
                        seen.add(key)
                        prompts.append(t)
        print(f"shard {k}: {len(prompts):,} prompts from {n_matches:,} matches ({n_val:,} validation matches skipped)", flush=True)
    rng.shuffle(prompts)
    args.out.write_text("".join(json.dumps({"id": f"{args.prefix}-{i:05d}", "insult": t}, ensure_ascii=False) + "\n"
                                for i, t in enumerate(prompts[:args.n_prompts])), encoding="utf-8")
    print(f"wrote {min(len(prompts), args.n_prompts):,} prompts -> {args.out}")


if __name__ == "__main__":
    main()

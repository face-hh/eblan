from __future__ import annotations

import collections
import json
import math
import re
import statistics
from pathlib import Path

W = re.compile(r"[a-z']+")


def terms(text: str) -> tuple[set[str], set[tuple[str, str]]]:
    t = W.findall(text.lower())
    return set(t), set(zip(t, t[1:]))


def real_rates(files: list[str]) -> tuple[dict, dict, int]:
    uni, bi, n = collections.Counter(), collections.Counter(), 0
    for f in files:
        for l in Path(f).read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            for line in (r["lines"] if "lines" in r else [r["insult"]]):
                u, b = terms(line)
                uni.update(u)
                bi.update(b)
                n += 1
    return uni, bi, n


def report(win: list[dict], uni, bi, n, k: int = 25) -> None:
    c = collections.Counter()
    bc = collections.Counter()
    for r in win:
        u, b = terms(r["best"])
        c.update(u)
        bc.update(b)
    N = len(win)
    print(f"{N} winners, {len({r['best'].lower() for r in win})} distinct, mean words "
          f"{statistics.mean(len(r['best'].split()) for r in win):.1f}, mean tox {statistics.mean(r['tox'] for r in win):.2f}")
    print("word            win%  real%")
    for w, v in c.most_common(k):
        print(f"  {w:<13} {100 * v / N:5.1f} {100 * uni[w] / n:5.1f}")
    print("top pairs:", [(" ".join(g), round(100 * v / N, 1)) for g, v in bc.most_common(10)])
    ent = -sum(v / sum(c.values()) * math.log(v / sum(c.values())) for v in c.values())
    print(f"unigram entropy {ent:.2f}, vocab {len(c)}")


def build_tokens(win: list[dict], out_dir: Path, val_from: Path, tokenizer: str) -> None:
    import numpy as np

    from pretokenize import load_tokenizer

    tok = load_tokenizer(tokenizer)
    piece = lambda s: tok.encode(s, add_special_tokens=False).ids
    loss = np.uint32(1 << 31)
    a_lab, b_lab = piece("A:"), piece("B:")
    docs = []
    for r in win:
        ctx = np.array(a_lab + piece(f" {r['insult']}\n") + b_lab, np.uint32)
        docs.append(np.concatenate([ctx, np.array(piece(f" {r['best']}\n"), np.uint32) | loss]))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.concatenate(docs).tofile(out_dir / "train-00000.bin")
    np.cumsum([len(d) for d in docs]).astype(np.int64).tofile(out_dir / "train-00000.idx")
    for f in val_from.glob("val-*"):
        (out_dir / f.name).write_bytes(f.read_bytes())
    print(f"{len(docs)} docs, {sum(len(d) for d in docs)} tokens -> {out_dir}")

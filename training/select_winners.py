from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics
from pathlib import Path

import selection as S
from slurs import SLUR

CAPS = {
    "n_word": (re.compile(SLUR["n_word"]), 0.15),
    "fucking": (re.compile(r"\bf(uck|uk|ck)(ing|in|n)\b|\bfkn\b|\bfking\b"), 0.20),
    "mom": (re.compile(r"\b(mom|moms|mother|mothers|mum|mommy|momma|mama)\b"), 0.30),
    "f_slur": (re.compile(SLUR["f_slur"]), 0.06),
    "dick": (re.compile(r"\b(dick|dicks|cock|cocks)\b"), 0.10),
}


WORD = re.compile(r"[a-z0-9']+")


def grams(text: str, n: int) -> set:
    w = WORD.findall(text.lower())
    return set(zip(*[w[i:] for i in range(n)]))


def select(rows: list[dict], warm: int = 500, seed: int = 1, min_words: int = 2, budget: tuple | None = None) -> list[dict]:
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    cnt, out, seen = collections.Counter(), [], set()
    gc = [collections.Counter() for _ in range(3)]
    caps = [max(3, int(b * len(rows))) for b in budget] if budget else None
    for r in rows:
        cs = [c for c in r["cands"] if not c[3] and len(c[0].split()) >= min_words and c[0].lower() not in seen]
        if not cs:
            continue
        pm = sorted(c[2] for c in cs)
        med = pm[len(pm) // 2] if len(cs) > 1 else pm[0]
        d = max(len(out), warm)
        best, best_s = None, -1e9
        for reply, tox, pmi, _ in cs:
            low = reply.lower()
            fams = [k for k, (rx, _) in CAPS.items() if rx.search(low)]
            if any(cnt[k] / d >= CAPS[k][1] for k in fams):
                continue
            gs = [grams(reply, n) for n in (2, 3, 4)] if budget else None
            if budget and any(gc[i][g] >= caps[i] for i in range(3) for g in gs[i]):
                continue
            s = tox - 10.0 * (pmi < med)
            if s > best_s:
                best, best_s = (reply, tox, pmi, fams), s
        if best is None:
            continue
        seen.add(best[0].lower())
        cnt.update(best[3])
        if budget:
            for i, n in enumerate((2, 3, 4)):
                gc[i].update(grams(best[0], n))
        out.append({"id": r["id"], "insult": r["insult"], "best": best[0], "tox": best[1], "pmi": best[2], "score": best_s})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("--out-jsonl", type=Path)
    ap.add_argument("--tokens-out", type=Path)
    ap.add_argument("--ref", nargs="+", default=["prompts.jsonl"])
    ap.add_argument("--val-from", type=Path, default=Path("tokens_comeback"))
    ap.add_argument("--tokenizer", default="unsloth/Qwen3.5-4B-Base")
    ap.add_argument("--budget", type=float, nargs=3, metavar=("B2", "B3", "B4"), default=None)
    a = ap.parse_args()
    uni, bi, n = S.real_rates(a.ref)
    rows = [json.loads(l) for l in a.raw.read_text(encoding="utf-8").splitlines() if l.strip()]
    allt = statistics.mean(c[1] for r in rows for c in r["cands"])
    win = select(rows, budget=tuple(a.budget) if a.budget else None)
    print(f"mean tox of all candidates {allt:.2f}")
    S.report(win, uni, bi, n, k=12)
    N = len(win)
    print({k: round(sum(bool(rx.search(w["best"].lower())) for w in win) / N, 3) for k, (rx, _) in CAPS.items()})
    toks = [x for w in win for x in WORD.findall(w["best"].lower())]
    bg = collections.Counter(zip(toks, toks[1:]))
    g3 = collections.Counter(g for w in win for g in grams(w["best"], 3))
    print("distinct-2 ratio", round(len(bg) / max(1, len(toks) - 1), 3), "| top 3-grams %", [(" ".join(g), round(100 * v / N, 1)) for g, v in g3.most_common(4)])
    if a.out_jsonl:
        a.out_jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in win), encoding="utf-8")
    if a.tokens_out:
        S.build_tokens(win, a.tokens_out, a.val_from, a.tokenizer)


if __name__ == "__main__":
    main()

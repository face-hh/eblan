from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()


def grams(t: str, n: int = 5) -> set[str]:
    t = norm(t).replace(" ", "")
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def text_of(r: dict) -> str:
    return r.get("insult") or r.get("text") or ""


class LeakGuard:
    def __init__(self, texts: list[str], threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.exact = {norm(t) for t in texts}
        self.grams = [grams(t) for t in texts]
        self.inv: dict[str, set[int]] = {}
        for i, g in enumerate(self.grams):
            for x in g:
                self.inv.setdefault(x, set()).add(i)

    def leaks(self, t: str) -> bool:
        if norm(t) in self.exact:
            return True
        g = grams(t)
        cand: set[int] = set()
        for x in g:
            cand |= self.inv.get(x, set())
        return any(len(g & self.grams[i]) / max(1, len(g | self.grams[i])) >= self.threshold for i in cand)


def read(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, metavar="NAME=FILE:N")
    ap.add_argument("--ban", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=50)
    ap.add_argument("--seed", type=int, default=8)
    a = ap.parse_args()

    guard = LeakGuard([text_of(r) for f in a.ban for r in read(f)])
    rng = random.Random(a.seed)
    rows, seen = [], set()
    for spec in a.source:
        name, rest = spec.split("=", 1)
        path, n = rest.rsplit(":", 1)
        pool = [" ".join(text_of(r).split()) for r in read(path)]
        rng.shuffle(pool)
        kept = []
        for t in pool:
            if not a.min_words <= len(t.split()) <= a.max_words or norm(t) in seen or guard.leaks(t):
                continue
            seen.add(norm(t))
            kept.append(t)
            if len(kept) >= int(n):
                break
        rows += [{"id": f"{name}{i}", "insult": t, "src": name} for i, t in enumerate(kept)]
        print(f"{name}: {len(kept)} of {len(pool)} candidates kept")
    assert not any(guard.leaks(r["insult"]) for r in rows)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} prompts -> {a.out}")


if __name__ == "__main__":
    main()

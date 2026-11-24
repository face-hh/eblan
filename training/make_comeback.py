from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

import pretokenize as pt
from toxic_words import toxic

LOSS = np.uint32(1 << 31)
YOU = re.compile(r"\b(you|your|youre|you're|ur|u|ya|yourself|urself|ure|u're)\b", re.I)
SELF_OWN = re.compile(r"^\W*(i'?m|i am|im)\b", re.I)
WORD = re.compile(r"[a-z']+")


def words(t: str) -> set[str]:
    return set(WORD.findall(t.lower()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("tokens"))
    ap.add_argument("--scores", type=Path, default=Path("scores_rest"))
    ap.add_argument("--out", type=Path, default=Path("tokens_comeback"))
    ap.add_argument("--val-from", type=Path, default=Path("tokens_s10f"))
    ap.add_argument("--tokenizer", default="unsloth/Qwen3.5-0.8B-Base")
    ap.add_argument("--min-rel", type=float, default=1.5)
    ap.add_argument("--max-echo", type=float, default=0.6)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    if args.summary:
        tot = {}
        for p in args.out.glob("stats-*.json"):
            for k, v in json.loads(p.read_text()).items():
                tot[k] = tot.get(k, 0) + v
        tok_n = sum(p.stat().st_size // 4 for p in args.out.glob("train-*.bin"))
        print(json.dumps(tot), f"| {tok_n / 1e6:.1f}M tokens")
        return

    tok = pt.load_tokenizer(args.tokenizer)
    vocab_nl = np.zeros(tok.get_vocab_size() + 1, bool)
    for t, i in tok.get_vocab().items():
        if "\n" in tok.decode([i]):
            vocab_nl[i] = True
    lab_a, lab_b = tok.encode("A:").ids, tok.encode("B:").ids

    stems = sorted(p.stem for p in args.data.glob("train-*.bin"))
    k, n = map(int, args.part.split("/"))
    args.out.mkdir(exist_ok=True)
    st = dict(runs=0, toxic_rel=0, prev_toxic=0, kept=0, docs=0, pair_tok=0, full_tok=0)
    for stem in stems[k::n]:
        if (args.out / f"{stem}.idx").exists():
            continue
        sc = np.load(args.scores / f"{stem}.npz")
        rel = 0.5 * (sc["none"] + sc["mis"]) - sc["real"]
        off = np.concatenate([[0], np.cumsum(sc["n_runs"])])
        ends = np.fromfile(args.data / f"{stem}.idx", np.int64)
        starts = np.concatenate([[0], ends[:-1]])
        data = np.memmap(args.data / f"{stem}.bin", np.uint32, "r")

        cand = []
        reps, prevs = [], []
        raws = []
        for j, d in enumerate(sc["doc"]):
            raw = np.array(data[starts[d]: ends[d]])
            raws.append(raw)
            m = (raw & LOSS) != 0
            dm = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
            runs = list(zip(np.flatnonzero(dm == 1).tolist(), np.flatnonzero(dm == -1).tolist()))
            st["runs"] += len(runs)
            ids = raw & ~LOSS
            nl = np.flatnonzero(vocab_nl[ids])
            for r, (a, b) in enumerate(runs):
                if rel[off[j] + r] < args.min_rel:
                    continue
                before = nl[nl < a]
                if len(before) == 0:
                    continue
                k1 = before[-1]
                k2 = before[-2] if len(before) > 1 else -1
                cand.append((j, r, a, b, k2 + 1, k1 + 1))
                reps.append(ids[a:b].tolist())
                prevs.append(ids[k2 + 1: k1 + 1].tolist())
        rep_t = tok.decode_batch(reps)
        prev_t = tok.decode_batch(prevs)

        keep_runs: dict[int, list[tuple[int, int]]] = {}
        pairs = []
        for (j, r, a, b, plo, phi), rt, pt_ in zip(cand, rep_t, prev_t):
            rt = rt.strip()
            pspk, _, ptxt = pt_.strip().partition(":")
            ptxt = ptxt.strip()
            if not toxic(rt):
                continue
            st["toxic_rel"] += 1
            if not ptxt or not toxic(ptxt):
                continue
            st["prev_toxic"] += 1
            cur_lab = tok.decode((raws[j][phi:a] & ~LOSS).tolist()).strip().rstrip(":")
            if cur_lab == pspk.strip() or not YOU.search(rt):
                continue
            pw, rw = words(ptxt), words(rt)
            if rw and len(rw & pw) / len(rw) >= args.max_echo:
                continue
            if SELF_OWN.search(rt) and not YOU.search(rt.split(" ", 1)[-1]):
                continue
            st["kept"] += 1
            keep_runs.setdefault(j, []).append((a, b))
            body = tok.encode(f" {ptxt}\n").ids
            rep_ids = (raws[j][a:b] & ~LOSS).astype(np.uint32)
            pairs.append(np.concatenate([np.array(lab_a + body + lab_b, np.uint32), rep_ids | LOSS]))

        docs = []
        for j, kept in keep_runs.items():
            raw = raws[j] & ~LOSS
            for a, b in kept:
                raw[a:b] |= LOSS
            docs.append(raw[: max(b for _, b in kept)])
        st["docs"] += len(docs)
        allx = docs + pairs
        st["full_tok"] += sum(len(x) for x in docs)
        st["pair_tok"] += sum(len(x) for x in pairs)
        rng = np.random.default_rng(int(stem[6:]))
        order = rng.permutation(len(allx))
        allx = [allx[i] for i in order]
        arr = np.concatenate(allx) if allx else np.zeros(0, np.uint32)
        arr.tofile(args.out / f"{stem}.bin")
        np.cumsum([len(x) for x in allx]).astype(np.int64).tofile(args.out / f"{stem}.idx")
        print(f"{stem}: kept {len(pairs):,} comebacks, {len(arr) / 1e6:.2f}M tokens", flush=True)

    (args.out / f"stats-{k}.json").write_text(json.dumps(st))
    if k == 0:
        for f in args.val_from.glob("val-*"):
            shutil.copy2(f, args.out / f.name)


if __name__ == "__main__":
    main()

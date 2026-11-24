from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

import pretokenize as pt
from toxic_words import toxic

LOSS = np.uint32(1 << 31)


def loss_runs(doc: np.ndarray) -> list[tuple[int, int]]:
    m = (doc & LOSS) != 0
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("tokens"))
    ap.add_argument("--scores", type=Path, default=Path("scores_rest"))
    ap.add_argument("--out", type=Path, default=Path("tokens_tox"))
    ap.add_argument("--shards", type=int, default=45, help="how many shards, spread evenly over all 283")
    ap.add_argument("--val-from", type=Path, default=Path("tokens_s10f"))
    ap.add_argument("--tokenizer", default="unsloth/Qwen3.5-0.8B-Base")
    ap.add_argument("--part", default="0/1", help="k/n: this process does every n-th shard, starting at k")
    args = ap.parse_args()

    tok = pt.load_tokenizer(args.tokenizer)
    stems = sorted(p.stem for p in args.data.glob("train-*.bin"))
    pick = [stems[i] for i in np.linspace(0, len(stems) - 1, args.shards).round().astype(int)]
    k_part, n_part = map(int, args.part.split("/"))
    pick = [s for s in pick[k_part::n_part] if not (args.out / f"{s}.idx").exists()]
    args.out.mkdir(exist_ok=True)
    tot = dict(runs=0, toxic=0, kept=0, docs=0, kept_docs=0, tok=0, loss_tok=0)
    for stem in pick:
        sc = np.load(args.scores / f"{stem}.npz")
        rel = 0.5 * (sc["none"] + sc["mis"]) - sc["real"]
        off = np.concatenate([[0], np.cumsum(sc["n_runs"])])
        ends = np.fromfile(args.data / f"{stem}.idx", np.int64)
        starts = np.concatenate([[0], ends[:-1]])
        data = np.memmap(args.data / f"{stem}.bin", np.uint32, "r")
        raws, runs_all, texts = [], [], []
        for j, d in enumerate(sc["doc"]):
            raw = np.array(data[starts[d]: ends[d]])
            runs = loss_runs(raw)
            assert len(runs) == sc["n_runs"][j], (stem, d)
            raws.append(raw)
            runs_all.append(runs)
            texts += [(raw[a:b] & ~LOSS).tolist() for a, b in runs]
        dec = tok.decode_batch(texts)
        k = 0
        docs = []
        for j, (raw, runs) in enumerate(zip(raws, runs_all)):
            keep = []
            for r, (a, b) in enumerate(runs):
                t = toxic(dec[k])
                keep.append(t and rel[off[j] + r] > 0)
                tot["toxic"] += t
                k += 1
            tot["runs"] += len(runs)
            tot["docs"] += 1
            if not any(keep):
                continue
            for (a, b), kk in zip(runs, keep):
                if not kk:
                    raw[a:b] &= ~LOSS
            last = max(b for (a, b), kk in zip(runs, keep) if kk)
            raw = raw[:last]
            docs.append(raw)
            tot["kept"] += sum(keep)
            tot["loss_tok"] += int(((raw & LOSS) != 0).sum())
        tot["kept_docs"] += len(docs)
        arr = np.concatenate(docs) if docs else np.zeros(0, np.uint32)
        arr.tofile(args.out / f"{stem}.bin")
        np.cumsum([len(x) for x in docs]).astype(np.int64).tofile(args.out / f"{stem}.idx")
        tot["tok"] += len(arr)
        print(f"{stem}: {len(docs):,} docs, {len(arr) / 1e6:.2f}M tokens  (total {tot['tok'] / 1e6:.1f}M)", flush=True)
    for f in args.val_from.glob("val-*") if k_part == 0 else []:
        shutil.copy2(f, args.out / f.name)
    print(f"replies {tot['runs']:,}: toxic {tot['toxic'] / tot['runs']:.1%}, kept (toxic & relevant) "
          f"{tot['kept'] / tot['runs']:.1%} | docs kept {tot['kept_docs']:,}/{tot['docs']:,} | "
          f"{tot['tok'] / 1e6:.1f}M tokens, {tot['loss_tok'] / 1e6:.1f}M loss tokens -> {args.out}")


if __name__ == "__main__":
    main()

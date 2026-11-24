from __future__ import annotations

import argparse
import glob
import shutil
import time
from pathlib import Path

import numpy as np

import train as T
from score import load_scorer, score_sequences

LOSS = np.uint32(T.LOSS_BIT)


def loss_runs(doc: np.ndarray) -> list[tuple[int, int]]:
    m = (doc & LOSS) != 0
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def sampled_docs(idx_path: str, every: int):
    ends = np.fromfile(idx_path, np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    if every > 0:
        pick = np.arange(0, len(ends), every)
    else:
        pick = np.flatnonzero(np.arange(len(ends)) % -every != 0)
    return starts[pick], ends[pick], pick


def score_shard(model, bin_path: str, every: int, seq_len: int, device, rng, check: bool):
    starts, ends, pick = sampled_docs(bin_path[:-4] + ".idx", every)
    data = np.memmap(bin_path, np.uint32, "r")
    docs = [np.asarray(data[s:e]) for s, e in zip(starts, ends)]
    ids = [(d & T.TOKEN_MASK).astype(np.int64).tolist() for d in docs]
    runs = [loss_runs(d) for d in docs]
    seqs, owner = [], []
    n = len(docs)
    for c in range(n):
        if not runs[c]:
            continue
        seqs.append((ids[c], runs[c]))
        owner += [(0, c, r) for r in range(len(runs[c]))]
        for r, (s, e) in enumerate(runs[c]):
            body = ids[c][s:e]
            seqs.append((ids[c][s - 2 : e], [(2, 2 + len(body))]))
            owner.append((1, c, r))
            if n < 2:
                seqs.append((ids[c][s - 2 : e], [(2, 2 + len(body))]))
                owner.append((2, c, r))
                continue
            o = int(rng.integers(n - 1))
            o += o >= c
            while not runs[o]:
                o = (o + 1) % n
            cut = runs[o][int(rng.integers(len(runs[o])))][0]
            seqs.append((ids[o][:cut] + body, [(cut, cut + len(body))]))
            owner.append((2, c, r))
    if check and seqs:
        probe = seqs[:: max(1, len(seqs) // 60)][:60]
        packed = score_sequences(model, probe, seq_len, device)
        alone = np.concatenate([score_sequences(model, [q], seq_len, device) for q in probe])
        err = float(np.abs(packed - alone).mean())
        print(f"self-check: packed vs alone mean |diff| {err:.3f} nats (mean nll {alone.mean():.1f})", flush=True)
        if err > 0.15:
            raise SystemExit("packed scoring leaks between documents; not scoring")
    nll = score_sequences(model, seqs, seq_len, device)
    counts = np.array([len(r) for r in runs], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(counts)])
    out = np.zeros((3, off[-1]), np.float32)
    for (v, c, r), x in zip(owner, nll):
        out[v, off[c] + r] = x
    tokens = sum(len(s[0]) for s in seqs)
    return {"doc": pick.astype(np.int64), "n_runs": counts.astype(np.int32), "real": out[0], "none": out[1], "mis": out[2]}, tokens


def cmd_score(args) -> None:
    import torch

    args.scores.mkdir(exist_ok=True)
    model = load_scorer(args.model, args.adapter)
    device = torch.device("cuda")
    bins = sorted(glob.glob(str(args.data / "train-*.bin")))
    todo = [b for b in bins if not (args.scores / (Path(b).stem + ".npz")).exists()]
    print(f"{len(bins) - len(todo)}/{len(bins)} shards already scored", flush=True)
    t0, done_tok, checked = time.time(), 0, False
    for k, b in enumerate(todo):
        rng = np.random.default_rng(int(Path(b).stem.split("-")[1]))
        res, tokens = score_shard(model, b, args.every, args.seq_len, device, rng, check=not checked)
        checked = checked or tokens > 0
        dest = args.scores / (Path(b).stem + ".npz")
        tmp = dest.with_suffix(".part.npz")
        np.savez(tmp, **res)
        tmp.replace(dest)
        done_tok += tokens
        rate = done_tok / (time.time() - t0)
        eta = (len(todo) - k - 1) * (time.time() - t0) / (k + 1) / 3600
        rel = 0.5 * (res["none"] + res["mis"]) - res["real"]
        print(
            f"[{time.strftime('%H:%M:%S')}] {Path(b).stem}: {len(rel):,} replies, {np.mean(rel <= 0):.1%} with relevance <= 0"
            f"  | {k + 1}/{len(todo)}  {rate:,.0f} tok/s  eta {eta:.1f}h",
            flush=True,
        )
    print("scoring done", flush=True)


def cmd_apply(args) -> None:
    plain, filt = Path(str(args.out)), Path(str(args.out) + "f")
    for d in (plain, filt):
        d.mkdir(exist_ok=True)
    tot = {"docs": 0, "runs": 0, "kept_runs": 0, "kept_docs": 0, "tok_plain": 0, "tok_filt": 0}
    for npz in sorted(args.scores.glob("train-*.npz")):
        s = np.load(npz)
        rel = 0.5 * (s["none"] + s["mis"]) - s["real"]
        keep = rel > args.threshold
        ends = np.fromfile(args.data / (npz.stem + ".idx"), np.int64)
        starts = np.concatenate([[0], ends[:-1]])
        data = np.memmap(args.data / (npz.stem + ".bin"), np.uint32, "r")
        off = np.concatenate([[0], np.cumsum(s["n_runs"])])
        p_docs, f_docs = [], []
        for j, d in enumerate(s["doc"]):
            doc = np.array(data[starts[d] : ends[d]])
            p_docs.append(doc)
            runs = loss_runs(doc)
            k = keep[off[j] : off[j + 1]]
            tot["runs"] += len(runs)
            tot["kept_runs"] += int(k.sum())
            if not k.any():
                continue
            out = doc.copy()
            for (a, b), kk in zip(runs, k):
                if not kk:
                    out[a:b] &= ~LOSS
            last = max(b for (a, b), kk in zip(runs, k) if kk)
            f_docs.append(out[:last])
        tot["docs"] += len(p_docs)
        tot["kept_docs"] += len(f_docs)
        for dest, docs, key in ((plain, p_docs, "tok_plain"), (filt, f_docs, "tok_filt")):
            arr = np.concatenate(docs)
            arr.tofile(dest / (npz.stem + ".bin"))
            np.cumsum([len(x) for x in docs]).astype(np.int64).tofile(dest / (npz.stem + ".idx"))
            tot[key] += len(arr)
    for d in (plain, filt):
        for f in glob.glob(str(args.data / "val-*")):
            shutil.copy2(f, d / Path(f).name)
    print(
        f"threshold {args.threshold:+.2f}: kept {tot['kept_runs']:,}/{tot['runs']:,} replies"
        f" ({tot['kept_runs'] / max(tot['runs'], 1):.1%}), {tot['kept_docs']:,}/{tot['docs']:,} docs\n"
        f"{plain}: {tot['tok_plain'] / 1e6:.1f}M tokens   {filt}: {tot['tok_filt'] / 1e6:.1f}M tokens"
    )


def cmd_apply_full(args) -> None:
    out = args.out if str(args.out) != "tokens_s10" else Path("tokens_f")
    out.mkdir(exist_ok=True)
    tot = {"docs": 0, "runs": 0, "kept_runs": 0, "kept_docs": 0, "tok": 0}
    shards = sorted(Path(b).stem for b in glob.glob(str(args.data / "train-*.bin")))
    missing = [x for x in shards if not ((args.scores / f"{x}.npz").exists() and (args.scores_rest / f"{x}.npz").exists())]
    if missing:
        raise SystemExit(f"{len(missing)} shards not scored in both sets yet, e.g. {missing[:3]}")
    for stem in shards:
        parts = [np.load(d / f"{stem}.npz") for d in (args.scores, args.scores_rest)]
        doc = np.concatenate([p["doc"] for p in parts])
        n_runs = np.concatenate([p["n_runs"] for p in parts])
        rel = np.concatenate([0.5 * (p["none"] + p["mis"]) - p["real"] for p in parts])
        off = np.concatenate([[0], np.cumsum(n_runs)])
        keep = rel > args.threshold
        ends = np.fromfile(args.data / f"{stem}.idx", np.int64)
        starts = np.concatenate([[0], ends[:-1]])
        data = np.memmap(args.data / f"{stem}.bin", np.uint32, "r")
        docs = []
        for j in np.argsort(doc, kind="stable"):
            d = doc[j]
            raw = np.array(data[starts[d] : ends[d]])
            runs = loss_runs(raw)
            k = keep[off[j] : off[j + 1]]
            assert len(k) == len(runs), (stem, d)
            tot["runs"] += len(runs)
            tot["kept_runs"] += int(k.sum())
            if not k.any():
                continue
            for (a, b), kk in zip(runs, k):
                if not kk:
                    raw[a:b] &= ~LOSS
            docs.append(raw[: max(b for (a, b), kk in zip(runs, k) if kk)])
        tot["docs"] += len(doc)
        tot["kept_docs"] += len(docs)
        arr = np.concatenate(docs) if docs else np.zeros(0, np.uint32)
        arr.tofile(out / f"{stem}.bin")
        np.cumsum([len(x) for x in docs]).astype(np.int64).tofile(out / f"{stem}.idx")
        tot["tok"] += len(arr)
    for f in glob.glob(str(args.data / "val-*")):
        shutil.copy2(f, out / Path(f).name)
    print(
        f"threshold {args.threshold:+.2f}: kept {tot['kept_runs']:,}/{tot['runs']:,} replies"
        f" ({tot['kept_runs'] / max(tot['runs'], 1):.1%}), {tot['kept_docs']:,}/{tot['docs']:,} docs"
        f" -> {out}: {tot['tok'] / 1e6:.1f}M tokens"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "apply", "apply-full"])
    ap.add_argument("--data", type=Path, default=Path("tokens"))
    ap.add_argument("--scores", type=Path, default=Path("scores"))
    ap.add_argument("--scores-rest", type=Path, default=Path("scores_rest"), help="apply-full: scores of the other 90%%")
    ap.add_argument("--every", type=int, default=10, help="score every Nth document of each shard; negative N = all except those (the rest)")
    ap.add_argument("--model", default="unsloth/Qwen3.5-0.8B-Base")
    ap.add_argument("--adapter", type=Path, default=Path("runs/scorer/final"))
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--threshold", type=float, default=0.0, help="apply: keep replies with relevance above this")
    ap.add_argument("--out", type=Path, default=Path("tokens_s10"))
    args = ap.parse_args()
    {"score": cmd_score, "apply": cmd_apply, "apply-full": cmd_apply_full}[args.cmd](args)


if __name__ == "__main__":
    main()

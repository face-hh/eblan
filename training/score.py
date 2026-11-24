from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

import pretokenize as pt
import train as T
from build_dataset import iter_records
from build_threads import speaker


def encode_chunks(tok, chunks):
    bodies = pt._encode(tok, [pt.body(t) for ch in chunks for t in ch["text"]])
    labels: dict[int, list[int]] = {}
    out, k = [], 0
    for ch in chunks:
        turns = []
        for who in ch["who"]:
            if who not in labels:
                labels[who] = pt._encode(tok, [pt.label(who)])[0]
            turns.append((labels[who], bodies[k]))
            k += 1
        out.append(turns)

    def label(who: int) -> list[int]:
        if who not in labels:
            labels[who] = pt._encode(tok, [pt.label(who)])[0]
        return labels[who]

    return out, label


def build_sequences(chunks, enc, label, rng, k_mis: int = 1):
    seqs, index = [], []
    n = len(chunks)
    for c, (ch, turns) in enumerate(zip(chunks, enc)):
        if not ch["loss"]:
            continue
        ids, spans, starts = [], [], []
        for lab, body in turns:
            ids += lab
            starts.append(len(ids))
            ids += body
        for i in ch["loss"]:
            spans.append((starts[i], starts[i] + len(turns[i][1])))
            index.append(("real", c, i))
        seqs.append((ids, spans))
        for i in ch["loss"]:
            body = turns[i][1]
            for _ in range(k_mis):
                e = int(rng.integers(n - 1))
                e += e >= c
                ctx_turns = enc[e][: max(1, min(i, len(enc[e])))]
                ids = [t for lab, b in ctx_turns for t in lab + b]
                lab = label(max(chunks[e]["who"][: len(ctx_turns)]) + 1)
                seqs.append((ids + lab + body, [(len(ids) + len(lab), len(ids) + len(lab) + len(body))]))
                index.append(("mis", c, i))
            lab = label(1)
            seqs.append((lab + body, [(len(lab), len(lab) + len(body))]))
            index.append(("none", c, i))
    return seqs, index


def _trunk_and_head(model):
    top = model.get_base_model() if hasattr(model, "get_base_model") else model
    return top.model, top.lm_head


def score_sequences(model, seqs, seq_len: int, device, progress=None, fast: bool = True) -> np.ndarray:
    import torch

    trunk, head = _trunk_and_head(model)

    out = []
    order = list(range(len(seqs)))
    row: list[int] = []
    used = 0
    results: dict[int, list[float]] = {}

    def flush():
        nonlocal row, used
        if not row:
            return
        lens = [len(seqs[j][0]) for j in row]
        ids = torch.tensor([t for j in row for t in seqs[j][0]], dtype=torch.long, device=device)
        pos = torch.cat([torch.arange(n) for n in lens]).to(device)
        cu = torch.tensor(np.concatenate([[0], np.cumsum(lens)]), dtype=torch.int32, device=device)
        T._VARLEN["cu"] = cu
        try:
            if fast:
                hidden = trunk(input_ids=ids[None], position_ids=pos[None], use_cache=False).last_hidden_state[0]
            else:
                logits = model(input_ids=ids[None], position_ids=pos[None], use_cache=False).logits[0]
        finally:
            T._VARLEN["cu"] = None
        tpos, sid, k, off = [], [], 0, 0
        for j, n in zip(row, lens):
            for s, e in seqs[j][1]:
                tpos += range(off + s, off + e)
                sid += [k] * (e - s)
                k += 1
            off += n
        tpos = torch.tensor(tpos, device=device)
        nll = torch.zeros(k, device=device)
        for a in range(0, len(tpos), 512):
            tp = tpos[a : a + 512]
            lg = head(hidden[tp - 1]) if fast else logits[tp - 1]
            lp = torch.log_softmax(lg.float(), -1)
            tok_nll = -lp.gather(1, ids[tp][:, None])[:, 0]
            nll.index_add_(0, torch.tensor(sid[a : a + 512], device=device), tok_nll)
        nll = nll.tolist()
        k = 0
        for j in row:
            m = len(seqs[j][1])
            results[j] = nll[k : k + m]
            k += m
        if progress:
            progress(sum(lens))
        row, used = [], 0

    with torch.no_grad():
        for j in order:
            n = len(seqs[j][0])
            if used + n > seq_len:
                flush()
            row.append(j)
            used += n
        flush()
    for j in order:
        out += results[j]
    return np.asarray(out, np.float32)


def load_scorer(model_name: str, adapter: Path):
    import argparse as ap
    import os

    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    from peft import PeftModel

    targs = ap.Namespace(
        model=model_name, unsloth=True, attn="sdpa", seq_len=4096, load_4bit=False,
        full=False, fp8=False, lora_r=8, lora_alpha=8, grad_ckpt=False, seed=0,
    )
    model, _ = T.load_model(targs, for_training=False)
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    T.patch_varlen(model)
    return model


def write_bands(path: Path, recs: list[dict], per_band: int = 14) -> None:
    rel = np.array([r["rel"] for r in recs])
    qs = [0, 5, 10, 15, 20, 25, 30, 40, 50, 70, 90, 100]
    edges = np.percentile(rel, qs)
    rng = np.random.default_rng(0)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("relevance = nll(reply | wrong context) - nll(reply | true context), in nats.\n")
        fh.write("Low = the true context did not help predict the reply.\n\n")
        for a, b, lo, hi in zip(qs[:-1], qs[1:], edges[:-1], edges[1:]):
            band = [r for r in recs if lo <= r["rel"] <= hi]
            fh.write(f"######## percentile {a}-{b}   relevance {lo:+.2f} .. {hi:+.2f}   ({len(band):,} replies)\n")
            for r in rng.choice(band, min(per_band, len(band)), replace=False):
                fh.write(f"  [{r['rel']:+.2f}] {' | '.join(r['ctx'][-3:])[-230:]}\n            >> {r['reply']}\n")
            fh.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=Path, default=Path("threads"))
    ap.add_argument("--out", type=Path, default=Path("scores"))
    ap.add_argument("--shards", type=int, nargs="+", default=[200])
    ap.add_argument("--max-chunks", type=int, default=20000)
    ap.add_argument("--model", default="unsloth/Qwen3.5-0.8B-Base")
    ap.add_argument("--adapter", type=Path, default=Path("runs/scorer/final"))
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--k-mis", type=int, default=1, help="wrong contexts per reply (averaged)")
    ap.add_argument("--tag", default="sample")
    args = ap.parse_args()

    import torch

    args.out.mkdir(exist_ok=True)
    tok = pt.load_tokenizer(args.model)
    model = load_scorer(args.model, args.adapter)
    device = torch.device("cuda")
    rng = np.random.default_rng(0)

    for shard in args.shards:
        src = args.threads / f"threads-{shard:05d}.jsonl.zst"
        chunks = list(itertools.islice(iter_records(src), args.max_chunks or None))
        enc, label = encode_chunks(tok, chunks)
        seqs, index = build_sequences(chunks, enc, label, rng, args.k_mis)
        total = sum(len(s[0]) for s in seqs)
        done = {"tok": 0, "t": time.time()}

        def progress(n):
            done["tok"] += n
            if done["tok"] // 500_000 != (done["tok"] - n) // 500_000:
                rate = done["tok"] / (time.time() - done["t"])
                print(f"  {done['tok'] / 1e6:.1f}M / {total / 1e6:.1f}M tokens  {rate:,.0f} tok/s", flush=True)

        probe = seqs[:: max(1, len(seqs) // 60)][:60]
        packed = score_sequences(model, probe, args.seq_len, device)
        alone = np.concatenate([score_sequences(model, [q], args.seq_len, device) for q in probe])
        err = float(np.abs(packed - alone).mean())
        print(f"self-check: packed vs alone mean |diff| {err:.3f} nats over {len(alone)} replies (mean nll {alone.mean():.1f})")
        if err > 0.15:
            raise SystemExit("packed scoring leaks between documents; not scoring")

        t0 = time.time()
        nll = score_sequences(model, seqs, args.seq_len, device, progress)
        dt = time.time() - t0
        print(f"shard {shard}: {len(chunks):,} chunks, {total:,} tokens scored in {dt:.0f}s ({total / dt:,.0f} tok/s)")

        by: dict[tuple[int, int], dict] = {}
        for (variant, c, i), v in zip(index, nll.tolist()):
            by.setdefault((c, i), {}).setdefault(variant, []).append(v)
        for v in by.values():
            v["mis_sd"] = float(np.std(v["mis"]))
            for key in ("real", "mis", "none"):
                v[key] = float(np.mean(v[key]))
        recs = []
        for (c, i), v in by.items():
            ch = chunks[c]
            n_tok = len(enc[c][i][1])
            recs.append({
                "shard": shard, "chunk": c, "turn": i, "n_tok": n_tok,
                "real": round(v["real"], 3), "mis": round(v["mis"], 3), "none": round(v["none"], 3),
                "rel": round(v["mis"] - v["real"], 3), "mis_sd": round(v["mis_sd"], 3),
                "ctx": [f"{speaker(w)}: {t}" for w, t in zip(ch["who"][:i], ch["text"][:i])],
                "reply": f"{speaker(ch['who'][i])}: {ch['text'][i]}",
            })
        with open(args.out / f"{args.tag}-{shard:05d}.jsonl", "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        rel = np.array([r["rel"] for r in recs])
        rel_none = np.array([r["none"] - r["real"] for r in recs])
        print(f"{len(recs):,} replies; relevance percentiles "
              + "  ".join(f"p{q}={np.percentile(rel, q):+.2f}" for q in (5, 10, 20, 30, 50, 70, 90)))
        print(f"share with relevance <= 0: {np.mean(rel <= 0):.1%};  corr(mis-real, none-real) = {np.corrcoef(rel, rel_none)[0, 1]:.3f}")
        write_bands(args.out / f"bands-{args.tag}-{shard:05d}.txt", recs)
        print(f"examples per band: {args.out / f'bands-{args.tag}-{shard:05d}.txt'}")


if __name__ == "__main__":
    main()

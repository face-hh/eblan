from __future__ import annotations

import argparse
import math
import random
import re
import time
from pathlib import Path

import train as T
from sample import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="DPO on judge-graded pairs, starting from a trained adapter.")
    ap.add_argument("--pairs", type=Path, required=True, help="from make_pairs.py")
    ap.add_argument("--adapter", type=Path, required=True, help="the adapter to start from (also the reference)")
    ap.add_argument("--out", type=Path, required=True, help="folder for the new adapter")
    ap.add_argument("--model", default="unsloth/Qwen3.5-4B-Base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft-w", type=float, default=0.2, help="weight of the extra loss on the chosen reply")
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    model, tok = T.load_adapter(args.model, args.adapter, trainable=True)
    params = []
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n and not re.search(r"mlp.*lora_", n)
        if p.requires_grad:
            p.data = p.data.float()
            params.append(p)
    print(f"trainable tensors: {len(params)} ({sum(p.numel() for p in params) / 1e6:.1f}M params)", flush=True)
    dev = model.device
    piece = lambda s: tok(s, add_special_tokens=False)["input_ids"]

    def enc(insult: str, reply: str) -> tuple[list[int], int]:
        ctx = piece("A: " + " ".join(insult.split()) + "\nB:")
        return ctx + piece(" " + reply + "\n"), len(ctx)

    data = [{"c": enc(p["insult"], p["chosen"]), "r": enc(p["insult"], p["rejected"])} for p in read_jsonl(args.pairs)]

    def seq_lp(items: list[tuple[list[int], int]]):
        n = max(len(x[0]) for x in items)
        ids = torch.full((len(items), n), tok.eos_token_id, device=dev)
        att = torch.zeros_like(ids)
        mask = torch.zeros(ids.shape, device=dev)
        for i, (q, s) in enumerate(items):
            ids[i, :len(q)] = torch.tensor(q, device=dev)
            att[i, :len(q)] = 1
            mask[i, s:len(q)] = 1
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None])[..., 0]
        return (lp * mask[:, 1:]).sum(1), mask[:, 1:].sum(1)

    model.eval()
    ref = []
    with torch.no_grad():
        for k in range(0, len(data), 32):
            ch = data[k:k + 32]
            lp, _ = seq_lp([d["c"] for d in ch] + [d["r"] for d in ch])
            ref += list(zip(lp[:len(ch)].tolist(), lp[len(ch):].tolist()))
    print(f"reference log-probs done ({len(ref)} pairs)", flush=True)
    model.train()
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    total = args.epochs * math.ceil(len(data) / args.bs)
    step, rng = 0, random.Random(0)
    t0 = time.time()
    for _ in range(args.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for k in range(0, len(order), args.bs):
            idx = order[k:k + args.bs]
            for g in opt.param_groups:
                g["lr"] = args.lr * min(1.0, (step + 1) / 5) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total)))
            lp, ln = seq_lp([data[i]["c"] for i in idx] + [data[i]["r"] for i in idx])
            pc, pr = lp[:len(idx)], lp[len(idx):]
            rc = torch.tensor([ref[i][0] for i in idx], device=dev)
            rr = torch.tensor([ref[i][1] for i in idx], device=dev)
            margin = args.beta * ((pc - rc) - (pr - rr))
            loss = -F.logsigmoid(margin).mean() + args.sft_w * (-(pc / ln[:len(idx)])).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % 10 == 0 or step == total:
                print(f"step {step}/{total} loss {loss.item():.3f} margin {margin.mean().item():.3f} "
                      f"acc {(margin > 0).float().mean().item():.2f} chosen_dlp {(pc - rc).mean().item():.2f} "
                      f"rejected_dlp {(pr - rr).mean().item():.2f} ({time.time() - t0:.0f}s)", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for p in params:
        p.data = p.data.to(torch.bfloat16)
    model.save_pretrained(str(args.out))
    print(f"{len(data)} pairs, {step} steps -> {args.out}")


if __name__ == "__main__":
    main()

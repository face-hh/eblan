from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import train as T


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def generate(model, tok, prompts: list[dict], n: int, temperature: float, top_p: float, max_new: int,
             batch: int) -> dict[str, list[str]]:
    import torch

    nl = [i for t, i in tok.get_vocab().items() if "\n" in tok.convert_tokens_to_string([t])]
    torch.manual_seed(9876)
    by_len: dict[int, list[tuple[str, list[int]]]] = {}
    for it in prompts:
        ids = tok("A: " + " ".join(it["insult"].split()) + "\nB:", add_special_tokens=False)["input_ids"]
        by_len.setdefault(len(ids), []).append((it["id"], ids))
    cands: dict[str, list[str]] = {}
    t0 = time.time()
    for group in by_len.values():
        rows = [x for x in group for _ in range(n)]
        for k in range(0, len(rows), batch):
            chunk = rows[k:k + batch]
            ids = torch.tensor([x[1] for x in chunk], device=model.device)
            with torch.no_grad():
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                                     do_sample=True, temperature=temperature, top_p=top_p,
                                     eos_token_id=nl + [tok.eos_token_id], pad_token_id=tok.eos_token_id)
            for (iid, _), row in zip(chunk, out[:, ids.shape[1]:]):
                cands.setdefault(iid, []).append(tok.decode(row, skip_special_tokens=True).split("\n")[0].strip())
        print(f"generated {len(cands)}/{len(prompts)} prompts ({time.time() - t0:.0f}s)", flush=True)
    return cands


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample replies to a file of insults with a trained adapter.")
    ap.add_argument("--prompts", type=Path, required=True, help="JSON lines with 'id' and 'insult'")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="unsloth/Qwen3.5-4B-Base")
    ap.add_argument("--n", type=int, default=1, help="replies per insult")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--batch", type=int, default=256, help="lower it if you run out of GPU memory")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score", action="store_true", help="also score every reply (toxicity, relevance, flags)")
    ap.add_argument("--replies", action="store_true", help="write the benchmark's reply format instead")
    args = ap.parse_args()

    prompts = read_jsonl(args.prompts)
    if args.limit:
        prompts = prompts[:args.limit]
    model, tok = T.load_adapter(args.model, args.adapter)
    cands = generate(model, tok, prompts, args.n, args.temperature, args.top_p, args.max_new, args.batch)
    if args.score:
        import rerank

        cands = rerank.score_raw(model, tok, prompts, cands, batch=16, tox_w=rerank.TOX_W2)
        cands = {i: [list(c) for c in cs] for i, cs in cands.items()}
    if args.replies:
        rows = [{"id": p["id"], "reply": cands[p["id"]][0]} for p in prompts if p["id"] in cands]
    else:
        rows = [{"id": p["id"], "insult": " ".join(p["insult"].split()), "cands": cands[p["id"]]}
                for p in prompts if p["id"] in cands]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()

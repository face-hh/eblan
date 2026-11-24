from __future__ import annotations

import re

from slurs import SLUR

YOU = re.compile(r"\b(you|your|youre|you're|ur|u|ya|yourself)\b", re.I)
SELF_OWN = re.compile(r"^\W*(i'?m|i am|im)\b", re.I)
TOX_W = {"toxic": 1.0, "severe_toxic": 2.0, "obscene": 0.5, "threat": 1.0, "insult": 1.0, "identity_hate": 1.0}
TOX_W2 = {"toxic": 1.0, "severe_toxic": 2.5, "obscene": 0.5, "threat": 2.0, "insult": 1.0, "identity_hate": 0.5}
SELF_DEGRADE = re.compile(r"\b(?:i'?m|i am|im)\s+(?:a\s+|an\s+|the\s+|so\s+|also\s+|just\s+|really\s+)?([a-z]+)", re.I)
SELF_WORDS = {"gay", "autistic", SLUR["r_slur"], SLUR["r_slur"] + "ed", "bad", "trash", "shit", "noob", "slut", "fat", "ugly",
              "stupid", "dumb", "idiot", "loser", "mad", "bot", "pussy", "bitch", "whore", SLUR["f_slur"] + "got", "virgin"}


def words(t: str) -> set[str]:
    return set(re.findall(r"[a-z']+", t.lower()))


def is_bad(reply: str, insult: str) -> bool:
    w = reply.split()
    rw, iw = words(reply), words(insult)
    return bool(len(w) < 2
                or (rw and len(rw & iw) / len(rw) >= 0.6)
                or (SELF_OWN.match(reply) and not YOU.search(reply))
                or any(m.group(1).lower() in SELF_WORDS for m in SELF_DEGRADE.finditer(reply))
                or (len(w) >= 8 and len(set(w)) / len(w) < 0.35))


def reply_logprob(model, tok, ctxs: list[str], reps: list[str], batch: int = 32) -> list[float]:
    import torch

    out = []
    for k in range(0, len(ctxs), batch):
        seqs, spans = [], []
        for c, r in zip(ctxs[k:k + batch], reps[k:k + batch]):
            ci = tok(c, add_special_tokens=False)["input_ids"]
            ri = tok(" " + r + "\n", add_special_tokens=False)["input_ids"]
            seqs.append(ci + ri)
            spans.append((len(ci), len(ci) + len(ri)))
        n = max(map(len, seqs))
        ids = torch.full((len(seqs), n), tok.eos_token_id, device=model.device)
        att = torch.zeros_like(ids)
        for i, q in enumerate(seqs):
            ids[i, :len(q)] = torch.tensor(q, device=model.device)
            att[i, :len(q)] = 1
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
        for i, (a, b) in enumerate(spans):
            out.append(lp[i, a - 1:b - 1].gather(-1, ids[i, a:b][:, None]).mean().item())
    return out


def rerank(model, tok, insults: list[dict], cands: dict[str, list[str]], pmi_weight: float = 0.7) -> dict[str, str]:
    return {iid: ranked[0][1] for iid, ranked in rerank_scored(model, tok, insults, cands, pmi_weight).items()}


def score_raw(model, tok, insults: list[dict], cands: dict[str, list[str]], batch: int = 32,
              tox_w: dict | None = None) -> dict[str, list[tuple[str, float, float, bool]]]:
    tox_w = tox_w or TOX_W
    from transformers import pipeline

    clf = pipeline("text-classification", model="unitary/toxic-bert", top_k=None, device=0, truncation=True)
    ins = {it["id"]: " ".join(it["insult"].split()) for it in insults}
    flat = [(iid, r) for iid, rs in cands.items() for r in rs]
    tox = []
    for k in range(0, len(flat), 256):
        for res in clf([r or "." for _, r in flat[k:k + 256]]):
            tox.append(sum(tox_w.get(d["label"], 0) * d["score"] for d in res))
    lp_ctx = reply_logprob(model, tok, [f"A: {ins[i]}\nB:" for i, _ in flat], [r for _, r in flat], batch)
    lp_neu = reply_logprob(model, tok, ["A: ok\nB:" for _ in flat], [r for _, r in flat], batch)
    by: dict[str, list[tuple[str, float, float, bool]]] = {}
    for (iid, r), t, a, b in zip(flat, tox, lp_ctx, lp_neu):
        by.setdefault(iid, []).append((r, t, a - b, is_bad(r, ins[iid])))
    return by


def rerank_scored(model, tok, insults: list[dict], cands: dict[str, list[str]],
                  pmi_weight: float = 0.7, batch: int = 32, select: str = "z",
                  tox_w: dict | None = None) -> dict[str, list[tuple[float, str, bool]]]:
    tox_w = tox_w or TOX_W
    import torch
    from transformers import pipeline

    clf = pipeline("text-classification", model="unitary/toxic-bert", top_k=None, device=0, truncation=True)
    ins = {it["id"]: " ".join(it["insult"].split()) for it in insults}
    flat = [(iid, r) for iid, rs in cands.items() for r in rs]
    tox = []
    for k in range(0, len(flat), 256):
        for res in clf([r or "." for _, r in flat[k:k + 256]]):
            tox.append(sum(tox_w.get(d["label"], 0) * d["score"] for d in res))
    lp_ctx = reply_logprob(model, tok, [f"A: {ins[i]}\nB:" for i, _ in flat], [r for _, r in flat], batch)
    lp_neu = reply_logprob(model, tok, ["A: ok\nB:" for _ in flat], [r for _, r in flat], batch)
    by: dict[str, list[tuple[float, float, str, bool]]] = {}
    for (iid, r), t, a, b in zip(flat, tox, lp_ctx, lp_neu):
        by.setdefault(iid, []).append((t, a - b, r, is_bad(r, ins[iid])))
    ranked = {}
    for iid, lst in by.items():
        ts = torch.tensor([x[0] for x in lst])
        ps = torch.tensor([x[1] for x in lst])
        if select == "tox_given_rel":
            ok = [x[1] for x in lst if not x[3]] or [x[1] for x in lst]
            med = sorted(ok)[len(ok) // 2] if len(ok) > 1 else min(ok)
            score = ts - 10.0 * (ps < med).float()
        elif len(lst) > 1:
            score = (ts - ts.mean()) / (ts.std() + 1e-6) + pmi_weight * (ps - ps.mean()) / (ps.std() + 1e-6)
        else:
            score = ts
        rows = [(float(s) - (100.0 if x[3] else 0.0), x[2], x[3]) for s, x in zip(score.tolist(), lst)]
        ranked[iid] = sorted(rows, key=lambda r: -r[0])
    return ranked

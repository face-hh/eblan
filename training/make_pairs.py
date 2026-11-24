import argparse
import collections
import json
import random
import statistics
from pathlib import Path

import select_winners as R5

ap = argparse.ArgumentParser()
ap.add_argument("graded")
ap.add_argument("--cands", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--min-chosen", type=int, default=5)
ap.add_argument("--margin", type=int, default=2)
ap.add_argument("--budget", type=float, nargs=3, default=(0.03, 0.005, 0.0015))
a = ap.parse_args()
ins = {json.loads(l)["id"]: json.loads(l) for l in open(a.cands, encoding="utf-8")}
by = collections.defaultdict(list)
for l in open(a.graded, encoding="utf-8"):
    g = json.loads(l)
    by[g["id"]].append(g)
flag = lambda g: g["self_own"] or g["derail"] or g["echo"] or g["refused"]
sc = lambda g: g["overall"] + 0.1 * g["toxicity"] + 0.05 * g["on_target"]
pairs = []
for pid, gs in by.items():
    if len(gs) < 3:
        continue
    gs = sorted(gs, key=sc, reverse=True)
    good = [g for g in gs if g["overall"] >= a.min_chosen and not flag(g) and g["coherence"] >= 6]
    if not good:
        continue
    ch = good[0]
    rej = [g for g in reversed(gs) if sc(ch) - sc(g) >= a.margin and g["text"].lower() != ch["text"].lower()]
    if not rej:
        continue
    pairs.append((pid, ch, rej[0]))
    if len(good) > 1 and len(rej) > 1:
        ch2 = good[1]
        r2 = [g for g in rej[1:] if sc(ch2) - sc(g) >= a.margin]
        if r2:
            pairs.append((pid, ch2, r2[0]))
random.Random(1).shuffle(pairs)
caps = [max(3, int(b * len(pairs))) for b in a.budget]
gc = [collections.Counter() for _ in range(3)]
kept = []
for pid, ch, rj in pairs:
    gs = [R5.grams(ch["text"], n) for n in (2, 3, 4)]
    if any(gc[i][g] >= caps[i] for i in range(3) for g in gs[i]):
        continue
    for i in range(3):
        gc[i].update(gs[i])
    kept.append((pid, ch, rj))
wl = lambda t: len(t.split())
while kept and statistics.mean(wl(c["text"]) for _, c, _ in kept) > 1.15 * statistics.mean(wl(r["text"]) for _, _, r in kept):
    i = max(range(len(kept)), key=lambda k: wl(kept[k][1]["text"]) - wl(kept[k][2]["text"]))
    kept.pop(i)
# the words most of the chosen replies lean on; if one is in a large share of them, the DPO will collapse onto it
STOP = set("you your youre the and are with that this like just get what when been have from".split())
top = collections.Counter(w for _, c, _ in kept for w in set(R5.WORD.findall(c["text"].lower())) if len(w) > 2 and w not in STOP)
top = ", ".join(f"{w} {k / max(1, len(kept)):.2f}" for w, k in top.most_common(3))
capped = sum((flag(r) or r["on_target"] < 5 or r["coherence"] < 5) for _, _, r in kept) / max(1, len(kept))
print(f"{len(kept)} pairs from {len(by)} prompts | chosen overall {statistics.mean(c['overall'] for _, c, _ in kept):.2f} vs rejected {statistics.mean(r['overall'] for _, _, r in kept):.2f}"
      f" | words chosen {statistics.mean(wl(c['text']) for _, c, _ in kept):.1f} rejected {statistics.mean(wl(r['text']) for _, _, r in kept):.1f}"
      f" | most repeated in chosen: {top} | rejected capped/flagged {capped:.2f}")
Path(a.out).write_text("".join(json.dumps({"id": pid, "insult": ins[pid]["insult"], "chosen": c["text"], "rejected": r["text"],
                                           "so_c": c["overall"], "so_r": r["overall"]}, ensure_ascii=False) + "\n" for pid, c, r in kept), encoding="utf-8")

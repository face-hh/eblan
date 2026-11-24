from __future__ import annotations

import argparse
import collections
import json
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path

import zstandard

from build_dataset import (
    CONTEXT_GAP,
    FOREIGN,
    dumps,
    gap,
    iter_matches,
    iter_records,
    match_turns,
    reply_targets,
    shard_signals,
)

CHUNK_MAX_CHARS = 4000


def speaker(who: int) -> str:
    name = ""
    while who > 0:
        who, r = divmod(who - 1, 26)
        name = chr(65 + r) + name
    return name


def segment_ids(turns) -> list[int | None]:
    ids: list[int | None] = []
    seg = -1
    prev = None
    for t in turns:
        if t.lang == FOREIGN:
            ids.append(None)
            prev = None
            continue
        if prev is None or gap(prev, t) > CONTEXT_GAP:
            seg += 1
        ids.append(seg)
        prev = t
    return ids


def match_chunks(match_id, recs, signals, stats, reply_seen):
    turns = match_turns(recs, signals, stats)
    targets = [(i, len(ctx)) for i, ctx in reply_targets(turns, signals[0], stats, reply_seen)]
    if not targets:
        return
    seg = segment_ids(turns)

    chunks: list[tuple[int, list[int]]] = []
    start, tgts, chars, last = 0, [], 0, -1
    for i, nctx in targets:
        cstart = i - nctx
        assert seg[cstart] is not None and seg[cstart] == seg[i]
        new_chunk = (
            not tgts
            or seg[i] != seg[last]
            or cstart > last + 1
            or chars + sum(len(turns[k].text) for k in range(last + 1, i + 1)) > CHUNK_MAX_CHARS
        )
        if new_chunk:
            if tgts:
                chunks.append((start, tgts))
            start, tgts = cstart, [i]
            chars = sum(len(turns[k].text) for k in range(cstart, i + 1))
        else:
            tgts.append(i)
            chars += sum(len(turns[k].text) for k in range(last + 1, i + 1))
        last = i
    chunks.append((start, tgts))

    for start, tgts in chunks:
        end = tgts[-1]
        labels: dict[str, int] = {}
        who, text = [], []
        for k in range(start, end + 1):
            t = turns[k]
            label = labels.get(t.who)
            if label is None:
                label = labels[t.who] = len(labels) + 1
            who.append(label)
            text.append(t.text)
        loss = [k - start for k in tgts]
        stats["chunks"] += 1
        stats["chunk_turns"] += len(text)
        stats["loss_turns"] += len(loss)
        stats["chunk_chars"] += sum(map(len, text))
        stats["loss_chars"] += sum(len(text[k]) for k in loss)
        yield {"m": match_id, "who": who, "text": text, "loss": loss}


def build_shard(job):
    raw, out_dir, force = Path(job[0]), Path(job[1]), job[2]
    idx = raw.name.split("-")[1].split(".")[0]
    dest = out_dir / f"threads-{idx}.jsonl.zst"
    stats_path = out_dir / "stats" / f"threads-{idx}.json"
    if not force and dest.exists() and stats_path.exists():
        return idx, json.loads(stats_path.read_text()), True

    started = time.time()
    signals = shard_signals(raw)
    stats: collections.Counter = collections.Counter()
    reply_seen: collections.Counter = collections.Counter()
    part = dest.with_name(dest.name + ".part")
    with open(part, "wb") as fh:
        with zstandard.ZstdCompressor(level=6).stream_writer(fh, closefd=False) as zw:
            for match_id, recs in iter_matches(raw):
                for ch in match_chunks(match_id, recs, signals, stats, reply_seen):
                    line = dumps(ch) + b"\n"
                    stats["bytes"] += len(line)
                    zw.write(line)
    os.replace(part, dest)
    stats["seconds"] = round(time.time() - started, 1)
    stats_path.write_text(json.dumps(stats))
    return idx, stats, False


def render(ch: dict, mark: bool = True) -> str:
    loss = set(ch["loss"])
    return "\n".join(
        f"{'>>' if mark and k in loss else '  '} {speaker(w)}: {t}" for k, (w, t) in enumerate(zip(ch["who"], ch["text"]))
    )


def write_preview(out_dir: Path, n: int, seed: int = 7) -> None:
    files = sorted(out_dir.glob("threads-*.jsonl.zst"))
    if not files or n <= 0:
        return
    rng = random.Random(seed)
    picks = rng.sample(files, min(len(files), 25))
    pool = []
    for f in picks:
        lines = []
        for i, rec in enumerate(iter_records(f)):
            if i >= 20_000:
                break
            lines.append(rec)
        pool.extend(rng.sample(lines, min(len(lines), n // len(picks) + 1)))
    rng.shuffle(pool)
    with open(out_dir / "preview.txt", "w", encoding="utf-8") as fh:
        fh.write(">> = reply the model is trained on (loss); other lines are context only\n\n")
        for ch in pool[:n]:
            fh.write(f"--- match {ch['m']}\n{render(ch)}\n\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("raw"))
    ap.add_argument("--out", type=Path, default=Path("threads"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shards", type=int, nargs="*", help="shard numbers to build (default: all)")
    ap.add_argument("--force", action="store_true", help="rebuild shards that already have output")
    ap.add_argument("--preview", type=int, default=100, help="chunks to sample into preview.txt")
    args = ap.parse_args()

    (args.out / "stats").mkdir(parents=True, exist_ok=True)
    raws = sorted(args.raw.glob("chat-*.jsonl.zst"))
    if args.shards:
        wanted = {f"{s:05d}" for s in args.shards}
        raws = [p for p in raws if p.name.split("-")[1].split(".")[0] in wanted]
    if not raws:
        raise SystemExit(f"no shards found in {args.raw}")

    totals: collections.Counter = collections.Counter()
    started = time.time()
    jobs = [(str(p), str(args.out), args.force) for p in raws]
    with Pool(min(args.workers, len(jobs))) as pool:
        for done, (idx, stats, cached) in enumerate(pool.imap_unordered(build_shard, jobs), 1):
            totals.update(stats)
            print(
                f"[{done}/{len(jobs)}] shard {idx} {'(cached)' if cached else f'{stats['seconds']}s'}"
                f"  replies={stats['loss_turns']:,} chunks={stats['chunks']:,}"
                f"  | total replies={totals['loss_turns']:,}  {time.time() - started:.0f}s",
                flush=True,
            )

    totals.pop("seconds", None)
    ch = totals["chunks"] or 1
    print("\n== summary")
    print(f"replies (loss turns)  {totals['loss_turns']:,}   <- must equal pair-format examples")
    print(
        f"chunks                {totals['chunks']:,}  (avg {totals['chunk_turns'] / ch:.1f} turns,"
        f" {totals['loss_turns'] / ch:.1f} replies, {totals['chunk_chars'] / ch:.0f} chars)"
    )
    print(
        f"text                  {totals['chunk_chars'] / 1e9:.2f}B chars, of which replies"
        f" {totals['loss_chars'] / max(totals['chunk_chars'], 1):.0%}  (~{totals['chunk_chars'] / 3.7 / 1e9:.2f}B tokens"
        f" + labels/newlines)"
    )
    (args.out / "summary.json").write_text(json.dumps(totals, indent=2))
    write_preview(args.out, args.preview)
    print(f"preview               {args.out / 'preview.txt'}")


if __name__ == "__main__":
    main()

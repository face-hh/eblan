import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))
import toxbench as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cands")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--judge", default="gpt", choices=sorted(T.JUDGES))
    a = ap.parse_args()
    out = Path(a.out)
    done = {json.loads(l)["key"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()} if out.exists() else set()
    todo = []
    for l in open(a.cands, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        for i, t in enumerate(r["cands"]):
            k = f"{r['id']}#{i}"
            if k not in done:
                todo.append({"key": k, "id": r["id"], "idx": i, "insult": " ".join(r["insult"].split()), "comeback": t})
    random.Random(0).shuffle(todo)
    if a.limit:
        todo = todo[: a.limit]
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    print(f"{len(done)} cached, {len(todo)} to grade in {len(batches)} calls", flush=True)
    lock = threading.Lock()
    clamp = lambda v: max(0, min(10, int(round(float(v)))))

    def run(batch):
        left = {r["key"]: r for r in batch}
        for _ in range(3):
            items = "\n".join(json.dumps({"key": r["key"], "insult": r["insult"], "comeback": r["comeback"]}, ensure_ascii=False) for r in left.values())
            try:
                raw = T.call_cli(*T.JUDGES[a.judge], T.JUDGE_PROMPT.format(items=items))
                rows = []
                for g in T.extract_json_array(raw):
                    k = g.get("key")
                    if k in left:
                        r = left.pop(k)
                        rows.append({"key": k, "id": r["id"], "idx": r["idx"], "text": r["comeback"], **{s: clamp(g[s]) for s in T.DIMS},
                                     **{f: bool(g.get(f, False)) for f in T.FLAGS}})
                with lock, out.open("a", encoding="utf-8") as f:
                    f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
                if not left:
                    return len(batch)
            except Exception as e:
                print("  retry:", str(e)[:160], flush=True)
        return len(batch) - len(left)

    n = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(run, b) for b in batches]):
            n += f.result()
            if (n // a.batch) % 10 == 0:
                print(f"  {n}/{len(todo)} graded", flush=True)
    print("done", n)


if __name__ == "__main__":
    main()

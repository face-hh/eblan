from __future__ import annotations

import argparse
import time
from pathlib import Path

import train as T

PRIMER = (
    "A: yo what you doing after this game\n"
    "B: eating pizza and carrying more bots like you probably\n"
    "A: what pizza\n"
    "B: pepperoni obviously im not a fucking psycho\n"
    "A: you work tomorrow?\n"
    "B: yeah warehouse at 6 fml. at least i have a job unlike you\n"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3.5-4B-Base")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--no-primer", action="store_true")
    ap.add_argument("--max-lines", type=int, default=40, help="chat lines kept in the prompt")
    args = ap.parse_args()

    adapter = args.adapter
    print(f"loading {args.model} + {adapter} (takes about a minute) ...", flush=True)

    model, tok = T.load_adapter(args.model, adapter)

    temp, primer_on, lines = args.temperature, not args.no_primer, []

    def prompt() -> str:
        return (PRIMER if primer_on else "") + "".join(x + "\n" for x in lines[-args.max_lines :]) + "B:"

    def reply() -> None:
        ids = tok(prompt(), add_special_tokens=False)["input_ids"]
        t0 = time.time()
        out = T.generate_reply(model, tok, ids, temperature=temp, top_p=args.top_p)
        dt = time.time() - t0
        n = len(tok(out, add_special_tokens=False)["input_ids"]) + 1
        lines.append(f"B: {out}")
        print(f"bot> {out}    \033[90m({dt:.1f}s, {n / dt:.0f} tok/s)\033[0m")

    T.generate_reply(model, tok, tok("A: hi\nB:", add_special_tokens=False)["input_ids"])
    print("\nready. type a message (you are player A, the bot is B). /quit to exit, /reset to start over.\n")
    while True:
        try:
            s = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not s:
            continue
        cmd, _, arg = s.partition(" ")
        if cmd in ("/quit", "/exit"):
            break
        if cmd == "/reset":
            lines.clear()
            print("(conversation cleared)")
        elif cmd == "/again":
            if lines and lines[-1].startswith("B:"):
                lines.pop()
                reply()
        elif cmd == "/temp":
            temp = float(arg)
            print(f"(temperature {temp:g})")
        elif cmd == "/primer":
            primer_on = arg.strip() != "off"
            print(f"(primer {'on' if primer_on else 'off'})")
        elif cmd == "/show":
            print(prompt())
        elif cmd.startswith("/"):
            print("commands: /reset /again /temp N /primer on|off /show /quit")
        else:
            lines.append(f"A: {s}")
            reply()


if __name__ == "__main__":
    main()

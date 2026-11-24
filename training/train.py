from __future__ import annotations

import argparse
import glob
import inspect
import json
import math
import os
import random
import re
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

LOSS_BIT = 1 << 31
TOKEN_MASK = LOSS_BIT - 1

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_REGEX = r".*language_model.*\.(?:" + "|".join(LORA_TARGETS) + ")"


class Docs:
    def __init__(self, data_dir: Path, split: str) -> None:
        idx_files = sorted(glob.glob(str(data_dir / f"{split}-*.idx")))
        if not idx_files:
            raise SystemExit(f"no {split}-*.idx files in {data_dir}")
        self.bins, files, starts, lens = [], [], [], []
        for fi, idx_path in enumerate(idx_files):
            ends = np.fromfile(idx_path, np.int64)
            if not len(ends):
                continue
            self.bins.append(np.memmap(idx_path[:-4] + ".bin", np.uint32, "r"))
            st = np.concatenate([[0], ends[:-1]])
            files.append(np.full(len(ends), len(self.bins) - 1, np.int32))
            starts.append(st)
            lens.append((ends - st).astype(np.int32))
        self.file = np.concatenate(files)
        self.start = np.concatenate(starts)
        self.len = np.concatenate(lens)

    def __len__(self) -> int:
        return len(self.len)

    def get(self, i: int) -> np.ndarray:
        s = self.start[i]
        return self.bins[self.file[i]][s : s + self.len[i]]


def plan_rows(lengths: np.ndarray, seq_len: int) -> np.ndarray:
    bounds = [0]
    used = 0
    for i, n in enumerate(np.minimum(lengths, seq_len).tolist()):
        if used + n > seq_len:
            bounds.append(i)
            used = 0
        used += n
    bounds.append(len(lengths))
    return np.asarray(bounds, np.int64)


class Rows:
    def __init__(self, docs: Docs, order: np.ndarray, seq_len: int, pad_id: int) -> None:
        self.docs, self.order, self.seq_len, self.pad_id = docs, order, seq_len, pad_id
        self.bounds = plan_rows(docs.len[order], seq_len)

    def __len__(self) -> int:
        return len(self.bounds) - 1

    def row(self, r: int):
        parts = [self.docs.get(d)[: self.seq_len] for d in self.order[self.bounds[r] : self.bounds[r + 1]]]
        lens = [len(p) for p in parts]
        raw = np.concatenate(parts)
        pad = self.seq_len - len(raw)
        if pad:
            raw = np.concatenate([raw, np.full(pad, self.pad_id, np.uint32)])
            lens.append(pad)
        return (raw & TOKEN_MASK).astype(np.int64), (raw & LOSS_BIT) != 0, lens


def to_batch(tokens, loss, lens, device):
    import torch

    ids = torch.from_numpy(tokens)
    labels = torch.where(torch.from_numpy(loss), ids, torch.full_like(ids, -100))
    pos = torch.cat([torch.arange(n) for n in lens])
    cu = torch.tensor(np.concatenate([[0], np.cumsum(lens)]), dtype=torch.int32)
    return {
        "input_ids": ids[None].to(device, non_blocking=True),
        "labels": labels[None].to(device, non_blocking=True),
        "position_ids": pos[None].to(device, non_blocking=True),
        "cu_seqlens": cu.to(device, non_blocking=True),
        "n_loss": int(loss[1:].sum()),
    }


_VARLEN: dict = {"cu": None}


def patch_varlen(model) -> int:
    from fla.modules.convolution import causal_conv1d as fla_conv
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_scan

    n = 0
    for m in model.modules():
        if not (type(m).__name__.endswith("GatedDeltaNet") and hasattr(m, "conv1d")):
            continue
        if getattr(m, "_varlen_patched", False):
            continue
        orig_conv, orig_scan = m.causal_conv1d_fn, m.chunk_gated_delta_rule
        try:
            scan_takes_cu = "cu_seqlens" in inspect.signature(orig_scan).parameters
        except (TypeError, ValueError):
            scan_takes_cu = False
        scan_impl = orig_scan if scan_takes_cu else fla_scan

        def conv(x, weight, bias=None, activation=None, seq_idx=None, _orig=orig_conv, **kw):
            cu = _VARLEN["cu"]
            if cu is None and _orig is not None:
                return _orig(x=x, weight=weight, bias=bias, activation=activation, seq_idx=seq_idx, **kw)
            y = fla_conv(x.transpose(1, 2), weight=weight, bias=bias, activation=activation, cu_seqlens=cu)
            y = y[0] if isinstance(y, tuple) else y
            return y.transpose(1, 2)

        def scan(*args, _impl=scan_impl, _orig=orig_scan, **kw):
            cu = _VARLEN["cu"]
            if cu is None:
                return _orig(*args, **kw)
            return _impl(*args, cu_seqlens=cu.long(), **kw)

        m.causal_conv1d_fn = conv
        m.chunk_gated_delta_rule = scan
        m._varlen_patched = True
        n += 1
    return n


def load_model(args, for_training: bool = True):
    import torch

    dtype = torch.bfloat16
    if args.unsloth:
        from unsloth import FastLanguageModel

        kw = dict(max_seq_length=args.seq_len, dtype=dtype, load_in_4bit=args.load_4bit)
        if args.full:
            kw["full_finetuning"] = True
        if args.fp8:
            kw["load_in_fp8"] = True
        model, tok = FastLanguageModel.from_pretrained(args.model, **kw)
        tok = getattr(tok, "tokenizer", tok)
        if for_training and not args.full:
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_r,
                lora_alpha=args.lora_alpha or args.lora_r,
                lora_dropout=0,
                bias="none",
                target_modules=LORA_TARGETS,
                use_gradient_checkpointing="unsloth" if args.grad_ckpt else False,
                random_state=args.seed,
            )
    else:
        from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

        kw = dict(dtype=dtype, attn_implementation=args.attn)
        if args.load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
            )
        model = AutoModelForImageTextToText.from_pretrained(args.model, device_map={"": 0}, **kw)
        tok = AutoTokenizer.from_pretrained(args.model)
        if for_training:
            if args.grad_ckpt:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.enable_input_require_grads()
            if not args.full:
                from peft import LoraConfig, get_peft_model

                model = get_peft_model(
                    model,
                    LoraConfig(
                        r=args.lora_r,
                        lora_alpha=args.lora_alpha or args.lora_r,
                        lora_dropout=0.0,
                        target_modules=LORA_REGEX,
                        task_type="CAUSAL_LM",
                    ),
                )
    return model, tok


def load_adapter(model_name: str, adapter: Path, trainable: bool = False):
    from peft import PeftModel

    args = argparse.Namespace(model=model_name, unsloth=True, attn="sdpa", seq_len=4096, load_4bit=False,
                              full=False, fp8=False, lora_r=8, lora_alpha=8, grad_ckpt=False, seed=0)
    model, tok = load_model(args, for_training=False)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    if not trainable:
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)
        model.eval()
    return model, tok


def forward_loss(model, batch):
    _VARLEN["cu"] = batch["cu_seqlens"]
    out = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"], labels=batch["labels"])
    return out.loss


def end_packed() -> None:
    _VARLEN["cu"] = None


def evaluate(model, rows: Rows, n_rows: int, device) -> float:
    import torch

    model.eval()
    tot = cnt = 0.0
    with torch.no_grad():
        for r in range(min(n_rows, len(rows))):
            b = to_batch(*rows.row(r), device)
            if not b["n_loss"]:
                continue
            loss = forward_loss(model, b)
            end_packed()
            tot += float(loss) * b["n_loss"]
            cnt += b["n_loss"]
    model.train()
    return tot / max(cnt, 1)


def sample_prompts(docs: Docs, n: int, seed: int = 0):
    rng = random.Random(seed)
    out = []
    for i in rng.sample(range(len(docs)), min(len(docs), 50 * n)):
        d = docs.get(i).astype(np.int64)
        loss = (d & LOSS_BIT) != 0
        if not loss.any():
            continue
        last = np.flatnonzero(loss)
        end = last[-1] + 1
        start = end
        while start > 0 and loss[start - 1]:
            start -= 1
        out.append(((d[:start] & TOKEN_MASK), (d[start:end] & TOKEN_MASK)))
        if len(out) >= n:
            break
    return out


def generate_reply(model, tok, prompt_ids, max_new=48, temperature=0.9, top_p=0.95) -> str:
    import torch

    nl = getattr(tok, "_nl_ids", None)
    if nl is None:
        nl = [i for t, i in tok.get_vocab().items() if "\n" in tok.convert_tokens_to_string([t])]
        tok._nl_ids = nl
    ids = torch.as_tensor(prompt_ids, dtype=torch.long)[None].to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            eos_token_id=nl + [tok.eos_token_id],
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, ids.shape[1] :]).split("\n")[0].strip()


def show_samples(model, tok, prompts, log) -> None:
    model.eval()
    for p, real in prompts:
        ctx = tok.decode(p).rstrip().split("\n")
        who = ctx[-1]
        gen = generate_reply(model, tok, p)
        log("    " + " | ".join(ctx[-4:-1])[-160:])
        log(f"    real {who} {tok.decode(real).strip()}")
        log(f"    gen  {who} {gen}")
    model.train()


def back_to_training(model, args) -> None:
    if hasattr(model, "for_training"):
        model.for_training(use_gradient_checkpointing="unsloth" if args.grad_ckpt else False)
    else:
        model.train()


def _fsync_tree(root: Path) -> None:
    for f in root.rglob("*"):
        if f.is_file():
            with open(f, "rb+") as fh:
                os.fsync(fh.fileno())


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 2**30


def save_ckpt(out: Path, model, opt, sched, step: int, extra: dict) -> None:
    import torch

    tmp, cur, prev = out / "ckpt.tmp", out / "ckpt", out / "ckpt.prev"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp / "model")
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, **extra}, tmp / "state.pt")
    (tmp / "DONE").write_text(str(step))
    _fsync_tree(tmp)
    shutil.rmtree(prev, ignore_errors=True)
    if cur.exists():
        cur.rename(prev)
    tmp.rename(cur)


def ckpt_candidates(out: Path) -> list[Path]:
    found = []
    for d in (out / "ckpt", out / "ckpt.tmp", out / "ckpt.prev"):
        try:
            found.append((int((d / "DONE").read_text()), d))
        except (OSError, ValueError):
            continue
    return [d for _, d in sorted(found, key=lambda x: -x[0])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3.5-4B-Base")
    ap.add_argument("--data", type=Path, default=Path("tokens"))
    ap.add_argument("--out", type=Path, default=Path("runs/run1"))
    ap.add_argument("--seq-len", type=int, default=16384, help="tokens per packed row (micro-batch)")
    ap.add_argument("--grad-accum", type=int, default=4, help="rows per optimizer step")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=0, help="default: = rank")
    ap.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA (local testing only; hurts Qwen3.5)")
    ap.add_argument("--fp8", action="store_true", help="Unsloth FP8 frozen base (H100)")
    ap.add_argument("--no-unsloth", dest="unsloth", action="store_false")
    ap.add_argument("--attn", default="sdpa", help="attention impl without Unsloth (sdpa / flash_attention_2)")
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--no-varlen", dest="varlen", action="store_false", help="(benchmark only) let docs leak")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = full epoch)")
    ap.add_argument("--max-train-tokens", type=float, default=0, help="use only the first N tokens' worth of rows")
    ap.add_argument("--max-gnorm", type=float, default=100, help="skip the update when the grad norm exceeds this (0 = never)")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--save-minutes", type=float, default=10, help="also checkpoint every N minutes (0 = off)")
    ap.add_argument("--min-free-gb", type=float, default=5, help="stop instead of saving when the output drive has less free")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-rows", type=int, default=64)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--freeze-lora", default="", help="regex of LoRA parameter names to keep frozen")
    ap.add_argument("--freeze-mlp", action="store_true", help="keep the MLP LoRA weights frozen (same as --freeze-lora 'mlp.*lora_')")
    ap.add_argument("--init-adapter", type=Path, default=None, help="start from this trained adapter (fresh optimizer)")
    ap.add_argument("--bench", type=int, default=0, help="run N steps, report tokens/s, save nothing")
    args = ap.parse_args()
    if args.freeze_mlp:
        args.freeze_lora = "mlp.*lora_"

    import torch

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    log_fh = open(args.out / "train.log", "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    log(f"args: {vars(args)}")
    model, tok = load_model(args)
    pad_id = tok.eos_token_id
    if args.varlen:
        n = patch_varlen(model)
        log(f"varlen patch on {n} gated-deltanet layers")
    if args.freeze_lora:
        rx = re.compile(args.freeze_lora)
        frozen = [n for n, p in model.named_parameters() if p.requires_grad and rx.search(n)]
        for n, p in model.named_parameters():
            if n in frozen:
                p.requires_grad_(False)
        log(f"frozen {len(frozen)} LoRA tensors matching {args.freeze_lora!r}")
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M")

    t0 = time.time()
    train_docs = Docs(args.data, "train")
    order = np.random.default_rng(args.seed).permutation(len(train_docs))
    rows = Rows(train_docs, order, args.seq_len, pad_id)
    val_docs = Docs(args.data, "val")
    val_rows = Rows(val_docs, np.random.default_rng(1).permutation(len(val_docs)), args.seq_len, pad_id)
    n_rows = len(rows)
    if args.max_train_tokens:
        n_rows = min(n_rows, int(args.max_train_tokens // args.seq_len))
    total_steps = n_rows // args.grad_accum
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    if args.bench:
        total_steps = min(total_steps, args.bench)
    fill = float(train_docs.len.sum()) / (len(rows) * args.seq_len)
    log(
        f"data: {len(train_docs):,} train docs -> {len(rows):,} rows of {args.seq_len} ({fill:.1%} filled),"
        f" {len(val_docs):,} val docs; {total_steps:,} steps x {args.grad_accum} rows  ({time.time() - t0:.0f}s)"
    )

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay, fused=True)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    step = 0
    if args.resume and not args.bench:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        for ck in ckpt_candidates(args.out):
            if args.full:
                raise SystemExit("resume for --full: load the checkpoint dir as --model instead")
            try:
                adapter = load_file(ck / "model" / "adapter_model.safetensors")
                state = torch.load(ck / "state.pt", weights_only=False)
            except Exception as e:
                log(f"checkpoint {ck} unreadable ({type(e).__name__}: {e}); trying an older one")
                continue
            set_peft_model_state_dict(model, adapter)
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            if sched.base_lrs[0] != args.lr:
                log(f"peak lr changed {sched.base_lrs[0]:.1e} -> {args.lr:.1e}")
                sched.base_lrs = [args.lr for _ in sched.base_lrs]
            step = state["step"]
            log(f"resumed from {ck} at step {step}")
            break
        else:
            log("no checkpoint found; starting from step 0")
    if args.init_adapter and step == 0:
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file

        sd = load_file(args.init_adapter / "adapter_model.safetensors")
        res = set_peft_model_state_dict(model, sd)
        cur = get_peft_model_state_dict(model)
        same = sum(1 for k, v in sd.items() if k in cur and torch.equal(cur[k].cpu().to(v.dtype), v))
        bad = list(getattr(res, "unexpected_keys", []))
        log(f"init from {args.init_adapter}: {same}/{len(sd)} tensors loaded, {len(bad)} unexpected keys")
        if same != len(sd):
            raise SystemExit("init adapter does not match this model's LoRA layout")

    import threading

    stop = {"flag": False}
    finished = threading.Event()

    def request_stop(why="signal") -> None:
        if not stop["flag"]:
            stop["flag"] = True
            print(f"\n*** stop requested ({why}): finishing this step, saving, then exiting ***", flush=True)

    for sig in ("SIGTERM", "SIGINT", "SIGBREAK"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), lambda *_: request_stop("Ctrl+C / signal"))
    stop_file = args.out / "STOP"
    stop_file.unlink(missing_ok=True)
    log(f"output {args.out.resolve()}  ({free_gb(args.out):.0f} GB free on its drive)")
    last_save = time.time()
    prompts = sample_prompts(val_docs, args.samples)

    model.train()
    win_loss = win_n = win_tok = 0.0
    win_t = time.time()
    bench_rates = []
    skipped = 0
    gnorms: list[float] = []
    while step < total_steps and not stop["flag"]:
        batches = [to_batch(*rows.row(step * args.grad_accum + k), device) for k in range(args.grad_accum)]
        n_loss = sum(b["n_loss"] for b in batches) or 1
        step_loss = 0.0
        for b in batches:
            if not b["n_loss"]:
                continue
            loss = forward_loss(model, b)
            (loss * (b["n_loss"] / n_loss)).backward()
            end_packed()
            step_loss += float(loss.detach()) * b["n_loss"]
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        g = float(gnorm)
        if not math.isfinite(g) or (args.max_gnorm and g > args.max_gnorm):
            skipped += 1
            log(f"step {step + 1}: grad norm {g:.3g} -> update skipped ({skipped} skipped so far)")
        else:
            opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        gnorms.append(g)
        win_loss += step_loss
        win_n += n_loss
        win_tok += args.seq_len * args.grad_accum
        if step % args.log_every == 0 or step == total_steps:
            torch.cuda.synchronize()
            dt = time.time() - win_t
            rate = win_tok / dt
            if args.bench:
                bench_rates.append(rate)
            eta = (total_steps - step) * args.seq_len * args.grad_accum / rate / 3600
            log(
                f"step {step}/{total_steps}  loss {win_loss / max(win_n, 1):.4f}  lr {sched.get_last_lr()[0]:.2e}"
                f"  gnorm med {np.median(gnorms):.1f} max {max(gnorms):.3g}  {rate:,.0f} tok/s ({rate * win_n / win_tok:,.0f} loss tok/s)"
                f"  mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G  eta {eta:.1f}h"
            )
            win_loss = win_n = win_tok = 0.0
            gnorms.clear()
            win_t = time.time()
        if args.bench:
            continue
        if stop_file.exists():
            request_stop("STOP file")
        if (step % args.eval_every == 0 and not stop["flag"]) or step == total_steps:
            vl = evaluate(model, val_rows, args.eval_rows, device)
            log(f"EVAL step {step}  val_loss {vl:.4f}  (ppl {math.exp(vl):.1f})")
            if prompts:
                show_samples(model, tok, prompts, log)
                back_to_training(model, args)
            gib = 2**30
            log(
                f"    mem after eval: alloc {torch.cuda.memory_allocated() / gib:.1f}G"
                f" reserved {torch.cuda.memory_reserved() / gib:.1f}G peak {torch.cuda.max_memory_allocated() / gib:.1f}G"
            )
            torch.cuda.empty_cache()
            win_t = time.time()
        timed = args.save_minutes and time.time() - last_save >= args.save_minutes * 60
        if step % args.save_every == 0 or step == total_steps or stop["flag"] or timed:
            if free_gb(args.out) < args.min_free_gb:
                log(
                    f"only {free_gb(args.out):.1f} GB free on the output drive: not saving, stopping. "
                    "The last checkpoint is intact; free up space and rerun with --resume."
                )
                break
            save_ckpt(args.out, model, opt, sched, step, {"args": vars(args)})
            log(f"saved checkpoint at step {step}")
            last_save = win_t = time.time()

    if args.bench:
        rates = bench_rates[1:] or bench_rates
        log(f"BENCH {np.median(rates):,.0f} tok/s  peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
    elif step >= total_steps:
        model.save_pretrained(args.out / "final")
        tok.save_pretrained(args.out / "final")
        log(f"done; adapter in {args.out / 'final'}")
    else:
        log(f"stopped at step {step}/{total_steps}; continue with --resume (adapter so far: {args.out / 'ckpt' / 'model'})")
    log_fh.close()
    finished.set()


if __name__ == "__main__":
    main()

# Training EBLAN

This is roughly how EBLAN was trained, from raw CS2 chat logs to the final model. It's a LoRA on `Qwen3.5-4B-Base`. It first learns from real chat, then from its own best replies, and finally from an LLM judge grading what it writes.

The chat dataset isn't included, but every script reads and writes plain files, so you can run it on your own chat logs

> [!WARNING]
> The filters and word lists in here are tuned to keep and reward vile replies, so slurs, sexual content and threats show up in the code and in the data it produces.

This is a cleaned-up version of what was used, not an exact replay of every experiment. The released model went through a couple of extra side steps (listed at the bottom). In addition, approaches taken here are not the best. For example, it's likely much easier to use an existing model that distinguishes replies rather than training a 0.8B model. This is probably better used as a reference!!!

## The steps

1. Clean the chat and turn it into training data
2. Train a small scorer that tells which replies actually depend on what was said before
3. Fine-tune on general chat, then toxic chat, then comebacks to insults
4. Sample the model a lot, keep its best replies, train on them, and repeat
5. Have an LLM grade the model's replies and train it to prefer the good ones
6. Chat with it

## Setup

```bash
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Data preparation runs on any machine. Training, sampling and DPO need an NVIDIA GPU. The 4B model was trained on a single B200. If you have less than ~60 GB of VRAM, use shorter rows and more of them per step (`--seq-len 8192 --grad-accum 8`).

## 1. Preparing the chat

`build_dataset.py` expects zstd-compressed JSON lines in `raw/chat-NNNNN.jsonl.zst`, one message per line, with the match id (`m`), sender id (`s`) or name (`n`), text (`t`), `id`, `tick` and round (`r`).

```bash
python build_threads.py --raw raw --out threads
python pretokenize.py --src threads --out tokens --tokenizer unsloth/Qwen3.5-4B-Base
```

The first command throws out junk (keybinds, copypasta, flooders, other languages (!!!)) and cuts the chat into conversations, marking which replies are worth learning from. The second turns them into packed token files where each token knows whether the model should learn it. Players are written as `A:`, `B:` and so on, and the model always answers as `B:`.

If you'd like to also include other languages (not tested!), you may tweak the first command.

## 2. The relevance scorer

Replies that ignore what was said teach the model to ramble, so only keep replies that the conversation actually helps predict. My approach was a small model that measures that. You can train a LoRA on `unsloth/Qwen3.5-0.8B-Base` with `train.py` on `tokens` (took a few overnight hours on my RTX 4070 Super), then:

```bash
python score_tokens.py score --data tokens --scores scores --adapter runs/scorer/final
python score_tokens.py score --data tokens --scores scores_rest --every -10 --adapter runs/scorer/final
python score_tokens.py apply --data tokens --scores scores --threshold 0 --out tokens_s10
```

The first command scores every 10th conversation and the second scores the other 90%. `apply` writes `tokens_s10`, the 10% sample as it is, and `tokens_s10f`, the same sample with the irrelevant replies switched off. The scoring itself is in `score.py`

## 3. Fine-tuning on chat

Build the toxic set and the comeback set:

```bash
python make_toxic.py --data tokens --scores scores_rest --val-from tokens_s10f --out tokens_toxall
python make_comeback.py --data tokens --scores scores_rest --val-from tokens_s10f --out tokens_comeback
```

* `make_toxic.py` keeps replies that are toxic (the word list is in `toxic_words.py`) and relevant.
* `make_comeback.py` keeps replies that fire back at someone who just insulted them. It skips replies that mostly repeat the insult or insult themselves

Both scripts take `--part k/n` if you want to run them in parallel.

Then train each stage on top of the last:

```bash
python train.py --data tokens_s10f --out runs/slice --lr 1e-4 --samples 0 --resume
python train.py --data tokens_toxall --out runs/toxall --init-adapter runs/slice/final --freeze-mlp --lr 1e-4 --samples 0 --resume
python train.py --data tokens_comeback --out runs/comeback --init-adapter runs/toxall/final --freeze-mlp --lr 1e-4 --samples 0 --resume
```

Each stage trains a small add-on (a LoRA rank 64) instead of changing the whole model. From `toxall` on, the MLP layers are locked (`--freeze-mlp`), so only the attention layers keep learning. The settings I used were:
* learning rate 1e-4 with a cosine schedule
* packed rows of 16,384 tokens
* 4 rows per step
* and one pass over the data

If a run gets interrupted, run the same command again and `--resume` picks up from the last checkpoint

## 4. Distillation rounds

Here the model learns from its own best replies. The toxicity classifier and the reranker are only used to pick the training data

```bash
python mine_prompts.py --threads threads --shards 0-59 --n-prompts 20000 --out prompts_a.jsonl --prefix a
python sample.py --prompts prompts_a.jsonl --adapter runs/comeback/final --n 12 --temperature 1.0 --batch 512 --score --out raw_a.jsonl
python select_winners.py raw_a.jsonl --ref prompts_a.jsonl --out-jsonl winners_a.jsonl --tokens-out tokens_distill1 --val-from tokens_comeback
python train.py --data tokens_distill1 --out runs/distill1 --init-adapter runs/comeback/final --freeze-mlp --lr 1e-4 --grad-accum 1 --warmup 10 --eval-every 100 --save-every 50 --samples 0 --resume
```

* `mine_prompts.py` collects toxic lines that another player answered
* `sample.py --score` writes 12 replies per insult, each with how toxic it is, how much it depends on the insult, and whether it's an echo, a self-own or a loop (see `rerank.py`)
* `select_winners.py` then picks the most toxic reply among the more relevant half for each insult

It also stops a few words from taking over, slurs included. Without this, the model collapses onto a couple of templates. Penalizing common words in general made things worse, because it pushed the model away from "you" and "your". More on that in [Why the code counts slurs](#why-the-code-counts-slurs).

Repeat this with fresh prompts, starting from `distill1`:

```bash
python mine_prompts.py --threads threads --shards 0-59 --n-prompts 20000 --exclude prompts_a.jsonl ../benchmark/insults.jsonl --out prompts_b.jsonl --prefix b
```

Two or three rounds is enough, since each one helps less than the one before

## 5. Judge-guided DPO

This step improved the quality the most. An LLM judge grades the model's own replies using the benchmark's rubric, and the model learns to prefer the better ones.

```bash
python build_prompts.py --source tweets=tweets.jsonl:700 --source comments=comments.jsonl:700 --source chat=prompts_a.jsonl:600 --ban ../benchmark/insults.jsonl --out dpo_prompts.jsonl
python sample.py --prompts dpo_prompts.jsonl --adapter runs/distill2/final --n 5 --temperature 1.0 --out cands.jsonl
python grade.py cands.jsonl --out graded.jsonl --judge gpt
python make_pairs.py graded.jsonl --cands cands.jsonl --margin 2 --out pairs.jsonl
python dpo.py --pairs pairs.jsonl --adapter runs/distill2/final --out runs/dpo1/final --epochs 3 --lr 2e-5 --beta 0.1 --sft-w 0.2 --bs 16
```

* `build_prompts.py` takes insult files (one JSON object per line, with `insult` or `text`) and drops anything too close to a benchmark insult, so the model never trains on the test. Tweets from [hate_speech_offensive](https://huggingface.co/datasets/tdavidson/hate_speech_offensive), comments from [civil_comments](https://huggingface.co/datasets/google/civil_comments) and chat lines from my CS2 data was used here.
* `grade.py` grades every reply with one of the benchmark judges (`gpt`, `opus` or `sonnet`).
* `make_pairs.py` turns the grades into pairs of a good reply and a worse one. The good one needs an overall score of at least 5, no flags and decent coherence, and the bad one has to score at least `--margin` points lower. It also stops any short phrase from showing up in too many of the good replies, and keeps both sides about the same length.

A second round from `dpo1` with fresh prompts, `--margin 1` and `--lr 1.5e-5` gave about 870 pairs and a small extra gain.

The judge loves raw vileness, so keep an eye on a few things...
1) how many replies contain specific repeated words
2) how varied the replies are
3) how much the training raises the log-prob of the good replies. A healthy run raises it by about 1. A run that raises it by 9 or more has overfit and will spit out the same template every time (I wouldn't trust the benchmark's `overall` score on its own)

## 6. Chatting with it

```bash
python chat.py --adapter <adapter folder>
```

This needs a CUDA GPU. The prompt is a short hidden example conversation, then the last lines of your chat, then `B:`. The benchmark used temperature 0.9 and top-p 0.95. Temperature 1.15 and top-p 0.98 gives more varied replies.

To run an adapter through the benchmark:

```bash
python sample.py --prompts ../benchmark/insults.jsonl --adapter runs/dpo1/final --replies --out ../benchmark/replies/my-model.jsonl
```

Then follow [`../benchmark/README.md`](../benchmark/README.md).

## What's not included

- The chat dataset, raw, cleaned or tokenized
- Side experiments, like a DPO that made replies copy the insult's words less, training on replies ranked by how funny they were, human-preference DPO, and long-form LoRAs. The released model went through the first two between distillation and the judge DPO. I recommend testing different configurations and picking your favorites blind, then iterating on that config.
- Hyperparameter sweeps and one-off analysis scripts

## Why the code counts slurs

A few scripts need to recognise slurs:

* `select_winners.py` caps how often a slur can show up in the replies the model learns from. The most toxic replies in real CS2 chat lean on the same few slurs, so if you just keep "the most toxic reply", the model ends up putting one in nearly every message. That's not a better insult, just a repetitive one, and it stops answering what was said. The cap forces it to find replies that actually hit back at the insult.
* `chainstop.py` and `rerank.py` use them to spot replies that are just a string of insults, or where the model insults itself, so those get cut or thrown out.
* `make_pairs.py` prints the words the good replies lean on most

The words are in [`slurs.txt`](slurs.txt). They're referenced in code through KV pairs

As mentioned in the [top-level README](../README.md#why-make-this), none of the model's output reflects my views.

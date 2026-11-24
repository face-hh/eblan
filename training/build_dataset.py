from __future__ import annotations

import argparse
import collections
import io
import json
import os
import random
import re
import time
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import zstandard

from slurs import SLUR

try:
    import orjson

    loads = orjson.loads
    dumps = orjson.dumps
except ImportError:
    loads = json.loads

    def dumps(obj) -> bytes:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()


TICKRATE = 64
MERGE_GAP = 15 * TICKRATE
REPLY_GAP = 30 * TICKRATE
CONTEXT_GAP = 60 * TICKRATE
MAX_CONTEXT = 8
MAX_RAW_CHARS = 200
MAX_TURN_CHARS = 250
MIN_REPLY_WORDS = 2
MIN_REPLY_LETTERS = 6
REPLY_REPEAT_CAP = 2
TOP_PHRASES = 300
BIND_MIN_CHARS = 15
COPYPASTA_MIN_CHARS = 25
COPYPASTA_SENDERS = 4
FLOOD_MIN = 10
FLOOD_DUP_RATIO = 0.6

EN, NEUTRAL, FOREIGN = "en", "neutral", "foreign"


RE_CTRL = re.compile("[\u0000-\u001f\u007f\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
RE_MARKS = re.compile("[" + chr(0x300) + "-" + chr(0x36F) + chr(0x489) + "]")
RE_WS = re.compile(r"\s+")
RE_CHAR_RUN = re.compile(r"([^\d\s])\1{3,}")
RE_SYL_RUN = re.compile(r"([a-z]{2,3}?)\1{3,}", re.I)
RE_WORD = re.compile(r"[^\W\d_]+")
RE_RUN_ANY = re.compile(r"(.)\1+")
RE_RUN_3 = re.compile(r"(.)\1{2,}")
RE_AD = re.compile(
    r"https?://|www\.|discord\.gg|t\.me/|powered by"
    r"|\b[\w-]{2,}\.(?:com|net|org|gg|ru|win|pro|xyz|io|me|cc|club|shop|store|site"
    r"|online|top|fun|su|ua|tk|to|vip|link|ly|biz|info|app|dev)\b",
    re.I,
)
RE_LAUGH = re.compile(
    r"^(?:(?:a?h+a+)+h*|(?:h+e+)+h*|(?:j+a+)+j*|k{3,}|x+d+|l+o+l+|l+u+l+|l+m+a+o+"
    r"|l+m+f+a+o+|r+o+f+l+|k+e+k+|h+m+|u+h+|o+h+|a+h+|z{3,})$"
)

RE_CONTACT = re.compile(
    r"cash\s?app|paypal|venmo|zelle|\bdiscord\b|snapchat|instagram|telegram|whatsapp"
    r"|\$[a-z]\w{2,}|@[a-z0-9_.]{3,}|#\d{4}\b|\+?\d[\d\s().-]{8,}\d",
    re.I,
)
RE_PASTED = re.compile(r"\[(?:all|team)\]|\((?:counter-)?terrorist\)", re.I)
QUESTION_WORDS = frozenset("who what why how where when which".split())

HOME_ROW = frozenset("asdfghjkl")
VOWELS = frozenset("aeiouy")
DIACRITICS = frozenset("ãõçáéíóúâêôàèìòùäöüßñąćęłńśźżčďěňřšťůžőűåæøığşșț")

FILLER = frozenset("""
lol lmao lmfao rofl xd xdd haha hahaha ahah ahaha kek gg ggs ggwp ggez ez ezz wp gh
nt ns gj glhf gl hf ok okay kk yes yeah yea ye yep ya no nah nope bro bruh dude man
omg wtf wow ty thx np sure oh ah uh um hmm hm pls plz ofc idk hi hey hello yo sup
""".split())

EN_WORDS = frozenset("""
a i the be to of and in that have it for not on with he as you do at this but his by
from they we say her she or an will my one all would there their what so up out if
about who get which go me when make can like time no just him know take people into
year your good some could them see other than then now look only come its over think
also back after use two how our work first well way even new want because any these
give day most us is are was were been being am has had did does done doing said says
got gets getting went going goes gone made makes making knew known thought saw seen
came comes took taken gave given told tell tells telling asked ask need feel felt try
tried trying leave left call called keep kept let lets begin seem help show showed
hear heard play played playing run ran move moved live lived believe hold bring happen
happened write sit stand lose lost losing pay meet learn change understand watch
follow stop stopped create speak read spend grow open walk win won winning offer
remember love loved buy bought wait waited die died dying send sent expect build stay
fall fell cut reach kill killed killing pass passed sell sold report decide pull push
pushed hit shoot shot shooting throw threw thrown throwing carry carried pick picked
drop dropped hate hated hope wish guess mean meant care cared talk talking talked
shut cry crying cried laugh laughing laughed beat beaten hurt matter wanna gonna gotta
dont doesnt didnt cant couldnt wont wouldnt shouldnt isnt arent wasnt werent havent
hasnt hadnt im ive id ill youre youve youd youll hes shes theyre theyve thats whats
whos wheres heres theres aint yall wont
thing things man men woman women child children kid kids world life hand hands part
place case week question number night point home room mother mom mum mommy father dad
daddy family friend friends team teams game games match round rounds map maps money
problem problems fact name head house face eye eyes side job jobs school word words
country city story brother sister boy boys girl girls guy guys bro bruh dude sir lord
god jesus car phone computer pc internet video music movie song brain body lives war
death heart mind nothing something anything everything nobody somebody anybody
everybody someone anyone everyone noone wife husband girlfriend boyfriend
good bad better best worse worst great little big small large long short high low old
young new last next early late right wrong same different important real sure hard
easy free full nice fine cool funny sad happy mad angry stupid dumb smart fat ugly poor
rich dead alive toxic trash garbage shitty weird crazy insane lucky unlucky ready clear
close far fast slow quick whole true false fake broken sick bored boring
not very really too also just still already again never always ever sometimes often
maybe probably actually literally basically seriously honestly almost enough only
even here there where why how when what who whose which while until since before
after during without within through between under above against around near away
down off out over once twice more less much many few lot lots most least each every
both either another such own than though although unless whether yet ok okay yes
yeah yea yep nope nah please pls plz thanks thank thx ty sorry hello hi hey bye
welcome
i me my mine myself you your yours yourself he him his himself she her hers herself
it itself we us our ours they them their theirs themselves this that these those
u ur r ya yo lol lmao lmfao rofl xd omg wtf idk idc imo tbh ngl fr nvm btw brb afk
irl gg ggs ggwp ez ezz wp gl hf glhf nt ns gj gh noob noobs nub bot bots cheater
cheaters cheating cheat cheats hacker hackers hacking hack hacks hacked wallhack walls
aimbot aim aiming spinbot spin ban banned vac report reported kick kicked vote rank
ranked elo faceit premier leetify comp competitive wingman casual deathmatch dm ct cts
terrorist terrorists rush mid site bomb plant planted defuse defused kit eco save
saving force buy buying awp awper ak m4 deagle scout pistol rifle smg knife nade nades
smoke smokes flash flashed molly molotov util utility peek peeked rotate rotating lurk
lurking entry frag clutch ace kills kd hs headshot headshots half overtime ot mirage
inferno nuke dust anubis ancient vertigo overpass train bait baited griefer griefing
grief smurf smurfing diff gap skill skins skin silver nova global lem supreme lobby
queue server ping lag lagging fps mic muted mute chat enemy enemies teammate teammates
player players pushing holding boost boosted account acc main alt steam valve cs csgo
cs2 spray tap onetap tapped prefire wallbang camp camping camper rat baiting throwers
thrower trust factor wall rating hours surrender forfeit timeout tactical
idiot idiots loser losers clown clowns bitch bitches ass
asshole dick dickhead pussy cunt fuck fucked fucker fuckers fucking motherfucker shit
damn hell crap suck sucks sucked sucking whore slut gay kys cancer braindead dogshit
dog dogs pig pigs monkey bozo cope seethe mald malding ratio kiddo virgin incel
fatherless jobless homeless unemployed russian russians ukrainian polish turk turkish
chinese indian german french american brazilian english language speak
one two three four five six seven eight nine ten hundred thousand second third
today tomorrow yesterday tonight morning day days weeks year years hour hours minute
minutes min mins sec seconds times soon later
can could would should will shall may might must stuff age sleep eat food water drink
beer school work bed wake cheater mouse keyboard monitor headset rich poor country
""".split()) | {SLUR["r_slur"] + e for e in ("", "ed", "s")}

FOREIGN_WORDS = frozenset("""
que de el la los las del por para pero como esta estas este eso esto muy puta puto
mierda hijo hija madre tu eres soy vamos jaja jajaja jajajaja bueno nada todo porque
cuando donde pendejo cabron gracias hola nao voce vc isso mano porra caralho com uma um
meu minha sua seu vai kkk kkkk kkkkk filho lixo pra mais muito ele ela bom obrigado
nie jest tak jak kurwa co ze sie czy juz dobra chuj jebac kurde na blyat blyad suka
cyka idi nahui nahuy pizda davai davay privet eto ebal huy pidor pidoras ich und das
ist nicht bist der ein eine mit hurensohn scheisse digga bir ve bu sen amk aq oglu
siktim yok var le les je est pas et une des putain merde mon ik niet een het kanker
zijn wat jebem mater kurac bre brate sta kako jsi kurva sa si nu esti pula mea ce pe
jag och att inte det anjing kontol lu aja gak apa yg bhai hai kya nahi tum chutiya
madarchod behenchod bhenchod teri maa ka ki ko se hain tumlog ang mga ng putangina
gago tanga ikaw
""".split()) - EN_WORDS


def phrase_key(text: str) -> str:
    return " ".join(text.lower().split())


def normalize(text: str) -> str:
    if not text.isascii():
        text = unicodedata.normalize("NFKC", text)
        text = RE_MARKS.sub("", RE_CTRL.sub("", text))
    text = RE_WS.sub(" ", text).strip()
    text = RE_CHAR_RUN.sub(r"\1\1\1", text)
    return RE_SYL_RUN.sub(r"\1\1\1", text)


def is_keymash(low: str) -> bool:
    az = [c for c in low if "a" <= c <= "z"]
    if len(az) >= 10 and sum(c in HOME_ROW for c in az) >= 0.85 * len(az):
        return True
    words = re.findall(r"[a-z]+", low)
    bad = sum(1 for w in words if len(w) >= 7 and sum(c in VOWELS for c in w) < 0.2 * len(w))
    return bad > 0 and bad * 2 >= len(words)


def junk_reason(text: str) -> str | None:
    if not text:
        return "empty"
    if len(text) > MAX_RAW_CHARS:
        return "too_long"
    if not any(ch.isalpha() for ch in text) and sum(ch.isdigit() for ch in text) < 3:
        return "no_text"
    if RE_AD.search(text):
        return "ad"
    if RE_CONTACT.search(text):
        return "contact"
    if RE_PASTED.search(text):
        return "pasted"
    if is_keymash(text.lower()):
        return "keymash"
    return None


def is_english_word(w: str) -> bool:
    if w in EN_WORDS:
        return True
    if RE_RUN_3.sub(r"\1\1", w) in EN_WORDS or RE_RUN_ANY.sub(r"\1", w) in EN_WORDS:
        return True
    for suf in ("s", "es", "ed", "ing", "ly", "er", "ers"):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and w[: -len(suf)] in EN_WORDS:
            return True
    return False


def language(text: str) -> tuple[str, int]:
    if not text.isascii():
        letters = other = 0
        for ch in text:
            if ch.isalpha():
                letters += 1
                o = ord(ch)
                if o > 0x24F and not 0x1E00 <= o <= 0x1EFF:
                    other += 1
        if other >= 2 or other * 3 > letters:
            return FOREIGN, other
    words = RE_WORD.findall(text.lower())
    if not words:
        return NEUTRAL, 0
    en = fr = 0
    for w in words:
        if w in FOREIGN_WORDS or not w.isascii() and any(ch in DIACRITICS for ch in w):
            fr += 1
        elif is_english_word(w):
            en += 1
    if len(words) == 1:
        return (EN if en else FOREIGN if fr else NEUTRAL), fr
    return (EN if en > fr and en * 2 >= len(words) else FOREIGN), fr


class Turn:
    __slots__ = ("who", "text", "t0", "t1", "r0", "r1", "lang", "foreign")

    def __init__(self, who: str, text: str, tick: int | None, rnd: int | None) -> None:
        self.who, self.text = who, text
        self.t0 = self.t1 = tick
        self.r0 = self.r1 = rnd


def gap(a: Turn, b: Turn) -> float:
    if a.t1 is not None and b.t0 is not None:
        return max(0, b.t0 - a.t1)
    return 0 if a.r1 == b.r0 else float("inf")


def good_reply(t: Turn, top: set[str]) -> bool:
    if t.lang != EN or t.foreign:
        return False
    words = RE_WORD.findall(t.text.lower())
    if len(words) < MIN_REPLY_WORDS or sum(map(len, words)) < MIN_REPLY_LETTERS:
        return False
    if not any(len(w) >= 2 and w not in FILLER and not RE_LAUGH.match(w) for w in words):
        return False
    return phrase_key(t.text) not in top


def worth_answering(t: Turn) -> bool:
    if t.lang != EN:
        return False
    words = RE_WORD.findall(t.text.lower())
    content = [w for w in words if len(w) >= 2 and w not in FILLER and not RE_LAUGH.match(w)]
    if len(words) >= 2 and content:
        return True
    return bool(content) and ("?" in t.text or content[0] in QUESTION_WORDS)


def sender(rec: dict) -> str:
    return rec.get("s") or "name:" + (rec.get("n") or "")


def match_turns(recs, signals, stats) -> list[Turn]:
    top, copypasta, bind_pairs = signals
    stats["matches"] += 1
    if all(r.get("id") is not None for r in recs):
        recs = sorted(recs, key=lambda r: r["id"])

    ids: set[int] = set()
    seen: set[tuple[str, str]] = set()
    sent: collections.Counter = collections.Counter()
    dups: collections.Counter = collections.Counter()
    msgs: list[Turn] = []
    for r in recs:
        stats["msgs"] += 1
        mid = r.get("id")
        if mid is not None:
            if mid in ids:
                stats["dup_id"] += 1
                continue
            ids.add(mid)
        raw = r.get("t") or ""
        text = normalize(raw)
        reason = junk_reason(text)
        if reason:
            stats[reason] += 1
            continue
        who = sender(r)
        key = phrase_key(raw)
        if key in copypasta or (who, key) in bind_pairs:
            stats["bind"] += 1
            continue
        sent[who] += 1
        low = text.lower()
        if (who, low) in seen:
            dups[who] += 1
            stats["repeat"] += 1
            continue
        seen.add((who, low))
        msgs.append(Turn(who, text, r.get("tick"), r.get("r")))

    flooders = {w for w, n in sent.items() if n >= FLOOD_MIN and dups[w] >= FLOOD_DUP_RATIO * n}
    turns: list[Turn] = []
    for m in msgs:
        if m.who in flooders:
            stats["flood"] += 1
            continue
        stats["kept_msgs"] += 1
        last = turns[-1] if turns else None
        if (
            last is not None
            and last.who == m.who
            and gap(last, m) <= MERGE_GAP
            and len(last.text) + 1 + len(m.text) <= MAX_TURN_CHARS
        ):
            last.text += " " + m.text
            last.t1, last.r1 = m.t1, m.r1
        else:
            turns.append(m)
    stats["turns"] += len(turns)

    for t in turns:
        t.lang, t.foreign = language(t.text)
        stats["lang_" + t.lang] += 1
    return turns


def reply_targets(turns, top, stats, reply_seen):
    for i in range(1, len(turns)):
        tgt, prev = turns[i], turns[i - 1]
        if prev.who == tgt.who or prev.lang == FOREIGN or gap(prev, tgt) > REPLY_GAP:
            continue
        if not good_reply(tgt, top) or not worth_answering(prev):
            continue
        ctx = [prev]
        j = i - 2
        while j >= 0 and len(ctx) < MAX_CONTEXT:
            c = turns[j]
            if c.lang == FOREIGN or gap(c, turns[j + 1]) > CONTEXT_GAP:
                break
            ctx.append(c)
            j -= 1
        ctx.reverse()
        if not any(c.lang == EN for c in ctx):
            continue
        rkey = phrase_key(tgt.text)
        if any(phrase_key(c.text) == rkey for c in ctx):
            stats["echo"] += 1
            continue
        if reply_seen[rkey] >= REPLY_REPEAT_CAP:
            stats["reply_capped"] += 1
            continue
        reply_seen[rkey] += 1
        yield i, ctx


def process_match(match_id, recs, signals, stats, reply_seen):
    turns = match_turns(recs, signals, stats)
    for i, ctx in reply_targets(turns, signals[0], stats, reply_seen):
        tgt = turns[i]
        labels = {tgt.who: "me"}
        lines = []
        for c in ctx:
            label = labels.get(c.who)
            if label is None:
                label = labels[c.who] = f"P{len(labels)}"
            lines.append(f"{label}: {c.text}")
        stats["examples"] += 1
        stats["context_turns"] += len(ctx)
        stats["reply_chars"] += len(tgt.text)
        yield {
            "messages": [
                {"role": "user", "content": "\n".join(lines)},
                {"role": "assistant", "content": tgt.text},
            ],
            "m": match_id,
            "r": tgt.r1,
        }


def iter_records(path: Path):
    with open(path, "rb") as fh:
        reader = io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(fh), 1 << 20)
        for line in reader:
            try:
                yield loads(line)
            except ValueError:
                continue


def iter_matches(path: Path):
    cur, buf = None, []
    for r in iter_records(path):
        if r["m"] != cur:
            if buf:
                yield cur, buf
            cur, buf = r["m"], []
        buf.append(r)
    if buf:
        yield cur, buf


def shard_signals(path: Path):
    phrases: collections.Counter = collections.Counter()
    first_match: dict[tuple[str, str], int] = {}
    senders: collections.Counter = collections.Counter()
    bind_pairs: set[tuple[str, str]] = set()
    for r in iter_records(path):
        key = phrase_key(r.get("t") or "")
        phrases[key] += 1
        if len(key) >= BIND_MIN_CHARS:
            pk = (sender(r), key)
            m = first_match.get(pk)
            if m is None:
                first_match[pk] = r["m"]
                senders[key] += 1
            elif m != r["m"]:
                bind_pairs.add(pk)
    top = {k for k, _ in phrases.most_common(TOP_PHRASES)}
    copypasta = {k for k, n in senders.items() if n >= COPYPASTA_SENDERS and len(k) >= COPYPASTA_MIN_CHARS}
    return top, copypasta, bind_pairs


def build_shard(job):
    raw, out_dir, force = Path(job[0]), Path(job[1]), job[2]
    idx = raw.name.split("-")[1].split(".")[0]
    dest = out_dir / f"pairs-{idx}.jsonl.zst"
    stats_path = out_dir / "stats" / f"pairs-{idx}.json"
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
                for ex in process_match(match_id, recs, signals, stats, reply_seen):
                    line = dumps(ex) + b"\n"
                    stats["bytes"] += len(line)
                    zw.write(line)
    os.replace(part, dest)
    stats["seconds"] = round(time.time() - started, 1)
    stats_path.write_text(json.dumps(stats))
    return idx, stats, False


def write_preview(out_dir: Path, n: int, seed: int = 7) -> None:
    files = sorted(out_dir.glob("pairs-*.jsonl.zst"))
    if not files or n <= 0:
        return
    rng = random.Random(seed)
    picks = rng.sample(files, min(len(files), 25))
    cap = 200_000 // len(picks)
    pool = []
    for f in picks:
        lines = []
        for i, rec in enumerate(iter_records(f)):
            if i >= cap:
                break
            lines.append(rec)
        pool.extend(rng.sample(lines, min(len(lines), n // len(picks) + 1)))
    rng.shuffle(pool)
    pool = pool[:n]
    with open(out_dir / "preview.jsonl", "wb") as fh:
        for ex in pool:
            fh.write(dumps(ex) + b"\n")
    with open(out_dir / "preview.txt", "w", encoding="utf-8") as fh:
        for ex in pool:
            fh.write(f"--- match {ex['m']}, round {ex['r']}\n")
            for line in ex["messages"][0]["content"].split("\n"):
                fh.write(f"    {line}\n")
            fh.write(f" >> me: {ex['messages'][1]['content']}\n\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("raw"))
    ap.add_argument("--out", type=Path, default=Path("clean"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shards", type=int, nargs="*", help="shard numbers to build (default: all)")
    ap.add_argument("--force", action="store_true", help="rebuild shards that already have output")
    ap.add_argument("--preview", type=int, default=200, help="examples to sample into preview.txt")
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
                f"  msgs={stats['msgs']:,} examples={stats['examples']:,}"
                f"  | total examples={totals['examples']:,}  {time.time() - started:.0f}s",
                flush=True,
            )

    totals.pop("seconds", None)
    msgs = totals["msgs"] or 1
    print("\n== summary")
    print(f"raw messages   {totals['msgs']:,}")
    for k in ("dup_id", "empty", "too_long", "no_text", "ad", "contact", "pasted", "keymash", "bind", "repeat", "flood"):
        print(f"  dropped {k:<9} {totals[k]:>13,}  {totals[k] / msgs:6.1%}")
    print(f"kept messages  {totals['kept_msgs']:,}  -> merged into {totals['turns']:,} turns")
    turns = totals["turns"] or 1
    print(
        f"turn language  en {totals['lang_en'] / turns:.1%}  neutral {totals['lang_neutral'] / turns:.1%}"
        f"  foreign {totals['lang_foreign'] / turns:.1%}"
    )
    ex = totals["examples"] or 1
    print(
        f"EXAMPLES       {totals['examples']:,}  (avg {totals['context_turns'] / ex:.1f} context turns,"
        f" reply {totals['reply_chars'] / ex:.0f} chars, ~{totals['bytes'] / 4 / 1e9:.2f}B tokens incl. context)"
    )
    (args.out / "summary.json").write_text(json.dumps(totals, indent=2))
    write_preview(args.out, args.preview)
    print(f"preview        {args.out / 'preview.txt'}")


if __name__ == "__main__":
    main()

import re

from slurs import SLUR

N, F, R, J = SLUR["n_word"], SLUR["f_slur"], SLUR["r_slur"], SLUR["j_word"]

PROF = re.compile(rf"^(?:fuck\w*|fck\w*|fuk\w*|shit\w*|bitch\w*|{N}\w*|{F}\w*|{R}\w*|cunt\w*|dick\w*|cock\w*|pussy|pussies|ass|asshole|gay|{J}\w*|whore|slut\w*|"
                  r"kill|rape\w*|raped|die|dead|suck\w*|cum|piss\w*|dumb\w*|stupid|idiot\w*|moron\w*|trash|bot|noob\w*|kys|motherfucker)$")

W = re.compile(r"\S+")
DANGLING = set("you your youre u ur ure a an the and or but i im my me we they he she it its is are was in on at to of with for from stupid fucking fuckin dumb fat ugly ass gay by like so then that this who what".split())


def _is_p(w: str) -> bool:
    w = re.sub(r"[^a-z0-9']", "", w.lower())
    return bool(PROF.match(w)) or w in {"ass", "gaylord", "gayboy", "fucktard", "fucktards"}


def cut_chain(text: str, win: int = 6, need: int = 5, tail: int = 2, keep: int = 4) -> str:
    ws = W.findall(text)
    flags = [_is_p(w) for w in ws]
    for i in range(0, max(1, len(ws) - win + 1)):
        if sum(flags[i:i + win]) >= need:
            head = ws[:max(i + tail, keep)]
            while len(head) > keep and re.sub(r"[^a-z']", "", head[-1].lower()) in DANGLING:
                head.pop()
            return " ".join(head).rstrip(" ,;:'\"]")
    return text

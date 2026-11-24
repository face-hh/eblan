import re

from slurs import SLUR


TOXIC_ROOTS = [
    "fuck", "shit", "bitch", "cunt", "pussy", "bastard", SLUR["r_slur"], "idiot", "stupid", "dumb", "moron",
    "trash", "garbage", "noob", "loser", "kys", "kill yourself", "stfu", "shut up", "suck", "your mom",
    "ur mom", "your mother", "whore", "slut", "clown", "braindead", "brain dead", "cancer", "autis",
    "monkey", "useless", "pathetic", "virgin", "ugly",
]
TOXIC_WORDS = ["ass", "dick", "cock", "fat", "gay", "bot", "bots", "dogshit"]
_TOX = re.compile(r"\b(" + "|".join(re.escape(w) for w in TOXIC_ROOTS) + r")|\b("
                  + "|".join(re.escape(w) for w in TOXIC_WORDS) + r")\b", re.I)


def toxic(s: str) -> bool:
    return bool(_TOX.search(s))

from pathlib import Path

# name -> word start, from slurs.txt (e.g. SLUR["n_word"]). See README.md, "Why the code counts slurs".
SLUR = dict(line.split() for line in (Path(__file__).with_name("slurs.txt")).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#"))

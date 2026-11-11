"""Automation helpers for sim reels.

  python3 autosim.py run REEL SEED out.json [--record 1,5,9] [--kw gens=80]
      Runs a sim with a fresh seed. Sims that only record some generations (shortcut, drone)
      record nothing unless --record is given: run once to read stats, then again with --record.
  python3 autosim.py stats data.json
      Prints every generation's scalar numbers (the only numbers a spec may use).
  python3 autosim.py check data.json spec.json
      Fact-check: every number in say/cap/stat/post caption must exist in the run's stats,
      every beat gen must be recorded. Exits 1 on any problem.
"""
import json, re, sys, time
import numpy as np
import sims, sims2, sims3

SIMS = {**sims.SIMS, **sims2.SIMS, **sims3.SIMS}
PARTIAL = {"shortcut", "drone"}  # sims with a `record` argument


class NE(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def run(reel, seed, out, record=None, kw=None):
    kw = dict(kw or {}); kw["seed"] = seed
    if reel in PARTIAL: kw["record"] = tuple(record or ())
    t = time.time(); r = SIMS[reel](**kw); r["stats"]["reel"] = reel; r["stats"]["seed"] = seed
    json.dump(r, open(out, "w"), cls=NE)
    print(f"{reel} seed {seed}: {len(r['gens'])} gens in {time.time() - t:.0f}s -> {out}")
    stats(out, r)


def scalars(d):
    return {k: v for k, v in d.items() if isinstance(v, (int, float, bool, str)) and not isinstance(v, bool) or isinstance(v, bool)}


HEAVY = {"runs", "ys", "xs", "paths", "shots", "games", "frames", "guesses", "rocks", "fit", "fr", "usedat", "prog", "used_i", "right", "rings", "passed", "sc", "scores", "walls", "block", "hits", "sw", "ok", "crash", "hoop"}


def stats(path, r=None):
    r = r or json.load(open(path)); st = {k: v for k, v in r["stats"].items() if k not in ("pipes", "cacti")}
    print("STATS", json.dumps(st)[:3000])
    for d in r["gens"]:
        row = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in d.items() if k not in HEAVY and not isinstance(v, (list, dict))}
        rec = any(d.get(k) is not None for k in ("runs", "ys", "xs", "paths", "shots", "games", "frames", "guesses"))
        if "rings" in d and isinstance(d["rings"], list): row["best_rings"] = max(d["rings"])
        print(("REC " if rec else "    ") + json.dumps(row))


def numbers_in(text):
    return [float(x.replace(",", "")) for x in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", text)]


WORDNUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
           "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
           "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100}


def spoken_numbers(text):
    out = []; cur = None
    for w in re.findall(r"[a-z]+|[.,:;!?]", text.lower()):
        if w in WORDNUM:
            v = WORDNUM[w]
            if w == "hundred": cur = (cur or 1) * 100
            elif cur is not None and cur % 100 == 0 and cur >= 100: cur += v
            elif cur is not None and cur >= 20 and cur % 10 == 0 and v < 10: cur += v
            else:
                if cur is not None: out.append(cur)
                cur = v
        elif w == "and" and cur is not None and cur >= 100: continue
        else:
            if cur is not None: out.append(cur); cur = None
    if cur is not None: out.append(cur)
    return [n for n in out if n not in (1, 2, 100)]


def check(data, specp):
    r = json.load(open(data)); spec = json.load(open(specp)); G = {d["g"]: d for d in r["gens"]}; bad = []
    pool = set()
    def walk(x):
        if isinstance(x, bool): return
        if isinstance(x, (int, float)): pool.add(round(float(x), 1)); pool.add(float(int(x))); pool.add(float(round(x)))
        elif isinstance(x, (list, tuple)):
            if len(x) < 400: [walk(y) for y in x]
        elif isinstance(x, dict): [walk(v) for k, v in x.items() if k not in HEAVY or k in ("rings",)]
    walk(r["stats"])
    for d in r["gens"]:
        walk({k: v for k, v in d.items() if k not in HEAVY}); pool.add(float(d["g"]))
        if isinstance(d.get("rings"), list): pool.add(float(max(d["rings"])))
    for k in ("ys", "xs", "runs", "paths", "guesses"):
        n = next((len(d[k]) for d in r["gens"] if isinstance(d.get(k), list)), None)
        if n: pool.add(float(n))
    allowed = pool | {float(x) for x in spec.get("constants", [])}
    for i, sb in enumerate(spec["beats"]):
        g = sb["gen"]
        if g not in G: bad.append(f"beat {i}: gen {g} not in run"); continue
        if not any(G[g].get(k) is not None for k in ("runs", "ys", "xs", "paths", "shots", "games", "frames", "guesses")): bad.append(f"beat {i}: gen {g} not recorded")
        for field in ("say", "cap", "stat"):
            txt = sb.get(field, "")
            if "—" in txt: bad.append(f"beat {i} {field}: em dash")
            for n in numbers_in(txt):
                if n not in allowed: bad.append(f"beat {i} {field}: number {n:g} not found in the run's stats (add it to spec.constants only if it is a fixed setting of the sim)")
            if field == "say":
                for n in spoken_numbers(txt):
                    if float(n) not in allowed: bad.append(f"beat {i} say: spoken number {n} not found in stats")
    for field in ("caption", "title", "pinned"):
        txt = spec["post"].get(field, "")
        if "—" in txt: bad.append(f"post {field}: em dash")
        if field == "caption":
            for n in numbers_in(txt):
                if n not in allowed: bad.append(f"post caption: number {n:g} not found in stats")
    print("\n".join(bad) if bad else "CHECK OK"); sys.exit(1 if bad else 0)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "run":
        rec = None; kw = {}
        if "--record" in a: rec = [int(x) for x in a[a.index("--record") + 1].split(",") if x]
        if "--kw" in a: kw = {k: int(v) for k, v in (p.split("=") for p in a[a.index("--kw") + 1].split(","))}
        run(a[1], int(a[2]), a[3], rec, kw)
    elif a[0] == "stats": stats(a[1])
    elif a[0] == "check": check(a[1], a[2])

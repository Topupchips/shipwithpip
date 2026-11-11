"""Spec-driven story builder for the automated pipeline.

A spec is JSON written from the real stats of one sim run (see AUTOMATION.md):
{
  "reel": "drone",
  "beats": [ {"gen": 1, "head": ["100 DRONES.", "6 RINGS."], "stat": "BEST 2 RINGS", "stat_col": "INK",
              "say": "spoken line", "cap": "on-screen caption",
              "sfx": {"text": "FINISHED!", "col": "LIME", "burst": "GREEN", "t": 3.0}} , ... ],
  "end": ["100 drones.", "0 flight controllers.", "Just evolution."],
  "post": {"title": ..., "caption": ..., "tags": [...], "pinned": ..., "notes": ...}
}
The visuals for each beat come from the recorded generation in the sim data, never from the spec.
"""

LABEL = dict(snake="GAME", hunt="ROUND", moon="LAUNCH")


def _keep_hits(d, ph):
    hitTimes = []
    for run in d["runs"]:
        ht = []
        for s in range(1, len(run)):
            for k in range(1 + ph):
                if run[s - 1][2 + k * 2] >= 200 and run[s][2 + k * 2] < run[s - 1][2 + k * 2]: ht.append(s); break
        hitTimes.append(ht)
    return hitTimes


DATA = dict(
    hoops=lambda d, r: dict(hoop=d["hoop"], shots=d["shots"], sw=d["sw"], phase=d["phase"]),
    lander=lambda d, r: dict(runs=[x[:460] for x in d["runs"]], ok=d["ok"], crash=d["crash"]),
    snake=lambda d, r: dict(games=d["games"], scores=d["scores"]),
    dino=lambda d, r: dict(ys=[y[:420] for y in d["ys"]], sc=d["sc"]),
    hunt=lambda d, r: dict(frames=d["frames"], prey_learning=d["prey_learning"]),
    maze=lambda d, r: dict(paths=d["paths"], ok=d["ok"], phase=d["phase"]),
    flappy=lambda d, r: dict(ys=[y[:480] for y in d["ys"]], full=480),
    moon=lambda d, r: dict(shots=d["shots"], ok=d["ok"], crash=d["crash"], phase=d["phase"]),
    dodge=lambda d, r: dict(xs=d["xs"], rocks=d["rocks"], phase=d["phase"], full=480),
    pole=lambda d, r: dict(runs=d["runs"], phase=d["phase"], full=450),
    walk=lambda d, r: dict(runs=d["runs"], fr=d["fr"]),
    shortcut=lambda d, r: dict(runs=d["runs"], usedat=d["usedat"], prog=d["prog"]),
    drone=lambda d, r: dict(runs=d["runs"], rings=d["rings"], passed=d["passed"], phase=d["phase"]),
    password=lambda d, r: dict(guesses=d["guesses"], right=d["right"]),
    keepups=lambda d, r: dict(runs=d["runs"], nballs=1 + d["phase"], full=900, hitTimes=_keep_hits(d, d["phase"])),
)


def _maze_common(r):
    G = r["gens"]; w0 = next(g["walls"] for g in G if g.get("walls")); w1 = next((g["walls"] for g in G if g.get("walls") and g["phase"] == 1), w0)
    return dict(walls0=w0, walls1=w1, block=next((g["block"] for g in G if g.get("block")), None))


COMMON = dict(
    hoops=lambda r: dict(HOOP=[820, 420], RIM=40, SHOOTER=[120, 820]),
    lander=lambda r: dict(PADX=[400, 600], PADY=940),
    dino=lambda r: dict(cacti=r["stats"]["cacti"], GY=760),
    maze=_maze_common,
    flappy=lambda r: dict(pipes=r["stats"]["pipes"]),
    password=lambda r: dict(target=r["stats"]["target"]),
)


def spec_story(r, spec, pick, beat):
    reel = spec["reel"]; G = r["gens"]; B = []
    for sb in spec["beats"]:
        g = sb["gen"]; d = pick(G, g)
        missing = [k for k in ("runs", "ys", "xs", "paths", "shots", "games", "frames", "guesses") if k in d and d[k] is None]
        if missing: raise SystemExit(f"gen {g} was not recorded ({missing}); rerun the sim with this gen in record")
        b = beat("gen", head=sb["head"], label=sb.get("label", f"{LABEL.get(reel, 'GEN')} {g}"), stat=dict(text=sb["stat"], col=sb.get("stat_col", "INK")),
                 say=sb["say"], cap=sb["cap"], data=DATA[reel](d, r))
        if sb.get("sfx"): b["sfx"] = dict(dict(y=1000, size=110), **sb["sfx"])
        B.append(b)
    B.append(beat("end", lines=spec["end"], say=spec.get("end_say", "Follow for more AI from scratch.")))
    return B, COMMON.get(reel, lambda r: {})(r), spec["post"]

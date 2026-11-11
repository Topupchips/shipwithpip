"""Many-AIs-one-goal reels on the neon canvas engine. usage: python goal2.py REEL data.json out.mp4 [--stills t1,t2]"""
import argparse, functools, http.server, json, os, re, subprocess, sys, threading
import numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import voice  # noqa
from make_reel import music_bed  # noqa
from simfx import SR, tone  # noqa
from learnfx import write_post  # noqa
FPS = 30
MODEL = os.environ.get("KOKORO_DIR", "/tmp/claude-0/-home-claude/a75793b8-e870-5638-9e92-ffc2e9c0e649/scratchpad/tts/kokoro-multi-lang-v1_0")
VOICE, SPEED = "am_adam", 1.04
NUM = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}


def words(n):
    n = int(n)
    if n in NUM: return NUM[n]
    if n < 100: return NUM[n // 10 * 10] + " " + NUM[n % 10]
    if n == 100: return "one hundred"
    return str(n)


def pick(gens, g): return next(x for x in gens if x["g"] == g)


def beat(kind, **k): k["kind"] = kind; return k


# ---------------------------------------------------------------- story builders -> (beats, common, post)
def story_hoops(r):
    G = r["gens"]; mv = r["stats"]["moved_at"]; fin = G[-1]["g"]
    p1 = [g for g in G if g["phase"] == 1]; first = next(g["g"] for g in p1 if g["swish"] > 0); mid = next(g["g"] for g in p1 if g["swish"] >= 50)
    show = sorted(set([1, 2, 3, mv, mv + 1, mv + 3, first, mid, fin]))
    B = []
    for g in show:
        d = pick(G, g); n = d["swish"]; ph = d["phase"]
        head = ["100 AIs.", "1 HOOP."] if ph == 0 else (["I MOVED", "THE HOOP."] if g < fin else ["THEY LEARNED", "IT TWICE."])
        if g == 1: say, cap = f"A hundred AIs. One basketball. None of them has ever taken a shot. Generation one: {words(n)} go in.", f"100 AIs. 1 basketball. Never shot before. Gen 1: {n} go in."
        elif g == 2: say, cap = f"The closest misses breed. Their kids get a tiny random change. Generation two: {words(n)}.", f"The closest misses breed, with a tiny random change. Gen 2: {n}."
        elif g == 3: say, cap = f"Generation three: {words(n)}.", f"Gen 3: {n} in."
        elif g == mv: say, cap = f"Generation {words(g)}: {words(n)} out of a hundred. Solved. So I moved the hoop.", f"Gen {g}: {n}/100. Solved. So I moved the hoop."
        elif g == mv + 1: say, cap = "Zero. Every single one misses. The old shot is useless now.", "Zero. Every one misses. The old shot is useless now."
        elif g == mv + 3: say, cap = "Still zero. They keep throwing the old shot.", "Still zero. They keep throwing the old shot."
        elif g == first: say, cap = f"Generation {words(g)}: {words(n)} go in. The first ones through.", f"Gen {g}: {n} in. The first ones through."
        elif g == mid: say, cap = f"Generation {words(g)}: {words(n)}.", f"Gen {g}: {n}."
        else: say, cap = f"Generation {words(g)}: {words(n)} out of a hundred. They learned it twice. No instructions. Just trial, error, and breeding.", f"Gen {g}: {n}/100. Learned it twice. No instructions, just trial, error and breeding."
        b = beat("gen", head=head, label=f"GEN {g}", stat=dict(text=f"{n} / 100", col="GREEN" if n >= 90 else ("RED" if n == 0 and ph == 1 else "INK")), say=say, cap=cap, data=dict(hoop=d["hoop"], shots=d["shots"], sw=d["sw"], phase=ph))
        if g == mv + 1: b["sfx"] = dict(text="HOOP MOVED!", col="YEL", t=0.3, y=1000, size=110)
        if g == fin: b["sfx"] = dict(text="TWICE!", col="LIME", burst="GREEN", t=2.2, y=1000, size=130)
        B.append(b)
    B.append(beat("end", lines=["100 AIs.", "0 instructions.", "Just evolution."], say="Follow for more AI from scratch."))
    common = dict(HOOP=[820, 420], RIM=40, SHOOTER=[120, 820])
    post = dict(title="GOAL  100 AIs learn to shoot hoops (then I moved the hoop)",
                caption=f"100 AIs, each with two numbers for a brain: angle and power. None has ever taken a shot. The closest misses breed with a tiny random change. Gen 1: {pick(G,1)['swish']} in. Gen {mv}: {pick(G,mv)['swish']}/100. Then I moved the hoop: zero for {first-mv-1} generations, then {pick(G,first)['swish']}, then {pick(G,fin)['swish']}/100 by gen {fin}. A genetic algorithm, no instructions.",
                tags=["ai", "geneticalgorithm", "machinelearning", "simulation", "coding", "basketball", "shipwithpip"],
                pinned="Genetic algorithm: population 100, genes = (angle, power), keep the top 10 by distance-to-hoop (+bonus for a swish), breed with uniform crossover and Gaussian mutation that shrinks each generation. Moving the hoop restarts the search from the old population.",
                notes=f"Seed {r['stats']['seed']}. Projectile sim: gravity 0.5 px/step^2, release at (120,820), hoop at (820,420) rim radius 40, backboard at x=870, second hoop at (480,680) with no backboard. Swish = crossing rim height downward within 28 px of the hoop centre. Measured swishes per generation: {[(g['g'], g['swish']) for g in G]}. Visuals are a canvas render of the real trajectories.")
    return B, common, post


def story_lander(r):
    G = r["gens"]; fin = G[-1]["g"]; n_fin = G[-1]["landed"]; N = len(G[-1]["runs"])
    dip = max([g["g"] for g in G if g["landed"] == 0 and g["g"] > 4] or [0]); a = next(g["g"] for g in G if g["landed"] >= 9 and g["g"] > dip); bb = next(g["g"] for g in G if g["landed"] >= 30)
    show = sorted(set([1, 4, dip, a, bb, fin]))
    B = []
    for g in show:
        d = pick(G, g); n = d["landed"]
        head = ["60 ROCKETS.", "1 PAD."] if g < fin else ["NOBODY WROTE", "A LANDING RULE."]
        if g == 1: say, cap = f"Sixty rockets. Each has a tiny neural network for a brain, with random weights. Generation one: {words(n)} land.", f"60 rockets, each with a tiny random neural net. Gen 1: {n} land."
        elif g == 4: say, cap = f"The softest crashes, closest to the pad, breed. Generation four: {words(n)} land. Then it gets worse.", f"Softest crashes, closest to the pad, breed. Gen 4: {n} land. Then it gets worse."
        elif g == dip: say, cap = f"Generation {words(g)}: zero. They learned not to crash. So they hover. Nobody lands.", f"Gen {g}: 0. They learned not to crash. So they hover. Nobody lands."
        elif g == a: say, cap = f"Generation {words(g)}: {words(n)}. A few of them finally come down.", f"Gen {g}: {n}. A few finally come down."
        elif g == bb: say, cap = f"Generation {words(g)}: {words(n)} out of sixty. One generation later, most of them land.", f"Gen {g}: {n}/60. One generation later, most of them land."
        else: say, cap = f"Generation {words(g)}: {words(n)} out of sixty land softly, on the pad. Nobody wrote a landing rule. It evolved.", f"Gen {g}: {n}/60 land softly on the pad. Nobody wrote a landing rule. It evolved."
        b = beat("gen", head=head, label=f"GEN {g}", stat=dict(text=f"{n} / {N}", col="GREEN" if n >= N * 0.6 else ("RED" if n == 0 and g > 4 else "INK")), say=say, cap=cap, data=dict(runs=[x[:460] for x in d["runs"]], ok=d["ok"], crash=d["crash"]))
        if g == fin: b["sfx"] = dict(text="LANDED", col="LIME", burst="GREEN", t=3.0, y=1000, size=130)
        B.append(b)
    B.append(beat("end", lines=["60 rockets.", "0 rules.", "Just evolution."], say="Follow for more AI from scratch."))
    common = dict(PADX=[400, 600], PADY=940)
    post = dict(title="GOAL  60 AIs learn to land a rocket",
                caption=f"60 rockets, each controlled by a tiny neural network (5 inputs, 6 hidden, 2 outputs: main thrust and side thrust) with random weights. The softest, closest crashes breed. Gen 1: {pick(G,1)['landed']} land. Gen {fin}: {n_fin}/60 land softly on the pad. Nobody wrote a landing rule.",
                tags=["ai", "neuralnetwork", "geneticalgorithm", "rocket", "simulation", "coding", "shipwithpip"],
                pinned="Neuroevolution: fixed-topology MLP, weights evolved by a genetic algorithm (elite 6 of 60, 30% crossover, Gaussian mutation). Fitness rewards low touchdown speed, pad distance and a time bonus for landing. A landing counts as |vy|<2, |vx|<1.5, on the pad.",
                notes=f"Seed {r['stats']['seed']}. Gravity 0.12 px/step^2, max main thrust 0.32, side thrust 0.07, 600-step limit, fitness averaged over 3 start positions; the displayed run is start (500,80,vx=1.5). Landed counts per generation (displayed start): {[(g['g'], g['landed']) for g in G]}.")
    return B, common, post


def story_snake(r):
    G = r["gens"]; st = r["stats"]; B = []
    for d in G:
        e = d["g"]; avg = st["avg"][str(e)] if str(e) in st["avg"] else st["avg"][e]; best = st["best"][str(e)] if str(e) in st["best"] else st["best"][e]; a = round(avg, 1)
        head = ["16 AIs.", "16 SNAKES."] if e < 1500 else ["1,500 GAMES", "LATER."]
        if e == 1: say, cap = "Sixteen AIs. Each gets its own game of Snake. They know nothing. Game one: all sixteen drive straight into the wall.", "16 AIs, 16 games of Snake. They know nothing. Game 1: all 16 drive straight into the wall."
        elif e == 30: say, cap = f"After thirty games they've learned one thing: walls hurt. Average score {a}.", f"After 30 games they've learned one thing: walls hurt. Average {a}."
        elif e == 150: say, cap = f"Game one fifty. They chase the food now. Average {a}. Best {words(best)}.", f"Game 150. They chase the food now. Average {a}, best {best}."
        elif e == 500: say, cap = f"Game five hundred. Average {a}. Best {words(best)}.", f"Game 500. Average {a}. Best {best}."
        else: say, cap = f"Game fifteen hundred. Average {a}. The best one scores {words(best)}. This is Q learning: a table of what every situation is worth, updated after every single move.", f"Game 1500. Average {a}, best {best}. Q-learning: a table of what every situation is worth, updated after every move."
        b = beat("gen", head=head, label=f"GAME {e}", stat=dict(text=f"AVG {a}", col="GREEN" if a >= 15 else "INK"), say=say, cap=cap, data=dict(games=d["games"], scores=d["scores"]))
        if e == 1500: b["sfx"] = dict(text=f"BEST {best}", col="LIME", burst="GREEN", t=3.5, y=1000, size=120)
        B.append(b)
    B.append(beat("end", lines=["16 snakes.", "1,500 games.", "Q-learning."], say="Follow for more AI from scratch."))
    post = dict(title="GOAL  16 AIs learn Snake at the same time",
                caption=f"16 AIs each play their own Snake. Tabular Q-learning: the state is 11 yes/no facts (danger ahead/left/right, direction, where the food is), 3 actions, a table updated after every move. Game 1: everyone drives into the wall. Game 1500: average {round(st['avg'].get('1500', st['avg'].get(1500)),1)}, best {st['best'].get('1500', st['best'].get(1500))}.",
                tags=["ai", "reinforcementlearning", "qlearning", "snake", "machinelearning", "coding", "shipwithpip"],
                pinned="Q-learning (Watkins 1989): Q[s,a] += alpha * (reward + gamma * max Q[s',:] - Q[s,a]). Here alpha 0.1, gamma 0.9, epsilon-greedy exploration decaying from 0.5, reward +10 food, -10 death. 12x12 grid.",
                notes=f"Seed {st['seed']}. 16 independent agents trained 1500 episodes each; the shown games are greedy (no exploration) evaluations at episodes {[g['g'] for g in G]}. Average/best scores: {st['avg']} / {st['best']}.")
    return B, {}, post


def story_dino(r):
    G = r["gens"]; fin = G[-1]["g"]; N = len(G[0]["ys"]); fd = G[-1]["done"]
    first = next(g["g"] for g in G if g["done"] > 0); big = next(g["g"] for g in G if g["done"] >= 30)
    show = sorted(set([1, 2, first, max(first + 1, 4), big, big + 1, fin]))
    B = []
    for g in show:
        d = pick(G, g); n = d["done"]; best = int(d["best"] / 10)
        head = ["100 AIs.", "1 DINO GAME."] if g < fin else ["4 NUMBERS.", "THAT'S THE BRAIN."]
        if g == 1: say, cap = f"A hundred AIs play the Chrome dino game. Each brain is four numbers. Generation one: everyone dies. Best score, {best}.", f"100 AIs play the Chrome dino game. Each brain is 4 numbers. Gen 1: everyone dies. Best {best}."
        elif g == 2: say, cap = (f"The ones that ran furthest breed. Generation two: {words(n)} makes it to the end." if n == 1 else f"The ones that ran furthest breed. Generation two: {words(n)} make it to the end." if n else "The ones that ran furthest breed. Generation two: still nobody survives."), f"The furthest runners breed. Gen 2: {n} finish."
        elif g == first and first != 2: say, cap = f"Generation {words(g)}: the first one survives the whole run.", f"Gen {g}: the first one survives the whole run."
        elif g == big: say, cap = f"Generation {words(g)}: {words(n)} out of a hundred.", f"Gen {g}: {n}/100."
        elif g == big + 1: say, cap = f"Generation {words(g)}: {words(n)}.", f"Gen {g}: {n}."
        elif g == fin: say, cap = f"Generation {words(g)}: {words(n)} out of a hundred finish the whole run. Four numbers. That's the entire brain.", f"Gen {g}: {n}/100 finish the whole run. 4 numbers. That's the entire brain."
        else: say, cap = f"Generation {words(g)}: {words(n)} survive. They're jumping now.", f"Gen {g}: {n} survive. They're jumping now."
        b = beat("gen", head=head, label=f"GEN {g}", stat=dict(text=f"{n} / {N}", col="GREEN" if n >= 50 else "INK"), say=say, cap=cap, data=dict(ys=[y[:420] for y in d["ys"]], sc=d["sc"]))
        if g == fin: b["sfx"] = dict(text="SURVIVED", col="LIME", burst="GREEN", t=3.0, y=1000, size=120)
        B.append(b)
    B.append(beat("end", lines=["100 dinos.", "4 numbers each.", "Just evolution."], say="Follow for more AI from scratch."))
    common = dict(cacti=r["stats"]["cacti"], GY=760)
    post = dict(title="GOAL  100 AIs learn the Chrome dino game",
                caption=f"100 AIs play a Chrome-dino style runner. Each brain is 4 numbers: jump if w0*distance + w1*speed + w2*cactus height + w3 > 0. The furthest runners breed with small mutations. Gen 1: everyone dies. Gen {fin}: {fd}/100 survive the full run.",
                tags=["ai", "geneticalgorithm", "machinelearning", "dinogame", "coding", "simulation", "shipwithpip"],
                pinned="Genetic algorithm with a 4-weight linear policy. Keep the top 10 of 100, breed with uniform crossover and small Gaussian mutation. The game speeds up over time, which is what makes early gens die.",
                notes=f"Seed {r['stats']['seed']}. Speed 7 to 11.5 px/step, jump velocity -16.5, gravity 0.9, cactus gaps 420-760 px, run capped at 3000 steps (= survived). Survivors per generation: {[(g['g'], g['done']) for g in G]}. This is a simulation inspired by the Chrome dino game, not the original.")
    return B, common, post


def story_hunt(r):
    G = r["gens"]; pf = r["stats"]["prey_from"]; B = []
    for d in G:
        g = d["g"]; c = d["caught"]; learn = d["prey_learning"]
        head = (["8 WOLVES.", "40 RABBITS."] if g < pf else ["NOW THE RABBITS", "LEARN TOO."]) if g != G[-1]["g"] else ["IT'S AN", "ARMS RACE."]
        if g == 1: say, cap = f"Eight wolves. Forty rabbits. Both have random brains. Round one: {words(c)} caught, mostly by luck.", f"8 wolves, 40 rabbits, random brains. Round 1: {c} caught, mostly by luck."
        elif g == G[1]["g"]: say, cap = f"The wolves that catch the most breed. Round {words(g)}: {words(c)}.", f"The best hunters breed. Round {g}: {c}."
        elif g < pf: say, cap = f"Round {words(g)}: {words(c)} out of forty. The wolves hunt now. The rabbits can't learn yet.", f"Round {g}: {c}/40. The wolves hunt now. The rabbits can't learn yet."
        elif g == pf: say, cap = f"Round {words(g)}: {words(c)}. So I let the rabbits evolve too.", f"Round {g}: {c}. So I let the rabbits evolve too."
        elif g == G[-1]["g"]: say, cap = f"Round {words(g)}: {words(c)} caught. The rabbits learned to flee faster than the wolves learned to hunt. This is an arms race.", f"Round {g}: {c} caught. The rabbits learned to flee faster than the wolves learned to hunt. An arms race."
        else: say, cap = f"Round {words(g)}: {words(c)}. The rabbits are starting to run.", f"Round {g}: {c}. The rabbits are starting to run."
        b = beat("gen", head=head, label=f"ROUND {g}", stat=dict(text=f"{c} / 40 CAUGHT", col="RED" if c >= 25 else ("GREEN" if c <= 10 else "INK")), say=say, cap=cap, data=dict(frames=d["frames"], prey_learning=learn))
        if g == pf: b["sfx"] = dict(text="RABBITS LEARN", col="CYAN", burst="BLUE", t=2.6, y=1000, size=96)
        if g == G[-1]["g"]: b["sfx"] = dict(text=f"{c} CAUGHT", col="LIME", burst="GREEN", t=3.2, y=1000, size=120)
        B.append(b)
    B.append(beat("end", lines=["8 wolves.", "40 rabbits.", "An arms race."], say="Follow for more AI from scratch."))
    post = dict(title="GOAL  Wolves vs rabbits: two AI species evolve against each other",
                caption=f"8 wolves and 40 rabbits, each with a tiny neural net. First only the wolves evolve: catches go from {G[0]['caught']} to {max(g['caught'] for g in G if not g['prey_learning'])} of 40. Then the rabbits evolve too, and catches collapse to {G[-1]['caught']} by round {G[-1]['g']}. Co-evolution: an arms race between two populations.",
                tags=["ai", "evolution", "neuralnetwork", "simulation", "predatorprey", "coding", "shipwithpip"],
                pinned="Both species: MLP with 6 inputs (direction and distance to nearest enemy, position) and 2 outputs (turn, speed). Wolves are slightly faster. Fitness: catches for wolves, survival time for rabbits. Genetic algorithm each round.",
                notes=f"Seed {r['stats']['seed']}. 320 steps per round, catch radius 14 px, rabbits frozen (no evolution) until round {pf}. Catches per round: {r['stats']['caught']}. Names wolves/rabbits are a label for predators/prey in the sim.")
    return B, {}, post


STORY = dict(hoops=story_hoops, lander=story_lander, snake=story_snake, dino=story_dino, hunt=story_hunt)
from stories2 import STORY2  # noqa
from stories3 import STORY3  # noqa
STORY.update({k: (lambda r, f=f: f(r, words, pick, beat)) for k, f in list(STORY2.items()) + list(STORY3.items())})


# ---------------------------------------------------------------- audio helpers
def synth(text):
    parts = []
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        x, sr = voice.say(MODEL, s, VOICE, SPEED); assert sr == SR
        parts += [x / (np.max(np.abs(x)) or 1) * 0.95, np.zeros(int(SR * 0.18), np.float32)]
    return np.concatenate(parts[:-1])


def noise_burst(buf, t0, dur, amp, lo=True):
    n = int(dur * SR); i = int(t0 * SR); r = np.random.RandomState(int(t0 * 1000) % 9999); x = r.randn(n).astype(np.float32)
    if lo: x = np.convolve(x, np.ones(24) / 24, "same")
    env = np.exp(-np.linspace(0, 1, n) * 7); buf[i:i + n] += x / (np.max(np.abs(x)) or 1) * amp * env


def chime(b, t0, base=523):
    for k, f in enumerate((1, 1.26, 1.5, 2.0)): tone(b, t0 + k * 0.07, 1.2, base * f, base * f, 0.12, "sine")


class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass


def serve():
    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Q, directory=HERE)); threading.Thread(target=s.serve_forever, daemon=True).start(); return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("reel"); ap.add_argument("data"); ap.add_argument("out"); ap.add_argument("--stills"); ap.add_argument("--spec"); a = ap.parse_args()
    r = json.load(open(a.data))
    if a.spec:
        from specstory import spec_story
        B, common, post = spec_story(r, json.load(open(a.spec)), pick, beat)
    else:
        B, common, post = STORY[a.reel](r)
    LEAD = 0.5; V = []
    for b in B:
        if a.stills: V.append(None); continue
        V.append(synth(b["say"]))
    from playwright.sync_api import sync_playwright
    srv = serve(); port = srv.server_address[1]
    with sync_playwright() as p:
        br = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        pg = br.new_page(viewport={"width": 1080, "height": 1920}, device_scale_factor=1)
        pg.on("console", lambda m: print("console:", m.text) if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: print("pageerror:", e))
        pg.goto(f"http://127.0.0.1:{port}/index.html"); pg.wait_for_function("window.ready === true", timeout=60000)
        pg.evaluate("d => window.setData(d)", dict(reel=a.reel, common=common, beats=B))
        durs = []
        for i, b in enumerate(B):
            nf = pg.evaluate("i => window.frameCount(i)", i); vo = (LEAD + len(V[i]) / SR + 0.9) if V[i] is not None else 4.0
            sim = 0.35 + nf / FPS + 0.7 if b["kind"] != "end" else 0
            d = max(3.2, vo, sim)
            if b["kind"] != "end" and b.get("sfx"): d = max(d, b["sfx"]["t"] + 1.6)
            durs.append(round(d, 2))
        TL = pg.evaluate("c => window.setDur(c)", durs); T = TL["T"]; print("T", round(T, 1), durs)
        if a.stills:
            for t in [float(v) for v in a.stills.split(",")]:
                pg.evaluate("t => window.setFrame(t)", t); pg.screenshot(path=f"{os.path.splitext(a.out)[0]}_s{t:05.1f}.png")
            br.close(); return
        buf = np.zeros(int(T * SR) + SR * 3, np.float32); bed = music_bed(T, 50 + len(a.reel)); buf[:len(bed)] += bed[:len(buf)] * 0.3
        for i, b in enumerate(B):
            t0 = TL["beats"][i]["t0"]; x = V[i]; j = int((t0 + LEAD) * SR); buf[j:j + len(x)] += x
            tone(buf, t0, 0.22, 120, 50, 0.26, "sine"); noise_burst(buf, t0 + 0.01, 0.16, 0.18)
            if b.get("sfx"): ts = t0 + b["sfx"]["t"]; tone(buf, ts, 0.3, 90, 40, 0.3, "sine"); noise_burst(buf, ts, 0.25, 0.28); chime(buf, ts + 0.05)
            if b["kind"] == "end": chime(buf, t0 + 0.3, 440)
        buf = buf[:int(T * SR)]; wav = os.path.splitext(a.out)[0] + "_a.wav"; sf.write(wav, buf / (np.max(np.abs(buf)) or 1) * 0.92, SR)
        ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "mjpeg", "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-af", "loudnorm=I=-15:TP=-1.2:LRA=11", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
        n = int(T * FPS)
        for i in range(n):
            pg.evaluate("t => window.setFrame(t)", i / FPS); ff.stdin.write(pg.screenshot(type="jpeg", quality=92))
            if i % 150 == 0: print("frame", i, "/", n, flush=True)
        ff.stdin.close(); ff.wait(); br.close(); os.remove(wav)
    write_post(os.path.splitext(a.out)[0] + "_post.txt", post["title"], post["caption"], post["tags"], post["pinned"], post["notes"]); print("done", a.out)


main()

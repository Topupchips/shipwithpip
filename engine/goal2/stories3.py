"""Story builders: walk, shortcut, drone, password, keepups."""


def story_walk(r, words, pick, beat):
    G = r["gens"]; fin = G[-1]["g"]
    show = [1, 2, 4, 8, 15, 25, fin]; B = []
    for g in show:
        d = pick(G, g); best = int(round(d["best"])); mean = int(round(d["mean"])); bm = best / 100
        head = ["100 CREATURES.", "0 LEGS. 0 LESSONS."] if g < 8 else ["NOBODY", "TAUGHT THEM."]
        if g == 1: say, cap = "A hundred random creatures. Four blobs, six muscles, and a timer for each muscle. Nobody told them how to move. Generation one: the best one flops a few metres.", "100 random creatures. 4 blobs, 6 muscles, a timer per muscle. Gen 1: the best one flops a few metres."
        elif g == 2: say, cap = f"The ones that got furthest breed. Generation two: best, {words(best)}.", f"The furthest breed. Gen 2: best {best}."
        elif g == 4: say, cap = f"Generation {words(g)}: {words(best)}. Something that looks like a crawl.", f"Gen {g}: {best}. Something like a crawl."
        elif g == 8: say, cap = f"Generation {words(g)}: {words(best)}. Watch the leader. It found a rhythm.", f"Gen {g}: {best}. The leader found a rhythm."
        elif g == 15: say, cap = f"Generation {words(g)}: {words(best)}. The average creature now beats the gen-one champion.", f"Gen {g}: {best}. The average now beats the gen 1 champion."
        elif g == 25: say, cap = f"Generation {words(g)}: {words(best)}.", f"Gen {g}: {best}."
        else: say, cap = f"Generation {words(g)}: {words(best)}. Three thousand units in thirty seconds. Nobody designed the gait. Nobody wrote a walk cycle. The slow ones just didn't get kids.", f"Gen {g}: {best}. Nobody designed the gait. The slow ones just didn't get kids."
        b = beat("gen", head=head, label=f"GEN {g}", stat=dict(text=f"BEST {best}", col="GREEN" if best >= 2500 else "INK"), say=say, cap=cap, data=dict(runs=d["runs"], fr=d["fr"]))
        if g == 8: b["sfx"] = dict(text="A RHYTHM", col="YEL", burst="YEL", t=3.0, y=1000, size=110)
        if g == fin: b["sfx"] = dict(text=f"{best}", col="LIME", burst="GREEN", t=4.6, y=1000, size=130)
        B.append(b)
    B.append(beat("end", lines=["100 creatures.", "0 walk cycles.", "Just evolution."], say="Follow for more AI from scratch."))
    st = r["stats"]
    post = dict(title="GOAL  100 AI creatures learn to move (evolution, no instructions)",
                caption=f"100 soft-body creatures, 4 nodes and 6 muscles each. Each muscle has a short length, a long length, a timer and a phase. Each node has its own friction. The ones that travel furthest in 30 seconds breed with small mutations. Gen 1 best: {int(round(pick(G,1)['best']))}. Gen {fin}: {int(round(pick(G,fin)['best']))}. No walk cycle was programmed.",
                tags=["ai", "evolution", "geneticalgorithm", "simulation", "coding", "procedural", "shipwithpip"],
                pinned="Genome: 36 numbers. Verlet physics with gravity, ground contact, per-node friction and a velocity cap. Fitness = average x of the 4 nodes at the end of 900 steps. Top 10 breed, uniform crossover, Gaussian mutation on 20% of genes.",
                notes=f"Seed {st['seed']}. Best/mean per generation: {st['best']}.")
    return B, {}, post


def story_shortcut(r, words, pick, beat):
    G = r["gens"]; fin = G[-1]["g"]; st = r["stats"]
    show = [1, 2, 6, 9, 11, 28, fin]; B = []
    allwin = all(h[4] for h in st["hist"])
    for g in show:
        d = pick(G, g); u = d["used"]; laps = d["laps"]
        head = ["100 AI CARS.", "1 TRACK."] if g < 6 else (["ONE FOUND", "THE GAP."] if g < 28 else ["EVERY WINNER", "CHEATED."])
        if g == 1: say, cap = f"A hundred AI cars. Five distance sensors each, nothing else. I left a gap in the inner wall and told nobody. Lap one: pure chaos. But one car turns into the gap. By accident. And it wins.", "100 AI cars, 5 sensors each. I left a gap in the inner wall and told nobody. Gen 1: chaos. One car turns into the gap by accident. It wins."
        elif g == 2: say, cap = f"Generation two: the fastest cars breed. {words(laps)} laps now. Still just {words(u)} car taking the gap.", f"Gen 2: the fastest breed. {laps} laps. Still just {u} car taking the gap."
        elif g == 6: say, cap = f"Generation {words(g)}: {words(u)} cars cut through the middle.", f"Gen {g}: {u} cars cut through the middle."
        elif g == 9: say, cap = f"Generation {words(g)}: {words(u)}. And the first {words(laps)}-lap run, by a car that cheated.", f"Gen {g}: {u}. First {laps}-lap run, by a cheater."
        elif g == 11: say, cap = f"Generation {words(g)}: {words(u)} out of a hundred.", f"Gen {g}: {u}/100."
        elif g == 28: say, cap = f"Generation {words(g)}: {words(u)}. Here's the thing. The winner of every single generation so far took the gap.", f"Gen {g}: {u}. The winner of every single generation so far took the gap."
        else: say, cap = f"Generation {words(g)}: {words(u)} cars take the shortcut. And in all {words(fin)} generations, the fastest car cheated every time. I never told them the gap was there. Evolution doesn't care what you intended.", f"Gen {g}: {u} take the shortcut. In all {fin} generations the fastest car cheated every time. Evolution doesn't care what you intended."
        b = beat("gen", head=head, label=f"GEN {g}", stat=dict(text=f"{u} CHEATED", col="MAG" if u >= 10 else "INK"), say=say, cap=cap, data=dict(runs=d["runs"], usedat=d["usedat"], prog=d["prog"]))
        if g == 1: b["sfx"] = dict(text="THE GAP!", col="MAG", burst="RED", t=5.4, y=1000, size=120)
        if g == fin: b["sfx"] = dict(text="EVERY TIME", col="MAG", burst="RED", t=4.0, y=1000, size=110)
        B.append(b)
    B.append(beat("end", lines=["100 cars.", "1 gap.", "Evolution found it."], say="Follow for more AI from scratch."))
    post = dict(title="GOAL  I left a gap in the wall. The AI cars found it.",
                caption=f"100 AI cars with 5 distance sensors learn a race track by evolution. I left a 120 px gap in the inner wall and gave no hint. Gen 1: one car turns into it by accident and wins. By gen {fin}, {pick(G,fin)['used']} cars cut through, and in every one of the {fin} generations the fastest car was a car that took the gap. Fitness only counts laps.",
                tags=["ai", "neuralnetwork", "neuroevolution", "racing", "simulation", "coding", "shipwithpip"],
                pinned="Inputs: 5 ray distances (-70 to +70 degrees) and speed. Outputs: steering and throttle. Fitness = gates passed in 1400 steps (laps), minus a tiny penalty per wall bump. Crossing the corridor from the bottom lane counts as reaching the finish. Top 10 breed, 15% of weights mutated, 8 random newcomers per generation." + ("" if allwin else " Note: not every generation's winner used the gap."),
                notes=f"Seed {st['seed']}. (gen, max laps, mean progress, cars that used the gap, winner used the gap): {st['hist']}.")
    return B, {}, post


def story_drone(r, words, pick, beat):
    G = r["gens"]; st = r["stats"]; ma = st["move_at"]; fin = G[-1]["g"]
    show = [1, 11, 12, 18, 42, ma, 46, fin]; B = []
    for g in show:
        d = pick(G, g); n = d["done"]; ph = d["phase"]; best = max(d["rings"])
        head = ["100 DRONES.", "6 RINGS."] if ph == 0 else (["I SHRANK", "THE RINGS."] if g < fin else ["THEY THREADED", "THE NEEDLE."])
        if g == 1: say, cap = f"A hundred drones. A tiny neural net each. Fly through six rings in order. Generation one: the best drone gets through {words(best)}.", f"100 drones, tiny neural net each. 6 rings in order. Gen 1: best gets through {best}."
        elif g == 6: say, cap = f"Generation {words(g)}: the best gets {words(best)} rings. Nobody finishes.", f"Gen {g}: best {best} rings. Nobody finishes."
        elif g == 11: say, cap = f"Generation {words(g)}: {words(best)} rings. Still nobody finishes. But the ones that get furthest breed.", f"Gen {g}: {best} rings. Still nobody finishes."
        elif g == 12: say, cap = f"Generation {words(g)}: one drone finishes the whole course." if n == 1 else f"Generation {words(g)}: {words(n)} drones finish the whole course.", f"Gen {g}: {n} finish{'es' if n == 1 else ''} the whole course."
        elif g == 18: say, cap = f"Generation {words(g)}: {words(n)}.", f"Gen {g}: {n}."
        elif g == 42: say, cap = f"Generation {words(g)}: {words(n)} out of a hundred. So I shrank every ring to half the size.", f"Gen {g}: {n}/100. So I shrank every ring to half the size."
        elif g == ma: say, cap = f"{words(n).capitalize()}. Most of them clip the edge now.", f"{n}. Most clip the edge now."
        elif g == 46: say, cap = f"Generation {words(g)}: {words(n)}.", f"Gen {g}: {n}."
        else: say, cap = f"Generation {words(g)}: {words(n)} out of a hundred thread rings half the size. Nobody tuned a flight controller. The ones that crashed just didn't get kids.", f"Gen {g}: {n}/100 thread rings half the size. Nobody tuned a flight controller."
        b = beat("gen", head=head, label=f"GEN {g}", stat=dict(text=f"{n} / 100" if n else f"BEST {best} RINGS", col="GREEN" if n >= 25 else "INK"), say=say, cap=cap, data=dict(runs=d["runs"], rings=d["rings"], passed=d["passed"], phase=ph))
        if g == 12:
            tf = min(pa[5] for pa in d["passed"] if len(pa) >= 6); b["sfx"] = dict(text="FINISHED!", col="LIME", burst="GREEN", t=round(0.35 + tf / 4 / 30 + 0.15, 2), y=1000, size=110)
        if g == ma: b["sfx"] = dict(text="HALF SIZE", col="MAG", burst="RED", t=0.4, y=1000, size=110)
        B.append(b)
    B.append(beat("end", lines=["100 drones.", "0 flight controllers.", "Just evolution."], say="Follow for more AI from scratch."))
    post = dict(title="GOAL  100 AI drones learn a ring course (then I shrank the rings)",
                caption=f"100 drones, each a tiny neural net (6 inputs, 8 hidden, 2 outputs: thrust and tilt). Fly through 6 rings in order. 11 generations with no finisher, then gen 12: {pick(G,12)['done']}. Gen 42: {pick(G,42)['done']}/100. Then I halved the ring size: {pick(G,ma)['done']}, climbing back to {pick(G,fin)['done']}/100 by gen {fin}.",
                tags=["ai", "neuralnetwork", "drone", "neuroevolution", "simulation", "coding", "shipwithpip"],
                pinned="Inputs: offset to the next ring (x, y), velocity (x, y), bias, rings done. Gravity 0.22, thrust up to 0.55, side push up to 0.25, drag. Fitness = rings x 1000 minus distance to the next ring minus wall bumps. Top 10 breed, 15% of weights mutated.",
                notes=f"Seed {st['seed']}. Rings radius 46 then 26 from gen {ma}. Finishers per generation: {st['done']}.")
    return B, {}, post


def story_password(r, words, pick, beat):
    G = r["gens"]; st = r["stats"]; fin = G[-1]["g"]; target = st["target"]; L = len(target)
    first13 = next(g["g"] for g in G if g["best"] == L); stuck0 = next(g["g"] for g in G if g["best"] == L - 1)
    show = sorted(set([1, 2, 4, 7, stuck0, stuck0 + 8, first13 - 1, first13, first13 + 2, fin])); B = []
    for g in show:
        d = pick(G, g); best = d["best"]; solved = d["solved"]
        head = ["100 AIs GUESS", "MY PASSWORD."] if best < L - 1 else (["STUCK ON", "ONE LETTER."] if best < L else ["CRACKED.", "NOW IT SPREADS."])
        if g == 1: say, cap = f"A hundred AIs guess my password. Thirteen characters. They never see it. They only get told how many letters are in the right place. Generation one: best guess, {words(best)} right.", f"100 AIs guess my 13-character password. They only hear how many letters are in the right place. Gen 1: best {best} right."
        elif g == 2: say, cap = f"The closest guesses breed. Generation two: {words(best)} right.", f"Closest guesses breed. Gen 2: {best} right."
        elif g == 4: say, cap = f"Generation {words(g)}: {words(best)}. Letters lock in one by one.", f"Gen {g}: {best}. Letters lock in."
        elif g == 7: say, cap = f"Generation {words(g)}: {words(best)} of thirteen.", f"Gen {g}: {best}/13."
        elif g == stuck0: say, cap = f"Generation {words(g)}: twelve. One letter left. Watch what happens.", f"Gen {g}: 12. One letter left."
        elif g == stuck0 + 8: say, cap = f"Generation {words(g)}: still twelve. Every single one of them is stuck on the same letter.", f"Gen {g}: still 12. All stuck on the same letter."
        elif g == first13 - 1: say, cap = f"Generation {words(g)}: still twelve. A random mutation has to hit the right spot with the right letter. One in three hundred and fifty.", f"Gen {g}: still 12. A random mutation has to hit the right spot with the right letter. 1 in 351."
        elif g == first13: say, cap = f"Generation {words(g)}: one of them gets it." if solved == 1 else f"Generation {words(g)}: {words(solved)} of them get it.", f"Gen {g}: {solved} get{'s' if solved == 1 else ''} it."
        elif g == first13 + 2: say, cap = f"Two generations later: {words(solved)} have it. It spreads through breeding.", f"Gen {g}: {solved} have it. It spreads through breeding."
        else: say, cap = f"Generation {words(g)}: {words(solved)} out of a hundred have my password. Nobody told them a single letter. Just a score and breeding.", f"Gen {g}: {solved}/100 have it. Nobody told them a single letter."
        b = beat("gen", head=head, label=f"GEN {g}", stat=dict(text=f"{best} / {L} RIGHT" if solved == 0 else f"{solved} CRACKED", col="GREEN" if solved else ("RED" if best == L - 1 else "INK")), say=say, cap=cap, data=dict(guesses=d["guesses"], right=d["right"]))
        if g == first13: b["sfx"] = dict(text="CRACKED!", col="LIME", burst="GREEN", t=2.2, y=1000, size=120)
        B.append(b)
    B.append(beat("end", lines=["100 guesses.", "1 score.", "Just evolution."], say="Follow for more AI from scratch."))
    post = dict(title="GOAL  100 AIs crack a password with nothing but a score",
                caption=f"100 random 13-character guesses. Each guess is scored only by how many letters sit in the right place. The best 10 breed, with about one random letter changed per child. Gen 1: best {pick(G,1)['best']}/13. Gen {stuck0}: 12/13, then stuck on the last letter for {first13 - stuck0} generations. Gen {first13}: cracked. Gen {fin}: {pick(G,fin)['solved']}/100 have it. (Not how real password cracking works: real logins don't tell you how many letters are right.)",
                tags=["ai", "geneticalgorithm", "evolution", "simulation", "coding", "shipwithpip"],
                pinned="Alphabet: A to Z plus space (27). Fitness = positional matches. Uniform crossover from the top 10, mutation rate 1/13 per character. The stall at 12/13 is real: one specific position needs one specific letter, about 1 in 351 per child.",
                notes=f"Seed {st['seed']}. Target '{target}'. (gen, best, solved): {st['hist']}.")
    return B, dict(target=target), post


def story_keepups(r, words, pick, beat):
    G = r["gens"]; st = r["stats"]; ta = st["two_at"]; fin = G[-1]["g"]
    p0 = [g for g in G if g["phase"] == 0]; p1 = [g for g in G if g["phase"] == 1]; solved = p0[-1]["g"]; first1 = next(g["g"] for g in p1 if g["surv"] > 0); mid1 = next(g["g"] for g in p1 if g["surv"] >= 10)
    show = sorted(set([1, 3, solved, ta, first1, mid1, fin])); B = []
    for g in show:
        d = pick(G, g); n = d["surv"]; ph = d["phase"]; best = d["best"]
        head = ["100 AIs.", "KEEP THE BALL UP."] if ph == 0 else (["THEN I ADDED", "A SECOND BALL."] if g < fin else ["SOME CAN", "JUGGLE NOW."])
        if g == 1: say, cap = f"A hundred AIs. One paddle each. Keep the ball up for thirty seconds. Generation one: best run, {words(best)} touches. Nobody survives.", f"100 AIs, one paddle each. Keep the ball up 30 s. Gen 1: best {best} touches. Nobody survives."
        elif g == 3: say, cap = f"The longest rallies breed. Generation {words(g)}: {words(n)} survive the full thirty seconds.", f"Longest rallies breed. Gen {g}: {n} survive 30 s."
        elif g == 5: say, cap = f"Generation {words(g)}: {words(n)}.", f"Gen {g}: {n}."
        elif g == solved: say, cap = f"Generation {words(g)}: {words(n)} out of a hundred. Solved. So I dropped in a second ball.", f"Gen {g}: {n}/100. Solved. So I dropped in a second ball."
        elif g == ta: say, cap = "Zero. Two balls, one paddle. Every single one loses a ball in seconds.", "Zero. Two balls, one paddle. Everyone loses a ball in seconds."
        elif g == ta + 4: say, cap = f"Generation {words(g)}: still zero. But the rallies are getting longer.", f"Gen {g}: still 0. Rallies getting longer."
        elif g == first1: say, cap = f"Generation {words(g)}: {words(n)} keep{'s' if n == 1 else ''} both balls up for the full thirty seconds.", f"Gen {g}: {n} keep{'s' if n == 1 else ''} both up for 30 s."
        elif g == mid1: say, cap = f"Generation {words(g)}: {words(n)}. They're juggling.", f"Gen {g}: {n}. Juggling."
        else: say, cap = f"Generation {words(g)}: {words(n)} out of a hundred juggle two balls for thirty seconds. One paddle. No juggling rule. The ones that dropped a ball just didn't get kids.", f"Gen {g}: {n}/100 juggle two balls for 30 s. No juggling rule."
        rec = d["runs"]; hitTimes = []
        for run in rec:
            ht = []; 
            for s in range(1, len(run)):
                for k in range(1 + ph):
                    if run[s - 1][2 + k * 2] >= 200 and run[s][2 + k * 2] < run[s - 1][2 + k * 2]: ht.append(s); break
            hitTimes.append(ht)
        b = beat("gen", head=head, label=f"GEN {g}", stat=dict(text=f"{n} / 100", col="GREEN" if n >= 30 else ("RED" if n == 0 and ph == 1 else "INK")), say=say, cap=cap, data=dict(runs=rec, nballs=1 + ph, full=900, hitTimes=hitTimes))
        if g == ta: b["sfx"] = dict(text="2 BALLS", col="MAG", burst="RED", t=0.4, y=1000, size=120)
        if g == fin: b["sfx"] = dict(text="JUGGLING", col="LIME", burst="GREEN", t=4.0, y=1000, size=110)
        B.append(b)
    B.append(beat("end", lines=["100 paddles.", "2 balls.", "Just evolution."], say="Follow for more AI from scratch."))
    post = dict(title="GOAL  100 AIs learn keep-ups (then I added a second ball)",
                caption=f"100 AIs with one paddle each, tiny neural net. Keep the ball up for 30 seconds. Gen 1: nobody. Gen {solved}: {pick(G,solved)['surv']}/100. Then a second ball: 0 for {first1 - ta} generations, then {pick(G,first1)['surv']}, then {pick(G,fin)['surv']}/100 juggling both by gen {fin}. The reel shows 20 of the 100.",
                tags=["ai", "neuralnetwork", "neuroevolution", "simulation", "coding", "juggling", "shipwithpip"],
                pinned="Inputs: position and velocity of the two lowest balls relative to the paddle (8 numbers) plus bias. Output: paddle move. Bounce sets the ball's upward speed to a fixed value and adds spin from where it hits the paddle. Fitness = touches plus survival time. Top 10 breed, 5% of weights mutated.",
                notes=f"Seed {st['seed']}. (gen, survivors, best touches, mean touches): {st['hist']}.")
    return B, {}, post


STORY3 = dict(walk=story_walk, shortcut=story_shortcut, drone=story_drone, password=story_password, keepups=story_keepups)

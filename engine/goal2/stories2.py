"""Story builders for maze, flappy, moon, dodge, pole. Each returns (beats, common, post)."""


def _first(G, key, cond): return next((g["g"] for g in G if cond(g[key])), None)


def story_maze(r, words, pick, beat):
    G = r["gens"]; st = r["stats"]; bl = st["blocked_at"]; fin = G[-1]["g"]; N = len(G[0]["paths"])
    p0 = [g for g in G if g["phase"] == 0]; p1 = [g for g in G if g["phase"] == 1]
    first = next(g["g"] for g in p0 if g["reached"] > 0); solved = p0[-1]["g"]
    first1 = next(g["g"] for g in p1 if g["reached"] > 0); mid1 = next(g["g"] for g in p1 if g["reached"] >= 35)
    show = sorted(set([1, first - 1, first, solved, bl, bl + 1, first1, fin]))
    B = []
    for g in show:
        d = pick(G, g); n = d["reached"]; ph = d["phase"]
        head = ["100 AIs.", "1 MAZE. 0 MAPS."] if ph == 0 else (["I BLOCKED", "THEIR PATH."] if g < fin else ["THEY FOUND", "ANOTHER WAY."])
        if g == 1: say, cap = "A hundred AIs. One maze. No map. Each brain is just a list of a hundred moves. Generation one: nobody gets out.", "100 AIs. 1 maze. No map. Each brain is a list of 100 moves. Gen 1: nobody gets out."
        elif g == first - 1: say, cap = f"Generation {words(g)}: still nobody. But the ones that end closest to the exit get to breed.", f"Gen {g}: still nobody. But the ones that end closest to the exit breed."
        elif g == first: say, cap = f"Generation {words(g)}: {words(n)} gets out. The first one." if n == 1 else f"Generation {words(g)}: {words(n)} get out. The first ones.", f"Gen {g}: {n} out. The first."
        elif g == first + 2: say, cap = f"Two generations later: {words(n)}.", f"Gen {g}: {n}."
        elif g == solved: say, cap = f"Generation {words(g)}: {words(n)} out of a hundred. Solved. So I put a wall across their path.", f"Gen {g}: {n}/100. Solved. So I put a wall across their path."
        elif g == bl: say, cap = "Zero. Every single one walks into the new wall.", "Zero. Every one walks into the new wall."
        elif g == bl + 1: say, cap = "Still zero. They're all running the same dead route.", "Still zero. All running the same dead route."
        elif g == first1: say, cap = f"Generation {words(g)}: {words(n)}. One found a way around." if n == 1 else f"Generation {words(g)}: {words(n)} found a way around.", f"Gen {g}: {n} found a way around."
        elif g == mid1: say, cap = f"Generation {words(g)}: {words(n)}. The detour spreads through the population.", f"Gen {g}: {n}. The detour spreads."
        else: say, cap = f"Generation {words(g)}: {words(n)} out of a hundred. A new route, no map, no instructions. Just evolution.", f"Gen {g}: {n}/100. New route, no map, no instructions."
        b = beat("gen", head=head, label=f"GEN {g}", stat=dict(text=f"{n} / {N}", col="GREEN" if n >= 70 else ("RED" if n == 0 and ph == 1 else "INK")), say=say, cap=cap, data=dict(paths=d["paths"], ok=d["ok"], phase=ph))
        if g == bl: b["sfx"] = dict(text="BLOCKED!", col="RED", burst="RED", t=0.4, y=1000, size=120)
        if g == fin: b["sfx"] = dict(text="NEW ROUTE", col="LIME", burst="GREEN", t=2.4, y=1000, size=110)
        B.append(b)
    B.append(beat("end", lines=["100 AIs.", "0 maps.", "Just evolution."], say="Follow for more AI from scratch."))
    w0 = next(g["walls"] for g in G if g["walls"]); w1 = next((g["walls"] for g in G if g["walls"] and g["phase"] == 1), w0)
    common = dict(walls0=w0, walls1=w1, block=next((g["block"] for g in G if g["block"]), None))
    post = dict(title="GOAL  100 AIs escape a maze with no map (then I blocked the path)",
                caption=f"100 AIs in a 14x14 maze, no map. Each brain is a list of 100 moves. The ones that end closest to the exit breed, with a few random moves changed. Gen 1: 0 out. Gen {solved}: {pick(G,solved)['reached']}/100. Then I walled off their route: 0, 0, then {pick(G,first1)['reached']} by gen {first1}, {pick(G,fin)['reached']}/100 by gen {fin}. Genetic algorithm, no pathfinding.",
                tags=["ai", "geneticalgorithm", "maze", "simulation", "coding", "machinelearning", "shipwithpip"],
                pinned="Genome = 100 moves (up/down/left/right). Moving into a wall wastes the move. Fitness = -10 x distance-to-exit (true path distance) minus wasted moves, plus a bonus for reaching the exit. Top 10 breed with one-point crossover and ~2% mutation; 10% of kids re-roll their tail.",
                notes=f"Seed {st['seed']}. Maze: DFS perfect maze with 20 extra openings.  Block placed on the best path where it lengthens the route most (exit stays reachable). Reached per generation: {st['reached']}.")
    return B, common, post


def story_flappy(r, words, pick, beat):
    G = r["gens"]; st = r["stats"]; fin = G[-1]["g"]; N = len(G[0]["ys"]); ms = st["maxsteps"]
    first = next(g["g"] for g in G if g["surv"] > 0); big = next(g["g"] for g in G if g["surv"] >= 30)
    bests = [(g["g"], g["best"]) for g in G if g["g"] < first]
    show = sorted(set([1, 2, bests[len(bests) // 2][0], first - 1, first, first + 1, big, fin]))
    B = []
    def pipes_of(steps): return max(0, (steps * 5 - 600 + 90) // 320 + 1) if steps * 5 > 600 else 0
    for g in show:
        d = pick(G, g); n = d["surv"]; pp = pipes_of(d["best"]); done = n > 0
        head = ["100 BIRDS.", "1 BRAIN EACH."] if g < big else ["NOBODY", "TAUGHT THEM."]
        if g == 1: say, cap = f"A hundred birds. A tiny neural net each. Nobody told them what a pipe is. Generation one: the best bird makes it past {words(pp)} pipe{'s' if pp != 1 else ''}." if pp else "A hundred birds. A tiny neural net each. Nobody told them what a pipe is. Generation one: nobody even reaches the first pipe.", f"100 birds, a tiny neural net each. Gen 1: best bird passes {pp} pipe{'s' if pp != 1 else ''}." if pp else "100 birds, a tiny neural net each. Gen 1: nobody reaches the first pipe."
        elif g == 2: say, cap = f"The ones that fly furthest breed. Generation two: best bird, {words(pp)} pipes.", f"The furthest flyers breed. Gen 2: best bird {pp} pipes."
        elif not done and g == first - 1: say, cap = f"Generation {words(g)}: the best one gets {words(pp)} pipes in, then clips the edge. Still zero survive.", f"Gen {g}: best gets {pp} pipes in, then clips the edge. 0 survive."
        elif not done: say, cap = f"Generation {words(g)}: {words(pp)} pipes. Closer every time. Still nobody survives the full run.", f"Gen {g}: {pp} pipes. Closer every time. 0 survive."
        elif g == first: say, cap = f"Generation {words(g)}: {words(n)} bird{'s' if n != 1 else ''} survive{'' if n != 1 else 's'} the whole run. The first.", f"Gen {g}: {n} survive{'' if n != 1 else 's'} the whole run. The first."
        elif g == first + 1: say, cap = f"One generation later: {words(n)}.", f"Gen {g}: {n}."
        elif g == big: say, cap = f"Generation {words(g)}: {words(n)} out of a hundred.", f"Gen {g}: {n}/100."
        else: say, cap = f"Generation {words(g)}: {words(n)} out of a hundred survive all {words(pipes_of(ms))} pipes. Nobody taught them. The ones that crashed just didn't get kids.", f"Gen {g}: {n}/100 survive all {pipes_of(ms)} pipes. Nobody taught them. The ones that crashed just didn't get kids."
        b = beat("gen", head=head, label=f"GEN {g}", stat=dict(text=f"{n} / {N}" if done else f"BEST {pp} PIPES", col="GREEN" if n >= 40 else "INK"), say=say, cap=cap, data=dict(ys=[y[:480] for y in d["ys"]], full=480))
        if g == first: b["sfx"] = dict(text="SURVIVED!", col="LIME", burst="GREEN", t=4.2, y=1000, size=110)
        if g == fin: b["sfx"] = dict(text=f"{n} / 100", col="LIME", burst="GREEN", t=5.6, y=1000, size=130)
        B.append(b)
    B.append(beat("end", lines=["100 birds.", "0 lessons.", "Just evolution."], say="Follow for more AI from scratch."))
    common = dict(pipes=st["pipes"])
    post = dict(title="GOAL  100 AIs learn Flappy Bird from zero",
                caption=f"100 birds, each with a tiny neural net (4 inputs, 6 hidden, 1 output: flap or not). The furthest flyers breed with small mutations. {first-1} generations of zero survivors while the best bird gets a little further each time, then gen {first}: {pick(G,first)['surv']} survive the full run. Gen {fin}: {pick(G,fin)['surv']}/100. Neuroevolution, no instructions.",
                tags=["ai", "neuralnetwork", "flappybird", "neuroevolution", "simulation", "coding", "shipwithpip"],
                pinned="Inputs: height difference to the next gap, distance to the next pipe, vertical speed, bias. Output > 0 = flap. Fitness = steps survived. Keep the top 10, uniform crossover, Gaussian mutation on ~6% of weights.",
                notes=f"Seed {st['seed']}. Gravity 0.45, flap -8.5, pipe gap 250 px, pipes every 320 px, scroll 5 px/step, full run {ms} steps = {pipes_of(ms)} pipes. The reel shows the first 480 steps of each run. Survivors/best per generation: {st['surv']}.")
    return B, common, post


def story_moon(r, words, pick, beat):
    G = r["gens"]; st = r["stats"]; hv = st["heavy_at"]; fin = G[-1]["g"]; N = len(G[0]["shots"])
    p1 = [g for g in G if g["phase"] == 1]; first1 = next(g["g"] for g in p1 if g["hit"] > 0); mid1 = next(g["g"] for g in p1 if g["hit"] >= 50)
    show = sorted(set([1, 2, 3, hv - 1, hv, first1, mid1, fin]))
    B = []
    for g in show:
        d = pick(G, g); n = d["hit"]; ph = d["phase"]
        head = ["100 PROBES.", "1 MOON."] if ph == 0 else (["I DOUBLED", "THE GRAVITY."] if g < fin else ["THEY RE-LEARNED", "THE SLINGSHOT."])
        if g == 1: say, cap = f"A hundred probes launch from Earth. A planet sits between them and the Moon. Each probe has two numbers: angle and speed. Launch one: {words(n)} hit.", f"100 probes. A planet sits between them and the Moon. 2 numbers each: angle and speed. Launch 1: {n} hit."
        elif g == 2: say, cap = f"Closest misses breed. Launch two: {words(n)}. Some are starting to curve around the planet.", f"Closest misses breed. Launch 2: {n}. Some start to curve around the planet."
        elif g == 3: say, cap = f"Launch three: {words(n)}.", f"Launch 3: {n}."
        elif g == hv - 1: say, cap = f"Launch {words(g)}: {words(n)} out of a hundred. They found the slingshot. So I doubled the planet's gravity.", f"Launch {g}: {n}/100. They found the slingshot. So I doubled the planet's gravity."
        elif g == hv: say, cap = "Zero. The old path bends straight into the planet now.", "Zero. The old path bends straight into the planet now."
        elif g == first1: say, cap = f"Launch {words(g)}: {words(n)}. The first ones to survive the heavier pull.", f"Launch {g}: {n}. First through the heavier pull."
        elif g == mid1: say, cap = f"Launch {words(g)}: {words(n)}.", f"Launch {g}: {n}."
        else: say, cap = f"Launch {words(g)}: {words(n)} out of a hundred. A new slingshot for a new planet. No physics lessons. Just trial, error, and breeding.", f"Launch {g}: {n}/100. A new slingshot, no physics lessons. Just trial, error and breeding."
        b = beat("gen", head=head, label=f"LAUNCH {g}", stat=dict(text=f"{n} / {N}", col="GREEN" if n >= 80 else ("RED" if n == 0 and ph == 1 else "INK")), say=say, cap=cap, data=dict(shots=d["shots"], ok=d["ok"], crash=d["crash"], phase=ph))
        if g == hv: b["sfx"] = dict(text="GRAVITY x2", col="MAG", burst="RED", t=0.4, y=1000, size=110)
        if g == fin: b["sfx"] = dict(text="RE-LEARNED", col="LIME", burst="GREEN", t=2.4, y=1000, size=110)
        B.append(b)
    B.append(beat("end", lines=["100 probes.", "2 numbers each.", "Just evolution."], say="Follow for more AI from scratch."))
    post = dict(title="GOAL  100 AIs learn a gravity slingshot (then I doubled the gravity)",
                caption=f"100 probes launch from Earth toward a Moon hidden behind a planet. Each probe is two numbers, angle and speed. Closest misses breed. Launch 1: {pick(G,1)['hit']} hit. Launch {hv-1}: {pick(G,hv-1)['hit']}/100 via a slingshot around the planet. Then I doubled the planet's gravity: 0, then {pick(G,first1)['hit']}, then {pick(G,fin)['hit']}/100 by launch {fin}.",
                tags=["ai", "geneticalgorithm", "space", "physics", "simulation", "coding", "shipwithpip"],
                pinned="Point-mass gravity a = mu / r^2 toward the planet, Euler steps. Fitness = -closest distance to the Moon, +500 for a hit, -200 for crashing into the planet. Keep top 10, uniform crossover, Gaussian mutation that shrinks each launch.",
                notes=f"Seed {st['seed']}. Earth (110,520), planet (520,500) radius 80, Moon (880,300) radius 34, mu 9000 then 18000 from launch {hv}. Hits per launch: {st['hits']}.")
    return B, {}, post


def story_dodge(r, words, pick, beat):
    G = r["gens"]; st = r["stats"]; sa = st["storm_at"]; fin = G[-1]["g"]; N = len(G[0]["xs"])
    p0 = [g for g in G if g["phase"] == 0]; p1 = [g for g in G if g["phase"] == 1]
    first = next(g["g"] for g in p0 if g["surv"] > 0); solved = p0[-1]["g"]; mid0 = next(g["g"] for g in p0 if g["surv"] >= 40)
    mid1 = next(g["g"] for g in p1 if g["surv"] >= 40)
    show = sorted(set([1, first - 1, first, solved, sa, sa + 1, mid1, fin]))
    B = []
    for g in show:
        d = pick(G, g); n = d["surv"]; ph = d["phase"]
        head = ["100 AIs.", "1 ASTEROID STORM."] if ph == 0 else (["I ADDED", "70 MORE ROCKS."] if g < fin else ["THEY SURVIVED", "THAT TOO."])
        if g == 1: say, cap = "A hundred AIs in an asteroid storm. Each one sees the three nearest rocks, nothing else. Generation one: everyone dies.", "100 AIs in an asteroid storm. Each sees only the 3 nearest rocks. Gen 1: everyone dies."
        elif g == first // 2: say, cap = f"Generation {words(g)}: still nobody survives. But the ones that last longest get to breed.", f"Gen {g}: still nobody survives. The ones that last longest breed."
        elif g == first - 1: say, cap = f"Generation {words(g)}: so close. The last one dies with seconds left.", f"Gen {g}: so close. The last one dies with seconds left."
        elif g == first: say, cap = f"Generation {words(g)}: {words(n)} survive{'s' if n == 1 else ''} the whole storm.", f"Gen {g}: {n} survive{'s' if n == 1 else ''} the whole storm."
        elif g == mid0: say, cap = f"Generation {words(g)}: {words(n)}.", f"Gen {g}: {n}."
        elif g == solved: say, cap = f"Generation {words(g)}: {words(n)} out of a hundred. So I added seventy more rocks to the same storm.", f"Gen {g}: {n}/100. So I added 70 more rocks to the same storm."
        elif g == sa: say, cap = f"{words(n).capitalize()} survive{'s' if n == 1 else ''}. Everything they learned is suddenly not enough.", f"{n} survive{'s' if n == 1 else ''}. Everything they learned is suddenly not enough."
        elif g == sa + 1: say, cap = f"Generation {words(g)}: {words(n)}.", f"Gen {g}: {n}."
        elif g == mid1: say, cap = f"Generation {words(g)}: {words(n)}. They're adapting.", f"Gen {g}: {n}. Adapting."
        else: say, cap = f"Generation {words(g)}: {words(n)} out of a hundred survive the heavier storm. Nobody wrote a dodge rule. The ones that got hit just didn't get kids.", f"Gen {g}: {n}/100 survive the heavier storm. Nobody wrote a dodge rule."
        b = beat("gen", head=head, label=f"GEN {g}", stat=dict(text=f"{n} / {N}", col="GREEN" if n >= 50 else ("RED" if n <= 2 and ph == 1 else "INK")), say=say, cap=cap, data=dict(xs=d["xs"], rocks=d["rocks"], phase=ph, full=480))
        if g == sa: b["sfx"] = dict(text="+70 ROCKS", col="ORANGE", burst="RED", t=0.4, y=1000, size=120)
        if g == fin: b["sfx"] = dict(text="SURVIVED", col="LIME", burst="GREEN", t=4.0, y=1000, size=120)
        B.append(b)
    B.append(beat("end", lines=["100 AIs.", "0 dodge rules.", "Just evolution."], say="Follow for more AI from scratch."))
    post = dict(title="GOAL  100 AIs survive an asteroid storm (then I added 70 more rocks)",
                caption=f"100 AIs, each a tiny neural net that only sees the 3 nearest rocks. The ones that last longest breed. {first-1} generations of total wipeout, then gen {first}: {pick(G,first)['surv']} survive. Gen {solved}: {pick(G,solved)['surv']}/100. Then I added 70 rocks to the same storm: {pick(G,sa)['surv']}, then {pick(G,sa+1)['surv']}, then {pick(G,fin)['surv']}/100 by gen {fin}.",
                tags=["ai", "neuralnetwork", "neuroevolution", "simulation", "coding", "space", "shipwithpip"],
                pinned="Inputs: relative x and y of the 3 nearest rocks above the ship, own x, bias (8 inputs, 8 hidden, 1 output = move left/right). Fitness = steps survived. Top 10 breed, Gaussian mutation on ~5% of weights. Same rock layout every generation, so this is partly memorisation, which is why adding rocks hurts so much.",
                notes=f"Seed {st['seed']}. 480 steps per run, ship speed 6 px/step, rocks 4.5-7.5 px/step, radius 16-34. Storm 1: ~60 rocks; storm 2: the same plus ~70 more. Survivors per generation: {st['surv']}.")
    return B, {}, post


def story_pole(r, words, pick, beat):
    G = r["gens"]; st = r["stats"]; pa = st["push_at"]; fin = G[-1]["g"]; N = len(G[0]["runs"])
    p0 = [g for g in G if g["phase"] == 0]; p1 = [g for g in G if g["phase"] == 1]
    solved = p0[-1]["g"]; mid0 = next(g["g"] for g in p0 if g["surv"] >= 15); first1 = next(g["g"] for g in p1 if g["surv"] > 0); mid1 = next(g["g"] for g in p1 if g["surv"] >= 15)
    show = sorted(set([1, 2, mid0, solved, pa, first1, mid1, fin]))
    B = []
    for g in show:
        d = pick(G, g); n = d["surv"]; ph = d["phase"]
        head = ["50 AIs.", "1 POLE EACH."] if ph == 0 else (["THEN I", "PUSHED THEM."] if g < fin else ["THEY LEARNED", "TO TAKE A HIT."])
        if g == 1: say, cap = f"Fifty AIs. Each one has to balance a pole on a cart. The brain is a tiny neural net. Generation one: {words(n)} last{'s' if n == 1 else ''} the full nine seconds.", f"50 AIs balance a pole on a cart. Tiny neural net each. Gen 1: {n} last{'s' if n == 1 else ''} the full 9 seconds."
        elif g == 2: say, cap = f"The ones that stay up longest breed. Generation two: {words(n)}.", f"The longest standing breed. Gen 2: {n}."
        elif g == mid0: say, cap = f"Generation {words(g)}: {words(n)}. Watch the poles stop wobbling.", f"Gen {g}: {n}. Watch the poles stop wobbling."
        elif g == solved: say, cap = f"Generation {words(g)}: {words(n)} out of fifty. Solved. So I started shoving the pole halfway through.", f"Gen {g}: {n}/50. Solved. So I started shoving the pole halfway through."
        elif g == pa: say, cap = "Zero. One push and every single one falls.", "Zero. One push and every one falls."
        elif g == first1: say, cap = f"Generation {words(g)}: {words(n)} recover{'s' if n == 1 else ''} from the push.", f"Gen {g}: {n} recover{'s' if n == 1 else ''} from the push."
        elif g == mid1: say, cap = f"Generation {words(g)}: {words(n)}. They're learning to catch it.", f"Gen {g}: {n}. Learning to catch it."
        else: say, cap = f"Generation {words(g)}: {words(n)} out of fifty take the hit and stay up. Nobody programmed a recovery. The ones that fell just didn't get kids.", f"Gen {g}: {n}/50 take the hit and stay up. Nobody programmed a recovery."
        b = beat("gen", head=head, label=f"GEN {g}", stat=dict(text=f"{n} / {N}", col="GREEN" if n >= 30 else ("RED" if n == 0 and ph == 1 else "INK")), say=say, cap=cap, data=dict(runs=d["runs"], phase=ph, full=450))
        if ph == 1: b["sfx"] = dict(text="PUSH!", col="RED", burst="RED", t=0.35 + 220 / 3 / 30, y=1000, size=120)
        if g == fin: b["sfx"] = dict(text="PUSH!", col="RED", burst="RED", t=0.35 + 220 / 3 / 30, y=1000, size=120)
        B.append(b)
    B.append(beat("end", lines=["50 AIs.", "1 shove.", "Just evolution."], say="Follow for more AI from scratch."))
    post = dict(title="GOAL  50 AIs learn to balance a pole (then I pushed them)",
                caption=f"50 AIs balance a pole on a cart, tiny neural net each. The ones that stay up longest breed. Gen 1: {pick(G,1)['surv']}. Gen {solved}: {pick(G,solved)['surv']}/50. Then I added a shove to the pole at 4.4 seconds: 0 survive, then {pick(G,first1)['surv']}, then {pick(G,fin)['surv']}/50 by gen {fin}. Neuroevolution on the classic cart-pole problem.",
                tags=["ai", "neuralnetwork", "cartpole", "reinforcementlearning", "simulation", "coding", "shipwithpip"],
                pinned="Classic cart-pole physics (cart 1 kg, pole 0.1 kg, half-length 0.5 m, force up to 10 N, dt 0.02 s). Inputs: cart position, cart speed, pole angle, pole angular speed. Fail at 12 degrees or the track edge. 9-second runs; the push adds 1.9 rad/s to the pole at step 220.",
                notes=f"Seed {st['seed']}. Population 50, keep top 10, Gaussian mutation on 15% of weights. Survivors per generation: {st['surv']}.")
    return B, {}, post


STORY2 = dict(maze=story_maze, flappy=story_flappy, moon=story_moon, dodge=story_dodge, pole=story_pole)

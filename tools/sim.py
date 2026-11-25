#!/usr/bin/env python3
"""One-command helpers for a sim run, so each scheduled run spends few tokens.

  python3 tools/sim.py plan                  -> slot, sim, series part, voice mode, earlier topics, review due?
  python3 tools/sim.py seeds SIM [N] [--kw k=v,..] [--seeds a,b,..]
                                             -> runs N fresh seeds, prints ONE compact arc line per seed
  python3 tools/sim.py build SIM SEED SPEC [--kw k=v,..] [--record 1,5,..]
                                             -> rerun seed, check, render punch reel, verify, contact sheet
  python3 tools/sim.py prep SPEC             -> host video, print the final caption (series, caption, question, follow, tags)
  python3 tools/sim.py log SPEC SEED PERMALINK [note]  -> append log, advance state, commit, push
  python3 tools/sim.py fail SIM "reason"     -> log a failed run (state unchanged), commit, push
Outputs: /tmp/reel.mp4, /tmp/sheet.png (6 frames), /tmp/caption.txt, /tmp/video_url.txt
"""
import datetime, json, os, random, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G2 = os.path.join(REPO, "engine", "goal2")
STATE, LOG = os.path.join(REPO, "state", "state.json"), os.path.join(REPO, "state", "log.jsonl")
os.environ.setdefault("KOKORO_DIR", os.path.expanduser("~/kokoro-multi-lang-v1_0"))
HEAVY = {"runs", "ys", "xs", "paths", "shots", "games", "frames", "guesses", "rocks", "fit", "fr", "usedat", "prog", "used_i",
         "right", "rings", "passed", "sc", "scores", "walls", "block", "hits", "sw", "ok", "crash", "hoop"}


def sh(cmd, **k): return subprocess.run(cmd, shell=True, text=True, capture_output=True, **k)
def state(): return json.load(open(STATE))
def logs(): return [json.loads(l) for l in open(LOG) if l.strip()]
def posted(): return [l for l in logs() if str(l.get("status", "")).startswith("posted")]


def plan():
    s = state(); sim = s["sim_rotation"][s["sim_index"] % len(s["sim_rotation"])]
    prev = [l for l in posted() if l.get("sim") == sim]
    last = posted()[-1]["date"] if posted() else None
    now = datetime.datetime.now(datetime.timezone.utc)
    recent = last and (now - datetime.datetime.fromisoformat(last.replace("Z", "+00:00"))).total_seconds() < 5400
    lr = s.get("last_review"); due = not lr or (now - datetime.datetime.fromisoformat(lr.replace("Z", "+00:00"))).total_seconds() > 48 * 3600
    print(json.dumps(dict(sim=sim, part=len(prev) + 1, voice=s["voice_modes"][s["voice_index"] % len(s["voice_modes"])],
                          used_seeds=[l.get("seed") for l in prev], earlier_topics=[l["topic"] for l in prev],
                          posted_within_90_min=bool(recent), review_due=due, retired=s.get("retired_sims", []))))


def arc_line(r):
    out = []
    for d in r["gens"]:
        v = [f"{k}={round(x, 1) if isinstance(x, float) else x}" for k, x in d.items() if k not in HEAVY and k != "g" and not isinstance(x, (list, dict))]
        out.append(f"{d['g']}:" + ",".join(v))
    st = {k: v for k, v in r["stats"].items() if k not in ("pipes", "cacti", "reel", "seed") and not isinstance(v, (list, dict))}
    return json.dumps(st) + " | " + " ".join(out)


def kwargs(argv):
    kw = argv[argv.index("--kw") + 1] if "--kw" in argv else ""
    rec = argv[argv.index("--record") + 1] if "--record" in argv else ""
    return (f" --kw {kw}" if kw else "") + (f" --record {rec}" if rec else "")


def seeds(argv):
    sim = argv[0]; n = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 3
    used = {l.get("seed") for l in posted() if l.get("sim") == sim}
    pool = [int(x) for x in argv[argv.index("--seeds") + 1].split(",")] if "--seeds" in argv else [x for x in random.sample(range(10, 9999), 40) if x not in used][:n]
    for sd in pool:
        p = f"/tmp/s_{sim}_{sd}.json"; r = sh(f"cd {G2} && python3 autosim.py run {sim} {sd} {p}{kwargs(argv)}")
        if r.returncode: print(sd, "ERROR", r.stderr[-300:]); continue
        print(f"seed {sd}: " + arc_line(json.load(open(p)))[:1500])


def build(argv):
    sim, seed, spec = argv[0], argv[1], argv[2]; data = "/tmp/r.json"
    r = sh(f"cd {G2} && python3 autosim.py run {sim} {seed} {data}{kwargs(argv)}")
    if r.returncode: sys.exit("run failed: " + r.stderr[-500:])
    c = sh(f"cd {G2} && python3 autosim.py check {data} {spec}"); print(c.stdout.strip())
    if c.returncode: sys.exit(1)
    sp = json.load(open(spec)); bad = [m for m in re.findall("—", json.dumps(sp, ensure_ascii=False))]
    if bad: sys.exit("em dash in spec")
    sh(f"cd {REPO}/engine && python3 -c \"import scale; scale.patch_lexicon('{os.environ['KOKORO_DIR']}')\"")
    rr = sh(f"cd {G2} && timeout 1500 python3 -u goal2.py {sim} {data} /tmp/reel.mp4 --spec {spec}")
    log = rr.stdout + rr.stderr
    issues = [l for l in log.splitlines() if re.search(r"OOV|pageerror|Traceback|Error", l)]
    T = re.search(r"^T ([\d.]+)", log, re.M)
    if rr.returncode or "done /tmp/reel.mp4" not in log: sys.exit("render failed:\n" + log[-1500:])
    dur = float(sh("ffprobe -v error -show_entries format=duration -of csv=p=0 /tmp/reel.mp4").stdout or 0)
    aud = "audio" in sh("ffprobe -v error -show_entries stream=codec_type -of csv=p=0 /tmp/reel.mp4").stdout
    wh = sh("ffprobe -v error -select_streams v -show_entries stream=width,height -of csv=p=0 /tmp/reel.mp4").stdout.strip()
    ts = [0.3] + [round(dur * k / 5, 2) for k in (1, 2, 3, 4)] + [round(dur - 0.6, 2)]
    for i, t in enumerate(ts): sh(f"ffmpeg -v error -y -ss {t} -i /tmp/reel.mp4 -frames:v 1 -vf scale=270:-1 /tmp/f{i}.png")
    sh("ffmpeg -v error -y " + " ".join(f"-i /tmp/f{i}.png" for i in range(6)) + " -filter_complex hstack=6 /tmp/sheet.png")
    ok = 20 <= dur <= 35 and aud and wh == "1080,1920" and not issues
    print(json.dumps(dict(T=T and T.group(1), duration=round(dur, 1), audio=aud, size=wh, frames_at=ts, issues=issues[:8], ok=ok)))
    print("Read /tmp/sheet.png (6 frames) before posting.")


def prep(argv):
    sp = json.load(open(argv[0])); p = sp["post"]; s = state()
    sim = sp["reel"]; part = len([l for l in posted() if l.get("sim") == sim]) + 1
    cap = f"AI learns {sim}, part {part}.\n\n{p['caption']}\n\n{sp.get('end_say', 'What should it learn next?')}\n\nFollow @shipwithpip, a new part every day.\n\n" + " ".join("#" + t for t in p["tags"])
    if "—" in cap or len(cap) > 2200: sys.exit("caption has an em dash or is over 2200 chars")
    open("/tmp/caption.txt", "w").write(cap)
    url = sh(f"bash {REPO}/tools/host_video.sh /tmp/reel.mp4").stdout.strip().splitlines()[-1]
    open("/tmp/video_url.txt", "w").write(url)
    print(json.dumps(dict(url=url, caption=cap, account=s["account"]["platformAccountId"])))


def log(argv):
    spec, seed, link = argv[0], int(argv[1]), argv[2]; note = argv[3] if len(argv) > 3 else ""
    sp = json.load(open(spec)); s = state(); sim = sp["reel"]
    line = dict(date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), kind="sim", sim=sim, seed=seed,
                topic=sp["post"]["title"], permalink=link, style="punch", voice=sp.get("voice"), status="posted" + (f" ({note})" if note else ""))
    open(LOG, "a").write(json.dumps(line) + "\n")
    s["cycle_index"] += 1; s["next"] = s["cycle"][s["cycle_index"] % len(s["cycle"])]; s["sim_index"] += 1; s["voice_index"] += 1
    json.dump(s, open(STATE, "w"), indent=2); open(STATE, "a").write("\n")
    os.makedirs(f"{REPO}/state/specs", exist_ok=True); json.dump(sp, open(f"{REPO}/state/specs/{sim}_{seed}.json", "w"), indent=1)
    msg = f"log: {sp['post']['title']}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    r = sh(f"cd {REPO} && git add state && git commit -qm {json.dumps(msg)} && git push -q origin HEAD:main")
    print("logged and pushed" if r.returncode == 0 else "push problem: " + r.stderr[-400:])


def fail(argv):
    sim, reason = argv[0], argv[1]
    line = dict(date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), kind="sim", sim=sim, seed=None, topic=None, permalink=None, status="failed: " + reason)
    open(LOG, "a").write(json.dumps(line) + "\n")
    r = sh(f"cd {REPO} && git add state && git commit -qm {json.dumps('log: failed run')} && git push -q origin HEAD:main")
    print("failure logged" if r.returncode == 0 else "push problem: " + r.stderr[-400:])


if __name__ == "__main__":
    a = sys.argv[1:]
    dict(plan=lambda: plan(), seeds=lambda: seeds(a[1:]), build=lambda: build(a[1:]), prep=lambda: prep(a[1:]), log=lambda: log(a[1:]), fail=lambda: fail(a[1:]))[a[0]]()

# @shipwithpip auto-reels: runbook for each scheduled run

Every run makes **one** reel and posts it to Instagram **@shipwithpip** with no human review.
Ansh approved fully automatic posting, 2 reels a day, alternating an AI simulation reel and a dark explainer reel.
Because nobody reviews it, the checks below are the review. If any check fails and you cannot fix it honestly, **post nothing** and log the failure.

## Hard rules (never break)
1. **Honest numbers.** Every number in a sim reel comes from the run you just did (`autosim.py stats`). Never reuse numbers from an old run, never round in a way that changes meaning, never claim something the stats do not show. `autosim.py check` must print `CHECK OK`.
2. **Real facts for explainers.** Every claim in an explainer is verified with WebSearch/WebFetch against a primary or reputable source found in this run. List the URLs in the script's `sources`. No invented stats, quotes, companies or incidents. If you cannot verify a claim, cut it.
3. **No em dashes** anywhere (script, captions, on-screen text, post). Use commas, colons or full stops.
4. Post only to account id `29510734025188123` (@shipwithpip). Never to @anshgotenks or @anshgoten.
5. One reel per run. If the log shows a reel was already posted for this slot within the last 6 hours, stop.
6. Keep the video vertically centred, 1080x1920. Length 35 to 65 s.
7. Hooks: the first 2 seconds must be a surprising, specific line (a number, a reversal, a "then I..." twist). Ansh wants more random, unexpected hooks; capturing attention in the first 10 seconds is the priority.

## 0. Setup (every run, fresh container)
```bash
# attach + clone the repo with push access first (add_repo Topupchips/shipwithpip access=push), then:
cd /home/claude/shipwithpip && bash tools/setup.sh
export KOKORO_DIR=$HOME/kokoro-multi-lang-v1_0
```
Read `state/state.json` (what is `next`) and the last 40 lines of `state/log.jsonl` (what was already posted).

## A. Simulation reel (`next == "sim"`)
Sim = `sim_rotation[sim_index % len]`. Work in `engine/goal2/`. Run ONE heavy process at a time (2 cores, 8 GB; two renders at once get OOM-killed).

1. **Run with a fresh seed** (not used before for this sim in the log; pick e.g. a random number 10 to 9999):
   `python3 autosim.py run SIM SEED /tmp/r.json`
   For `shortcut` and `drone` the first run records nothing; read the stats, choose the generations to show, then rerun the same seed with `--record 1,6,12,...` (identical results, deterministic).
2. **Judge the arc** from the printed stats. A good arc: starts near zero, a slow climb, a payoff, a twist that drops it (where the sim has one: hoops moves the hoop, maze blocks the path, dodge adds rocks, pole pushes, moon doubles gravity, drone halves rings, keepups adds a ball, hunt lets prey learn, password stalls on the last letter), and a second climb. If the arc is flat, never reaches the payoff, or the twist never triggers, **try another seed** (up to 6 seeds). If none works, switch to the next sim in the rotation.
3. **Write the spec** `/tmp/spec.json` (format in `engine/goal2/specstory.py`). 6 to 9 beats plus the end card. Use `engine/goal2/stories2.py`, `stories3.py` and the builders in `goal2.py` as the tone and wording reference for that sim: short spoken lines, a two-line ALL CAPS head, a short caption, an `sfx` burst on the payoff and on the twist. Only state what the stats show. Fixed settings of the sim (population 100, 13 letters, 6 rings, 30 seconds...) go in `"constants"`. Derived numbers you compute (e.g. "stuck for 32 generations" = 41 - 9) also go in `constants` and must be arithmetic you actually did on the stats.
   `post`: title, caption (what happened, with the real numbers, ends with what the algorithm is), tags (5 to 7, include shipwithpip), pinned (how the sim works), notes (seed and the per-generation numbers).
4. `python3 autosim.py check /tmp/r.json /tmp/spec.json` must print `CHECK OK`.
5. **Stills check**: `python3 goal2.py SIM /tmp/r.json /tmp/reel.mp4 --spec /tmp/spec.json --stills 2,15,30` and Read the PNGs. Look for: text overflowing, empty arena, overlapping labels, wrong counter. Fix and repeat.
6. **Render**: `nohup python3 -u goal2.py SIM /tmp/r.json /tmp/reel.mp4 --spec /tmp/spec.json > /tmp/render.log 2>&1 &` then poll the log until `done` (10 to 20 min). Any `pageerror` = fix and re-render.
7. **Verify the mp4**: ffprobe duration 35 to 65 s, has audio; extract 3 frames (early, middle, last third) with ffmpeg and Read them.

## B. Explainer reel (`next == "explainer"`)
Style: the dark minimal reels (see the `min_scripts/` examples, all beats `"bg": "dark"`): UI mockups, big numbers, word-by-word captions with a highlighted keyword, the prompt at the end.

1. **Pick a topic**: something that happened or was published in the last ~2 weeks about AI, vibe coding, AI coding agents, security of AI-built apps, AI costs, or developer tools, that a builder would want to know. Use WebSearch. Check the log so you never repeat a topic. Good shapes: "this broke because...", a surprising number, a real incident and the one fix.
2. **Verify** every fact with at least one reputable source (official post, docs, major outlet, original researcher). Keep the URLs.
3. **Write the script** `/tmp/x.json` in the exact format of `engine/min_scripts/*.json` (read 2 or 3 of them first, e.g. m9, m14, m2). 7 to 10 beats. Every beat `"bg": "dark"`. Each beat has `say`, `key` (1 word from `say` to highlight) and usually a `vis` (types used before: rows, code, number, statement, timeline, chat, stack, ring, inbox, tool, browser, phone, dialog, prompt). Second to last beat is `{"vis": {"type": "prompt"}}` with a short line, last beat is `{"say": "Follow Ship With Pip for one of these every day.", "nocap": true, "bg": "dark"}`. Fill `id`, `title`, `caption`, `hashtags` (5 to 7), `prompt` (a copy-paste prompt the viewer can give an AI to apply the lesson), `sources`.
4. **Contact sheet**: `cd engine && python3 minimal.py /tmp/x.json /tmp/sheet.png --model $KOKORO_DIR --stills`, Read it, fix layout problems.
5. **Render**: `python3 minimal.py /tmp/x.json /tmp/reel.mp4 --model $KOKORO_DIR` (about 5 min). Verify like A.7.

## C. Post (both kinds)
1. `url=$(bash tools/host_video.sh /tmp/reel.mp4)` (prints a raw.githubusercontent.com URL).
2. vidIQ `vidiq_video_upload` with `url` = that URL -> `videoUrl`.
3. Caption: sims use `post.caption` + a blank line + `Follow @shipwithpip for more AI from scratch.` + blank line + hashtags (`#` + each tag). Explainers use the script `caption` + blank line + hashtags. Max 2,200 chars, no em dashes.
4. vidIQ `vidiq_instagram_publish_reel` with `platformAccountId` `29510734025188123`, `videoUrl`, `caption`. No trial params, no AI label. Continue with `containerId` / `mcpJobId` exactly as the tool says until `PUBLISHED`. Do not retry a failed publish with a new container; log the error and stop.
5. Append one JSON line to `state/log.jsonl`: `{"date": ISO time, "kind": "sim"|"explainer", "sim": name, "seed": n, "topic": title, "permalink": ..., "status": "posted"|"failed: reason"}`. Update `state/state.json`: flip `next`; after a sim, `sim_index += 1` (only when posted). Commit and push to `main`:
   `git add state && git commit -m "log: <title>" && git push origin HEAD:main`

## D. Report
End the run with a 2 line summary: what was posted (title, permalink) or why nothing was.

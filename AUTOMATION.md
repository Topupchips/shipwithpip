# @shipwithpip auto-reels: runbook for each scheduled run

Every run makes **one** reel and posts it to Instagram **@shipwithpip** with no human review.
Ansh approved fully automatic posting, 3 reels a day (Ansh, Nov 13 2026: up from 2), in a repeating cycle of sim, sim, visual math reel (visual reels replaced the dark explainers on Nov 13 2026).
Because nobody reviews it, the checks below are the review. If any check fails and you cannot fix it honestly, **post nothing** and log the failure.

## Hard rules (never break)
1. **Honest numbers.** Every number in a sim reel comes from the run you just did (`autosim.py stats`). Never reuse numbers from an old run, never round in a way that changes meaning, never claim something the stats do not show. `autosim.py check` must print `CHECK OK`.
2. **Real facts for visual reels.** Every landmark, distance and claim in a visual reel is verified with WebSearch/WebFetch against a primary or reputable source found in this run. List the URLs in the script's `sources`. No invented stats, quotes, companies or incidents. If you cannot verify a claim, cut it.
3. **No em dashes** anywhere (script, captions, on-screen text, post). Use commas, colons or full stops.
4. Post only to account id `29510734025188123` (@shipwithpip). Never to @anshgotenks or @anshgoten.
5. One reel per run. Runs are at 02:07, 04:07 and 14:07 Halifax time; each must finish within 110 minutes of starting. If the log shows a reel was already posted within the last 90 minutes, stop.
6. Keep the video vertically centred, 1080x1920. Length 35 to 65 s.
7. Hooks: the first 2 seconds must be a surprising, specific line (a number, a reversal, a "then I..." twist). Ansh wants more random, unexpected hooks; capturing attention in the first 10 seconds is the priority.

## 0. Setup (every run, fresh container)
```bash
# attach + clone the repo with push access first (add_repo Topupchips/shipwithpip access=push), then:
cd /home/claude/shipwithpip && bash tools/setup.sh
export KOKORO_DIR=$HOME/kokoro-multi-lang-v1_0
```
Read `state/state.json`: the reel to make is `next = cycle[cycle_index % len(cycle)]` (cycle is sim, sim, visual) and the last 40 lines of `state/log.jsonl` (what was already posted).

## A. Simulation reel (`next` is "sim")
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

## B. Visual math reel (`next` is "visual")
Style: a dark "scale climb". Something changes by a fixed factor every step and a log-scale camera moves past real landmarks, with a big step counter, a live readout and word-by-word captions. Engine: `engine/scale.py`. Examples in `engine/scale_scripts/`: `v1_folds.json` (paper folds, length, doubling), `v2_penny.json` (penny stack, length, doubling), and in `queue/` until posted: `v3_halving.json` (shrinking to an atom), `v4_seconds.json` (time, x10, `counter_fmt` pow10), `v5_tripling.json` (people, x3), `v6_odds.json` (odds, x10).

**Queue first.** If `state.json` has a non-empty `visual_queue`, make `visual_queue[0]`: its script is already written and checked in `engine/scale_scripts/queue/<id>.json`. In this run: re-open every URL in its `sources` and confirm each sourced number is still stated there (a figure that changed or vanished means fix the script honestly, or skip to the next queued id and say why in the log); redo the arithmetic in its `facts` with Python; then run steps 5 to 7 below on it (voice check, contact sheet, render). After posting, `git mv` the file from `queue/` to `scale_scripts/` and remove the id from `visual_queue` in the same commit as the state. Only when the queue is empty, write a new script with steps 1 to 7.

**Variety rule (Ansh, Nov 14 2026: no more near-repeats).** A new visual reel must differ from each of the last 3 visual reels in the log in at least two of: (a) the factor (x2, x3, x10, halving...), (b) the dimension (`dim`: length, time, count, odds), (c) the ending (the final landmark or twist). Two doubling-to-the-universe reels have already run (v1 folds, v2 penny): do not make another reel that ends at the observable universe.

1. **Pick a topic** not already in the log (check `topic` of every `"kind": "visual"` line) and passing the variety rule. It must be a quantity that changes by a fixed factor per step in one of the engine's dimensions: `dim` "length" (metres, growing, or shrinking with `factor` below 1), "time" (seconds), "count" (people or things) or "odds" (1 in N). Hook must be a specific, surprising number in the first 2 seconds.
2. **Verify** every landmark distance/size and the starting size with WebSearch/WebFetch against a reputable source in this run (NASA, JPL, Britannica, official records, major outlets). Keep the URLs in `sources`.
3. **Compute** every step number with Python in this run: `step = ceil(log(landmark_m / start_m) / log(factor))` and the size at each shown step as `start_m * factor ** step`. Every number spoken or written must be one of these results or a sourced fact. For a shrinking reel the step is the first n where `start_m * factor ** n` is below the landmark. Write the arithmetic in your notes.
4. **Write the script** `/tmp/v.json` like `v1_folds.json`: `start_m`, `factor`, `unit_word` (counter label, e.g. FOLD, DOUBLING, DAY), 5 to 9 `landmarks` (`name` ALL CAPS, `m`, `value` short sourced text, `icon` one of mountain, station, moon, sun, star, galaxy, andromeda, universe, hair, cell, virus, dna, atom, nucleus, clock, pyramid, skull, dino, earth, bang, person, crowd, stadium, flag, dice, bolt, cards, ticket, coin). Optional keys: `dim`, `readout_word` (text after the readout, default "thick"), `counter_fmt` "pow10" (counter shows 10 to the n), `col_rgb` / `col_rgb_d` (column colours), 8 to 11 `beats` (`to` = step reached by the end of the beat, `say`, `key`; first beat has `hook` = 3 short ALL CAPS lines; optional `stamp` for a red stamp, e.g. a real record or reality check; optional `speak` when the voice must differ from the caption, e.g. years written in words). Last beat: `{"to": <same as before>, "nocap": true, "say": "Follow Ship With Pip for more."}`. Fill `id` (v2_..., v3_...), `title`, `caption`, `hashtags` (5 to 7, include shipwithpip), `sources`. End with a reality-check twist where a real sourced one exists.
5. **Voice check**: the Kokoro lexicon silently skips unknown words (plurals like "kilometers", names, digits). Run the stills command and `grep OOV` its output; fix every OOV word with `speak` or by adding it to `LEX_ADD` in `scale.py`. No OOV lines allowed.
6. **Contact sheet**: `cd engine && python3 scale.py /tmp/v.json /tmp/sheet.png --model $KOKORO_DIR --stills 2>&1 | grep -v "^$"`, Read the PNG: labels readable, landmarks not overlapping each other, counter and readout matching the script.
7. **Render**: `python3 scale.py /tmp/v.json /tmp/reel.mp4 --model $KOKORO_DIR` (about 3 to 5 min). Verify like A.7. Copy the final script into `engine/scale_scripts/` and commit it with the state.

## C. Post (both kinds)
1. `url=$(bash tools/host_video.sh /tmp/reel.mp4)` (prints a raw.githubusercontent.com URL).
2. vidIQ `vidiq_video_upload` with `url` = that URL -> `videoUrl`.
3. Caption: sims use `post.caption` + a blank line + `Follow @shipwithpip for more AI from scratch.` + blank line + hashtags (`#` + each tag). Visual reels use the script `caption` + blank line + `Follow @shipwithpip for more.` + blank line + hashtags. Max 2,200 chars, no em dashes.
4. vidIQ `vidiq_instagram_publish_reel` with `platformAccountId` `29510734025188123`, `videoUrl`, `caption`. No trial params, no AI label. Continue with `containerId` / `mcpJobId` exactly as the tool says until `PUBLISHED`. Do not retry a failed publish with a new container; log the error and stop.
5. Append one JSON line to `state/log.jsonl`: `{"date": ISO time, "kind": "sim"|"visual", "sim": name, "seed": n, "topic": title, "permalink": ..., "status": "posted"|"failed: reason"}`. Update `state/state.json` only when the reel was posted: `cycle_index += 1`, set `next` to the new `cycle[cycle_index % len(cycle)]` (for readability), and after a sim also `sim_index += 1`. A failed run leaves the state unchanged so the next run retries the same slot. Commit and push to `main`:
   `git add state engine/scale_scripts && git commit -m "log: <title>" && git push origin HEAD:main`

## D. Report
End the run with a 2 line summary: what was posted (title, permalink) or why nothing was.

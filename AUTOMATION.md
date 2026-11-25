# @shipwithpip runbook (short, active). Read only this file.

One run = one punch-style AI sim reel posted to Instagram @shipwithpip, no human review. 2 runs a day, 02:07 and 14:07 Halifax.
Plan from Ansh, Nov 25 2026 (audit: https://claude.ai/artifact/BrVQeB76XyjkXbiUfJqa7c): sims only; what-if and visual reels paused (`docs/AUTOMATION_archive.md` holds them and the full history; do not read it unless Ansh asks for a paused format or something here is unclear).
**Token and credit budget (Ansh, Nov 25 2026: "use credit and token usage efficiency to max"):** use `tools/sim.py` for every mechanical step; do not read engine source, do not cat big files, do not poll renders with sleeps; Read exactly one image (`/tmp/sheet.png`) unless it shows a problem; at most 4 seeds; only free vidIQ tools (upload, publish, owner insights; never outlier search, profile reels or generation tools); no subagents.

## Hard rules
1. Every number on screen, spoken or in the caption comes from this run's stats (`sim.py seeds` / `autosim.py check` must print CHECK OK). Derived numbers (93 - 3 = 90) go in spec `constants` and `post.notes`.
2. No em dashes anywhere. 3. Post only to platformAccountId `29510734025188123` (@shipwithpip). 4. One reel per run; stop if `plan` says `posted_within_90_min`. 5. If a check fails and cannot be fixed honestly, post nothing and run `python3 tools/sim.py fail SIM "reason"`.

## Steps
1. `cd /home/claude/shipwithpip && bash tools/setup.sh && python3 tools/sim.py plan` gives sim, series part, voice mode, used seeds, earlier topics, review_due. Never make a sim in `retired`.
2. If `review_due`: section Review below, then continue.
3. `python3 tools/sim.py seeds SIM 3` prints one arc line per fresh seed (stats + per-gen numbers). Pick an arc that starts low, climbs, pays off, has the twist (hoops moves the hoop, maze blocks, dodge adds rocks, pole pushes, moon doubles gravity, drone halves rings, keepups adds a ball, hunt lets prey learn, password stalls), and tells a different story from `earlier_topics`. None good: one more call with 3 new seeds, then switch to the next sim in rotation (set `sim_index`). `shortcut`/`drone` need `--record g1,g2,..` on build.
4. Write `/tmp/spec.json` (punch style, keys below) from the arc numbers only.
5. `python3 tools/sim.py build SIM SEED /tmp/spec.json [--kw ..] [--record ..]` (about 2 min) renders and prints `ok`, duration, issues; then Read `/tmp/sheet.png`: text not overflowing, arena not empty, counters match. Fix the spec and rebuild if not ok.
6. `python3 tools/sim.py prep /tmp/spec.json` hosts the video and prints the URL and exact caption.
7. vidIQ `vidiq_video_upload` (url) then `vidiq_instagram_publish_reel` (platformAccountId 29510734025188123, videoUrl, caption exactly as printed; no trial, no AI label). Continue with `containerId` (wait about 30 s between calls) until PUBLISHED. Never retry a failed publish with a new container.
8. `python3 tools/sim.py log /tmp/spec.json SEED PERMALINK "short note"` appends the log, advances state (sim, voice), commits and pushes.
9. Reply with 2 lines: title and permalink, or why nothing posted (plus review headline numbers if one ran).

## Punch spec (engine/goal2/specstory.py format)
```json
{"reel": "hoops", "style": "punch", "voice": "<plan.voice>", "series": "AI LEARNS HOOPS, PART 3",
 "constants": [100, 3, 90],
 "beats": [{"gen": 1, "head": ["100 AIs. 0 SHOTS.", "I MOVE THE HOOP."], "stat": "2 / 100",
            "say": "A hundred AIs learn this shot. Then I move the hoop.", "cap": "100 AIs. Never shot a ball. Gen 1: 2 in."},
           {"gen": 8, "head": ["100 AIs.", "1 HOOP."], "stat": "93 / 100", "stat_col": "GREEN", "say": "...", "cap": "Gen 8: 93/100. Solved.",
            "sfx": {"text": "93 / 100!", "col": "LIME", "burst": "GREEN", "t": 1.6}}],
 "end": ["Part 3.", "100 AIs.", "0 instructions."], "end_say": "<either/or question about this reel>? Comment.",
 "post": {"title": "...", "caption": "<second hook line>. <what happened with real numbers>. <what the algorithm is>.",
          "tags": ["ai", "machinelearning", "<sim word>", "shipwithpip"], "pinned": "<how the sim works>",
          "notes": "Seed N. Per-gen numbers. Derived arithmetic. Idea test: relatable/question/reaction/unrealistic = x of 4."}}
```
- 6 to 8 beats plus the end; reel 20 to 35 s (`beat_max` 3.6 default; lower it or drop a beat if longer).
- Beat 1 = gen 1 (the swarm, from frame 1); its `head` is the hook: specific number or twist, a stranger gets it in under 5 s, passes 2 of 4 (relatable, raises a question, provokes a reaction, seems unrealistic). Its `say` is the only voiced line in `hook` mode (under 12 words).
- `cap` 8 words or fewer, carries the story; `say` is still required on every beat (checked). No beat that only repeats a number; one `sfx` on the payoff and one on the twist.
- Series part N from `plan`; each part should be harder or different from the last (new twist outcome, bigger `--kw npop=` only if the arc still works, different seed). Caption tags 3 to 5. Never promise anything the pipeline will not do.

## Review (every 48 h, free tools only)
`vidiq_instagram_owner_insights` platformAccountId 29510734025188123: `mode: "reels"`, `limit: 25` (per-reel lifetime metrics) and `mode: "account"`, metrics `["views","profile_views"]`, `timeframe: "last_14_days"`. Count only reels 24+ h old; match to log by permalink; work it out in one short python call, not by hand.
- Scorecard per punch reel: skip rate under 40%, share rate over 2%, like rate over 4%, save rate over 3% (on target / below / critical when more than 50% off). Baseline Nov 17 to 24 (pre-punch): skip 32%, likes 1%, shares 0.08%, saves 0.03%; profile visits 1.9 per 1,000 views.
- Voice A/B: once each mode has 4+ punch reels, if one wins on both median skipRate and avgWatchedPct by 3+ points, set `voice_modes` to it alone.
- Rotation: move the two best sims (median views, skip) to the front; move a sim with 3+ punch reels all under 200 views to the end.
- Judge the punch style only after 7 to 10 punch reels. Never bring back paused formats without Ansh.
- Append `{"date","profile_visits_per_1k","scorecard_medians","voice","changes","why"}` to `state/decisions.jsonl`, set `last_review` in state.json, commit and push. Data unavailable: skip and say so.

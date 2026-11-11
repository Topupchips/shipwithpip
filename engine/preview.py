"""Fast layout check: fake beat timings (no voice), dump a contact sheet of each beat's mid and late frame."""
import json, sys, math
import numpy as np
from PIL import Image, ImageDraw
import explain as E

def sheet(path, out):
    s = json.load(open(path))
    ac = E.ACCENTS[s.get("accent", "orange")]
    t = 0.2; tl = []
    for b in s["beats"]:
        dur = len(b["say"].split()) / 2.7
        tl.append({"start": t, "end": t + dur, "beat": b, "env": np.full(int(dur * 30) + 2, 0.5)})
        t += dur + E.GAP
    total = t + E.END
    scenes = []
    for i, x in enumerate(tl):
        v = x["beat"].get("vis")
        if v:
            e = min(i + v.get("span", 1), len(tl)) - 1
            scenes.append({"vis": v, "start": x["start"] - 0.1, "anim_end": tl[e]["end"]})
    for k, sc in enumerate(scenes):
        sc["end"] = scenes[k + 1]["start"] if k + 1 < len(scenes) else total - E.END
    bg = E.make_bg(ac)
    frames = []
    times = [0.5] + [x["start"] + (x["end"] - x["start"]) * 0.85 for x in tl] + [total - 0.5]
    for tt in times:
        img = bg.convert("RGB"); d = ImageDraw.Draw(img, "RGBA")
        end = tt >= total - E.END
        seg = [x for x in tl if x["start"] - E.GAP <= tt <= x["end"] + E.GAP]
        seg = seg[-1] if seg else tl[0]
        if not end:
            cur = [sc for sc in scenes if sc["start"] <= tt][-1]
            ts = tt - cur["start"]; p = E.clamp(ts / max(cur["anim_end"] - cur["start"], .5))
            E.SCENES[cur["vis"]["type"]](d, img, cur["vis"], p, ts, ac, {})
            E.draw_title(d, img, s["hook"], tt, ac)
            E.draw_caption(d, seg, min(tt, seg["end"]), ac)
        else:
            E.draw_end(d, 1.0, ac)
        E.draw_pip(img, tt, None if end else seg, end)
        E.draw_progress(d, tt / total, ac)
        frames.append(img.resize((270, 480)))
    W = 270 * len(frames)
    sh = Image.new("RGB", (W, 480))
    for i, f in enumerate(frames): sh.paste(f, (i * 270, 0))
    sh.save(out)

if __name__ == "__main__":
    for p in sys.argv[1:-1]:
        name = p.split("/")[-1][:-5]
        sheet(p, f"{sys.argv[-1]}/{name}.png")
        print(name)

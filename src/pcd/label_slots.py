import argparse
import csv

import cv2
import numpy as np

from slot_utils import (FRAMES_DIR, LABELS_CSV, fit_scale, frame_sort_key, load_slots, outside_fraction,
                        parse_frame, put_text, show)

WIN = "label slots"
COLORS = {0: (0, 200, 0), 1: (0, 0, 230), -1: (150, 150, 150)}
FIELDS = ["image", "date", "time", "who", "slot_id", "label"]


def read_labels():
    labels = {}
    if LABELS_CSV.exists():
        with open(LABELS_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                labels.setdefault(r["image"], {})[int(r["slot_id"])] = int(r["label"])
    return labels


def write_labels(labels):
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = LABELS_CSV.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for img in sorted(labels):
            p = parse_frame(img)
            for sid in sorted(labels[img]):
                w.writerow([img, p["date"], p["time"], p["who"], sid, labels[img][sid]])
    tmp.replace(LABELS_CSV)


def main():
    ap = argparse.ArgumentParser(
        description="Label every box: click = empty/occupied, right-click = unclear, space = next, b = back, "
                    "e/a = all empty/occupied, q = save and quit.")
    ap.add_argument("--who", help="only frames from this person's videos, e.g. keiza")
    ap.add_argument("--date", help="only frames of this day, e.g. 20261104")
    ap.add_argument("--max-w", type=int, default=1400)
    ap.add_argument("--max-h", type=int, default=820)
    a = ap.parse_args()

    frames = sorted(FRAMES_DIR.glob("*.jpg"), key=frame_sort_key)
    no_boxes = sorted({parse_frame(f.name)["video"] for f in frames if not load_slots(parse_frame(f.name)["video"])})
    if no_boxes:
        print("no boxes yet for: " + ", ".join(no_boxes) + "  (run draw_slots.py <video id>) - skipping them")
    frames = [f for f in frames if parse_frame(f.name)["video"] not in no_boxes]
    if a.who:
        frames = [f for f in frames if parse_frame(f.name)["who"] == a.who.lower()]
    if a.date:
        frames = [f for f in frames if parse_frame(f.name)["date"] == a.date]
    if not frames:
        raise SystemExit(f"no frames in {FRAMES_DIR}")

    labels = read_labels()
    i = next((k for k, f in enumerate(frames) if f.name not in labels), len(frames) - 1)
    state = {}

    def load(k):
        name = frames[k].name
        img = cv2.imread(str(frames[k]))
        slots = load_slots(parse_frame(name)["video"])
        # boxes that fall outside this aligned picture can't be judged -> unclear, locked
        locked = {sid for sid, q in enumerate(slots) if outside_fraction(img, q) > 0.2}
        same_video = k > 0 and parse_frame(frames[k - 1].name)["video"] == parse_frame(name)["video"]
        if name in labels:
            prev, note = labels[name], ""
        elif same_video and frames[k - 1].name in labels:
            prev, note = labels[frames[k - 1].name], "copied from previous frame: click only what changed"
        else:
            prev, note = {}, "NEW VIDEO: check every box"
        cur = {sid: (-1 if sid in locked else prev.get(sid, 1)) for sid in range(len(slots))}
        state.update(i=k, img=img, cur=cur, locked=locked, note=note, slots=slots,
                     scale=fit_scale(img, a.max_w, a.max_h))

    def redraw():
        img, cur, slots = state["img"], state["cur"], state["slots"]
        fill = img.copy()
        for sid, q in enumerate(slots):
            cv2.fillPoly(fill, [np.int32(q)], COLORS[cur[sid]])
        vis = cv2.addWeighted(fill, 0.35, img, 0.65, 0)
        for sid, q in enumerate(slots):
            cv2.polylines(vis, [np.int32(q)], True, COLORS[cur[sid]], 2)
            c = q.mean(axis=0).astype(int)
            put_text(vis, str(sid), (c[0] - 8, c[1] + 6), 0.5)
        free = sum(v == 0 for v in cur.values())
        valid = sum(v != -1 for v in cur.values())
        done = sum(f.name in labels for f in frames)
        put_text(vis, f"{frames[state['i']].name}  [{state['i'] + 1}/{len(frames)}, labeled {done}]"
                      f"   free {free}/{valid}", (10, 30), 0.8)
        put_text(vis, "[click] empty/occupied  [right] unclear  [space] next  [b] back  [e/a] all empty/occ  [q] quit",
                 (10, 60), 0.6)
        if state["note"]:
            put_text(vis, state["note"], (10, 90), 0.7, (0, 255, 255))
        show(WIN, vis, state["scale"])

    def on_mouse(ev, x, y, flags, _):
        if ev not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        x, y = x / state["scale"], y / state["scale"]
        for sid, q in enumerate(state["slots"]):
            if cv2.pointPolygonTest(q, (x, y), False) >= 0:
                if sid in state["locked"]:
                    break
                cur = state["cur"]
                if ev == cv2.EVENT_LBUTTONDOWN:
                    cur[sid] = 0 if cur[sid] == 1 else 1
                else:
                    cur[sid] = 1 if cur[sid] == -1 else -1
                break
        redraw()

    def commit():
        labels[frames[state["i"]].name] = dict(state["cur"])
        write_labels(labels)

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    load(i)
    redraw()
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k in (ord(" "), ord("n"), 13):
            commit()
            if state["i"] + 1 < len(frames):
                load(state["i"] + 1)
            else:
                state["note"] = "last frame done - press q"
            redraw()
        elif k == ord("b") and state["i"] > 0:
            commit()
            load(state["i"] - 1)
            redraw()
        elif k in (ord("e"), ord("a")):
            v = 0 if k == ord("e") else 1
            state["cur"] = {sid: (-1 if sid in state["locked"] else v) for sid in range(len(state["slots"]))}
            redraw()
        elif k in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            commit()
            break
    cv2.destroyAllWindows()
    print(f"labels saved to {LABELS_CSV}")


if __name__ == "__main__":
    main()

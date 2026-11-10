import argparse

import cv2
import numpy as np

from slot_utils import (REF_DIR, fit_scale, load_slots, order_quad, put_text, ref_path, save_slots, show,
                        slots_path)

WIN = "draw slots"
HELP = ("[click x4] add  [right] delete  [u] undo  [i/j/k/l] move all  [ [ / ] ] smaller/bigger all  "
        "[s] save  [q] quit")


def main():
    ap = argparse.ArgumentParser(
        description="Draw the parking-space boxes on one video's reference picture. "
                    "Use --from to start from another video's boxes and shift them into place.")
    ap.add_argument("video", nargs="?", help="video id, e.g. 20261104_evening_keiza (leave empty to list them)")
    ap.add_argument("--from", dest="src", help="copy the boxes of this video id as a starting point")
    ap.add_argument("--max-w", type=int, default=1400)
    ap.add_argument("--max-h", type=int, default=820)
    a = ap.parse_args()

    vids = sorted(p.stem for p in REF_DIR.glob("*.jpg"))
    if not a.video:
        print("videos (* = boxes already drawn):")
        for v in vids:
            print(f"  {'*' if slots_path(v).exists() else ' '} {v}")
        return
    img = cv2.imread(str(ref_path(a.video)))
    if img is None:
        raise SystemExit(f"no reference picture for {a.video}; run extract_frames.py first. Known: {', '.join(vids)}")
    scale = fit_scale(img, a.max_w, a.max_h)
    slots = load_slots(a.video) or (load_slots(a.src) if a.src else [])
    current = []

    def save():
        save_slots(a.video, slots)
        print(f"saved {len(slots)} boxes -> {slots_path(a.video)}")

    def redraw():
        vis = img.copy()
        for i, q in enumerate(slots):
            cv2.polylines(vis, [np.int32(q)], True, (0, 255, 255), 2)
            c = np.int32(q).mean(axis=0).astype(int)
            put_text(vis, str(i), (c[0] - 10, c[1] + 8))
        for p in current:
            cv2.circle(vis, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
        if len(current) > 1:
            cv2.polylines(vis, [np.int32(current)], False, (0, 0, 255), 2)
        put_text(vis, f"{a.video}   boxes: {len(slots)}", (10, 30), 0.8)
        put_text(vis, HELP, (10, 60), 0.55)
        show(WIN, vis, scale)

    def on_mouse(ev, x, y, flags, _):
        x, y = x / scale, y / scale
        if ev == cv2.EVENT_LBUTTONDOWN:
            current.append([x, y])
            if len(current) == 4:
                slots.append(order_quad(current))
                current.clear()
            redraw()
        elif ev == cv2.EVENT_RBUTTONDOWN:
            for i, q in enumerate(slots):
                if cv2.pointPolygonTest(np.float32(q), (x, y), False) >= 0:
                    slots.pop(i)
                    break
            redraw()

    moves = {ord("j"): (-5, 0), ord("l"): (5, 0), ord("i"): (0, -5), ord("k"): (0, 5)}
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k in moves and slots:
            d = np.float32(moves[k])
            slots[:] = [q + d for q in slots]
            redraw()
        elif k in (ord("["), ord("]")) and slots:
            c = np.mean([q.mean(axis=0) for q in slots], axis=0)
            f = 0.98 if k == ord("[") else 1.02
            slots[:] = [(q - c) * f + c for q in slots]
            redraw()
        elif k == ord("u"):
            if current:
                current.pop()
            elif slots:
                slots.pop()
            redraw()
        elif k == ord("s"):
            save()
        elif k in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            save()
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

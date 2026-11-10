import argparse
import csv

import cv2
import numpy as np

from align import Aligner
from slot_utils import (CHECK_DIR, FRAMES_DIR, RAW_DIR, VIDEO_EXT, VIDEOS_CSV, frame_name, guess_video_info,
                        load_slots, ref_path, to_frame_size, video_id, warp_slot)

FIELDS = ["file", "date", "time", "weather", "who", "note"]
INDEX = FRAMES_DIR / "frames_index.csv"
INDEX_FIELDS = ["frame", "video", "video_id", "date", "time", "weather", "who", "t_sec", "reason", "inliers"]


def sync_video_table():
    """Add every video in data/raw to videos.csv (guessing date/time/who from the name)."""
    rows = {}
    if VIDEOS_CSV.exists():
        with open(VIDEOS_CSV, newline="", encoding="utf-8") as f:
            rows = {r["file"]: r for r in csv.DictReader(f)}
    found = sorted(p for p in RAW_DIR.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    for p in found:
        rel = p.relative_to(RAW_DIR).as_posix()
        if rel not in rows:
            rows[rel] = {"file": rel, **guess_video_info(p), "note": "auto-guess, please check"}
            r = rows[rel]
            print(f"  new video: {rel} -> date={r['date']} time={r['time']} weather={r['weather'] or '?'} who={r['who']}")
    with open(VIDEOS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))
    return [rows[k] for k in sorted(rows)]


def slot_signature(img, slots):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.stack([warp_slot(g, q, 24).astype(np.float32) for q in slots])


def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def extract(row, aligner, every, change_thr):
    """Each frame is aligned to this video's reference (its sharpest frame in the first seconds)."""
    vid = video_id(row["date"], row["time"], row["who"])
    cap = cv2.VideoCapture(str(RAW_DIR / row["file"]))
    if not cap.isOpened():
        print(f"  !! cannot open {row['file']} - skipped")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step, check = max(1, round(every * fps)), max(1, round(fps))
    slots = load_slots(vid) if change_thr > 0 else []

    # reference = sharpest of the frames in the first 3 seconds (avoids a blurry start)
    ref, best = None, -1
    for idx in range(min(n, int(3 * fps))):
        if not cap.grab():
            break
        if idx % max(1, int(fps / 4)) == 0:
            ok, f = cap.retrieve()
            if ok and (s := sharpness(f)) > best:
                ref, best = to_frame_size(f), s
    if ref is None:
        print(f"  !! {row['file']}: no readable frames - skipped")
        return []
    if not ref_path(vid).exists():
        ref_path(vid).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ref_path(vid)), ref, [cv2.IMWRITE_JPEG_QUALITY, 95])
    ref = cv2.imread(str(ref_path(vid)))  # keep an existing reference (boxes may already be drawn on it)
    ref_feat = aligner.features(ref)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    last_sig, failed, n_change = None, 0, 0
    for idx in range(n):
        if not cap.grab():
            break
        regular = idx % step == 0
        if not regular and not (slots and idx % check == 0):
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        frame = to_frame_size(frame)
        H, inl = aligner.homography(aligner.features(frame), ref_feat)
        if H is None:
            failed += regular
            continue
        aligned = aligner.warp(frame, H)

        keep, reason = regular, "every"
        if slots:
            sig = slot_signature(aligned, slots)
            if last_sig is not None and not regular:
                changed = np.abs(sig - last_sig).mean(axis=(1, 2)) > change_thr
                if 0 < changed.sum() <= max(1, len(slots) // 3):
                    keep, reason = True, "change"
                    n_change += 1
            if keep:
                last_sig = sig
        if not keep:
            continue
        sec = int(round(idx / fps))
        name = frame_name(row["date"], row["time"], row["who"], sec)
        cv2.imwrite(str(FRAMES_DIR / name), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append([name, row["file"], vid, row["date"], row["time"], row["weather"], row["who"], sec, reason, inl])
        if regular:
            checks.append((name, frame, aligned))
    cap.release()

    if checks:
        # before/after picture: the frame that moved most, raw vs aligned, each blended with the reference
        CHECK_DIR.mkdir(parents=True, exist_ok=True)
        name, raw, aligned = max(checks, key=lambda c: np.mean(cv2.absdiff(c[1], ref)))
        small = lambda im: cv2.resize(im, (640, 360))
        before = small(cv2.addWeighted(ref, 0.5, raw, 0.5, 0))
        after = small(cv2.addWeighted(ref, 0.5, aligned, 0.5, 0))
        for im, t in ((before, "BEFORE alignment (doubled)"), (after, "AFTER alignment (sharp)")):
            cv2.putText(im, t, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(im, t, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(str(CHECK_DIR / f"{vid}.jpg"), np.hstack([before, after]))
    msg = f"  {row['file']}: {n / fps:.0f}s -> {len(rows)} frames"
    if slots:
        msg += f" ({n_change} on space change)"
    if failed:
        msg += f", {failed} too blurry to align (skipped)"
    print(msg)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Video -> aligned frames. Reads data/raw/videos.csv.")
    ap.add_argument("--every", type=float, default=15, help="seconds between frames (default 15)")
    ap.add_argument("--changes", type=float, default=0, metavar="THR",
                    help="also keep a frame when a space changes (needs boxes; try 10-15). 0 = off")
    ap.add_argument("--only", help="only process videos whose file name contains this text")
    ap.add_argument("--redo", action="store_true", help="re-extract videos that already have frames")
    a = ap.parse_args()

    videos = sync_video_table()
    if not videos:
        raise SystemExit(f"no videos in {RAW_DIR}")
    aligner = Aligner()

    old = {}
    if INDEX.exists():
        with open(INDEX, newline="", encoding="utf-8") as f:
            old = {r["frame"]: r for r in csv.DictReader(f) if "video_id" in r}
    done = {r["video"] for r in old.values()}
    for row in videos:
        if a.only and a.only.lower() not in row["file"].lower():
            continue
        missing = [k for k in ("date", "time", "who") if not row[k]]
        if missing:
            print(f"  !! {row['file']}: fill in {', '.join(missing)} in videos.csv first - skipped")
            continue
        if row["file"] in done and not a.redo:
            print(f"  {row['file']}: already extracted (use --redo to do it again)")
            continue
        old = {k: r for k, r in old.items() if r["video"] != row["file"]}
        for p in FRAMES_DIR.glob(f"{video_id(row['date'], row['time'], row['who'])}_t*.jpg"):
            p.unlink()
        for r in extract(row, aligner, a.every, a.changes):
            old[r[0]] = dict(zip(INDEX_FIELDS, r))

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, INDEX_FIELDS)
        w.writeheader()
        w.writerows(old[k] for k in sorted(old))
    print(f"{len(old)} frames listed in {INDEX}")
    print(f"before/after alignment pictures: {CHECK_DIR}")


if __name__ == "__main__":
    main()

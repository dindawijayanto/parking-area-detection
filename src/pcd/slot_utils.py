import json
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
VIDEOS_CSV = RAW_DIR / "videos.csv"          # one row per video: file, date, time, weather, who
FRAMES_DIR = RAW_DIR / "frames"              # aligned frames
CHECK_DIR = RAW_DIR / "align_check"          # images to check the alignment by eye
ANN_DIR = ROOT / "data" / "annotations"
REF_DIR = ANN_DIR / "references"             # one reference picture per video (its first sharp frame)
SLOTS_DIR = ANN_DIR / "slots"                # box corners per video, drawn on that video's reference
LABELS_CSV = ANN_DIR / "labels.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

FRAME_W, FRAME_H = 1920, 1080
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".3gp"}
TIMES = ["morning", "midday", "afternoon", "evening"]
NAMES = ["dinda", "keiza", "tobi"]

# Frames: 20261104_afternoon_dinda_t015.jpg
FRAME_RE = re.compile(r"^(\d{8})_([a-z]+)_([a-z]+)_t(\d+)", re.IGNORECASE)


def guess_video_info(path):
    """Best guess of date / time / weather / who from a messy file name. Check the result in videos.csv."""
    s = Path(path).stem.lower()
    m = re.search(r"(\d{1,2})[-_.](\d{1,2})[-_.](\d{4})", s)           # 04-11-2026
    m2 = re.search(r"(\d{4})[-_.](\d{1,2})[-_.](\d{1,2})", s)          # 2026-11-04
    date = (f"{m.group(3)}{int(m.group(2)):02d}{int(m.group(1)):02d}" if m else
            f"{m2.group(1)}{int(m2.group(2)):02d}{int(m2.group(3)):02d}" if m2 else "")
    time = next((t for t in TIMES if t in s), "")
    who = next((n for n in NAMES if n in s), "")
    weather = "rain" if ("rain" in s or "hujan" in s) else ""
    return {"date": date, "time": time, "weather": weather, "who": who}


def frame_name(date, time, who, sec):
    return f"{date}_{time}_{who}_t{sec:03d}.jpg"


def parse_frame(name):
    m = FRAME_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"'{name}' is not a frame name like 20261104_afternoon_dinda_t015.jpg")
    date, time, who, sec = m.groups()
    return {"date": date, "time": time.lower(), "who": who.lower(), "sec": int(sec),
            "video": f"{date}_{time.lower()}_{who.lower()}"}


def frame_sort_key(path):
    p = parse_frame(Path(path).name)
    t = TIMES.index(p["time"]) if p["time"] in TIMES else 9
    return (p["date"], t, p["who"], p["sec"])


def video_id(date, time, who):
    return f"{date}_{time}_{who}"


def ref_path(vid):
    return REF_DIR / f"{vid}.jpg"


def slots_path(vid):
    return SLOTS_DIR / f"{vid}.json"


def load_slots(vid):
    p = slots_path(vid)
    if not p.exists():
        return []
    return [np.float32(s["points"]) for s in json.loads(p.read_text(encoding="utf-8"))["slots"]]


def save_slots(vid, slots):
    SLOTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {"video": vid, "reference_image": ref_path(vid).name, "image_size": [FRAME_W, FRAME_H],
            "slots": [{"id": i, "points": np.asarray(s).round(1).tolist()} for i, s in enumerate(slots)]}
    slots_path(vid).write_text(json.dumps(data, indent=2), encoding="utf-8")


def to_frame_size(img):
    if img.shape[1] != FRAME_W or img.shape[0] != FRAME_H:
        img = cv2.resize(img, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    return img


def read_frame_at(path, sec):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * (cap.get(cv2.CAP_PROP_FPS) or 30)))
    ok, f = cap.read()
    cap.release()
    return to_frame_size(f) if ok else None


def order_quad(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def warp_slot(img, quad, size):
    """Perspective-correct one slot quad into a size x size crop."""
    dst = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    M = cv2.getPerspectiveTransform(np.float32(quad), dst)
    return cv2.warpPerspective(img, M, (size, size))


def outside_fraction(img, quad):
    """Share of a slot that is black because it fell outside the aligned picture."""
    crop = warp_slot(img, quad, 32)
    return float(np.mean(crop.max(axis=2) == 0))


def fit_scale(img, max_w, max_h):
    h, w = img.shape[:2]
    return min(1.0, max_w / w, max_h / h)


def show(win, img, scale):
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow(win, img)


def put_text(img, text, org, scale=0.7, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

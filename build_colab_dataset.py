import os
import csv
import zipfile
from pathlib import Path
import cv2
import numpy as np

DL_DIR = Path(os.path.expandvars(r"%USERPROFILE%\Downloads"))
DS_ROOT = Path("masking-parkir/public/dataset")
PRE_DIR = DS_ROOT / "preloaded-masks"
OUT_DIR = Path("colab_export")
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "images").mkdir(exist_ok=True)
(OUT_DIR / "masks").mkdir(exist_ok=True)

# 13 completed videos with exact metadata & anti-leakage day split (same test days 2026-11-04 & 2026-11-12 as Dinda's repo!)
VIDEO_META = {
    "01-03-11-2026-afternoon-top-keiza":      ("2026-11-03", "afternoon", "sunny",  "keiza", "train"),
    "02-03-11-2026-morning-top-keiza":        ("2026-11-03", "morning",   "sunny",  "keiza", "train"),
    "03-04-11-2026-afternoon-top-dinda":      ("2026-11-04", "afternoon", "sunny",  "dinda", "test"),
    "04-04-11-2026-evening-top-keiza":        ("2026-11-04", "evening",   "dusk",   "keiza", "test"),
    "06-06-11-2026-afternoon-top-keiza":      ("2026-11-06", "afternoon", "sunny",  "keiza", "train"),
    "07-06-11-2026-afternoonrain-top-keiza":  ("2026-11-06", "afternoon", "rain",   "keiza", "train"),
    "08-10-11-2026-afternoon-rainy-top-tobi": ("2026-11-10", "afternoon", "rain",   "tobi",  "val"),
    "09-12-11-2026-morning-top-keiza":        ("2026-11-12", "morning",   "sunny",  "keiza", "test"),
    "11-9-11-2026-afternoon-rainy-top-tobi":  ("2026-11-09", "afternoon", "rain",   "tobi",  "train"),
    "13-9-11-2026-morning-top-tobi":          ("2026-11-09", "morning",   "cloudy", "tobi",  "train"),
    "14-vid-20261117-141601":                 ("2026-11-17", "afternoon", "cloudy", "user",  "train"),
    "15-vid-20261117-154826":                 ("2026-11-17", "afternoon", "cloudy", "user",  "train"),
    "16-vid-20261118-082400":                 ("2026-11-18", "morning",   "sunny",  "user",  "val"),
}

# Exclude blurry 15-vid 00-00
EXCLUDE = {("15-vid-20261117-154826", "00-00")}

# Collect newest mask per (rec, ts)
sources = {}
def register(p: Path):
    name = p.name.split(" (")[0]
    if not name.endswith(".png"):
        name += ".png"
    if not name.endswith("_crop_mask.png"):
        return
    stem = name[:-len("_crop_mask.png")]
    if "_" not in stem:
        return
    rec, ts = stem.rsplit("_", 1)
    if rec not in VIDEO_META or (rec, ts) in EXCLUDE:
        return
    mt = p.stat().st_mtime
    key = (rec, ts)
    if key not in sources or mt > sources[key][0]:
        sources[key] = (mt, p)

for p in PRE_DIR.glob("*_crop_mask.png"):
    register(p)
for hdir in Path(".").glob("hasil-*"):
    for p in hdir.glob("*_crop_mask.png"):
        register(p)
for p in DL_DIR.glob("*_crop_mask*.png"):
    register(p)

MAX_SIDE = 960
rows = []
split_counts = {"train": 0, "val": 0, "test": 0}

for (rec, ts), (_, mpath) in sorted(sources.items()):
    img_path = DS_ROOT / rec / f"{rec}_{ts}_crop.jpg"
    if not img_path.exists():
        continue
    im = cv2.imread(str(img_path))
    m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
    if im is None or m is None:
        continue
    if m.shape[:2] != im.shape[:2]:
        m = cv2.resize(m, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Normalize any 254 -> 255 just in case
    m = np.where((m == 0) | (m == 1), m, 255).astype(np.uint8)

    h, w = im.shape[:2]
    if max(h, w) > MAX_SIDE:
        sc = MAX_SIDE / float(max(h, w))
        nw, nh = int(round(w * sc)), int(round(h * sc))
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
    h, w = im.shape[:2]

    fid = f"{rec}_{ts}"
    im_rel = f"images/{fid}.jpg"
    mk_rel = f"masks/{fid}.png"
    cv2.imwrite(str(OUT_DIR / im_rel), im, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    cv2.imwrite(str(OUT_DIR / mk_rel), m)

    date, tod, weather, cam, split = VIDEO_META[rec]
    split_counts[split] += 1
    rows.append({
        "frame_id": fid,
        "recording_id": rec,
        "timestamp": ts,
        "date": date,
        "time_of_day": tod,
        "weather": weather,
        "cameraman": cam,
        "split": split,
        "image_path": im_rel,
        "mask_path": mk_rel,
        "width": w,
        "height": h,
        "floor_px": int((m == 0).sum()),
        "motor_px": int((m == 1).sum()),
        "ignore_px": int((m == 255).sum()),
    })

manifest_path = OUT_DIR / "manifest.csv"
with manifest_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

zip_path = Path("colab_dataset_111.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(manifest_path, "manifest.csv")
    for r in rows:
        zf.write(OUT_DIR / r["image_path"], r["image_path"])
        zf.write(OUT_DIR / r["mask_path"], r["mask_path"])

print(f"Built {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB) | total={len(rows)} frames | splits={split_counts}")

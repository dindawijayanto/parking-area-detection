import argparse
import csv
import pickle

import cv2
import numpy as np
import pandas as pd

from slot_utils import FRAMES_DIR, LABELS_CSV, PROCESSED_DIR, load_slots, parse_frame, warp_slot

OUT = PROCESSED_DIR / "raw"


def main():
    ap = argparse.ArgumentParser(description="Cut every labeled box out of the frames and package dataset.pkl.")
    ap.add_argument("--size", type=int, default=224, help="PNG crop size (default 224, for DL)")
    ap.add_argument("--pkl-size", type=int, default=64, help="crop size stored in dataset.pkl (default 64, for ML)")
    a = ap.parse_args()

    slot_cache = {}
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if int(r["label"]) != -1]  # -1 = unclear, left out
    weather = {}
    index = FRAMES_DIR / "frames_index.csv"
    if index.exists():
        with open(index, newline="", encoding="utf-8") as f:
            weather = {r["frame"]: r["weather"] for r in csv.DictReader(f)}

    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    img_name, img = None, None
    meta, small = [], []
    for r in rows:
        if r["image"] != img_name:
            img_name, img = r["image"], cv2.imread(str(FRAMES_DIR / r["image"]))
        if img is None:
            print(f"  !! missing frame {r['image']}")
            continue
        sid = int(r["slot_id"])
        vid = parse_frame(r["image"])["video"]
        if vid not in slot_cache:
            slot_cache[vid] = load_slots(vid)
        crop = warp_slot(img, slot_cache[vid][sid], a.size)
        name = f"{r['image'].rsplit('.', 1)[0]}_s{sid:02d}.png"
        cv2.imwrite(str(OUT / "crops" / name), crop)
        small.append(cv2.resize(crop, (a.pkl_size, a.pkl_size), interpolation=cv2.INTER_AREA))
        meta.append({"crop": name, "image": r["image"], "date": r["date"], "time": r["time"],
                     "weather": weather.get(r["image"], ""), "who": r["who"], "slot_id": sid,
                     "label": int(r["label"])})

    df = pd.DataFrame(meta)
    df.to_csv(OUT / "crops.csv", index=False)
    with open(PROCESSED_DIR / "dataset.pkl", "wb") as f:
        pickle.dump({"X": np.stack(small) if small else np.empty((0, a.pkl_size, a.pkl_size, 3), np.uint8),
                     "y": df["label"].to_numpy() if len(df) else np.empty(0, int),
                     "meta": df,
                     "info": "X = BGR uint8 box crops (aligned within its video + perspective-corrected, no other PCD); "
                             "y: 0 empty, 1 occupied; split train/test by meta['date'], never randomly"}, f)
    n_occ = int(df["label"].sum()) if len(df) else 0
    print(f"{len(df)} crops (empty {len(df) - n_occ}, occupied {n_occ})")
    print(f"  PNGs -> {OUT / 'crops'}  ({a.size}x{a.size})")
    print(f"  .pkl -> {PROCESSED_DIR / 'dataset.pkl'}  ({a.pkl_size}x{a.pkl_size})")


if __name__ == "__main__":
    main()

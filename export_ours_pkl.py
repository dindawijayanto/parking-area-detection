import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "pcd"))
sys.path.insert(0, str(ROOT / "src" / "ml"))
import parking as pk
import pipeline as P

from train_rf_vs_xgb_v2 import BEV_MAP, compute_crop_to_bev, count_free_spaces, prepare_bev_geometries

DATA_DIR = ROOT / "colab_export"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEG_W, SEG_H = 960, 540
CLASSES = {0: "floor", 1: "motorcycle", 255: "ignore"}


def ts_to_sec(ts: str) -> int:
    m, s = ts.split("-")
    return int(m) * 60 + int(s)


def main():
    df = pd.read_csv(DATA_DIR / "manifest.csv")
    print(f"Loading {len(df)} frames from {DATA_DIR}...", flush=True)

    best = P.load_best()
    assert best is not None, "Missing results/pcd_eval/best_config.json"
    cfg, seg, t, pcd_name = best["pcd"], best["seg"], best["t"], best["variant"]

    geo_bev, sift, bf = prepare_bev_geometries()
    geo_dinda = {}
    for rec, vid in BEV_MAP.items():
        roi_path = P.ROI_DIR / f"{vid}.json"
        if roi_path.exists():
            geo_dinda[rec] = pk.prepare_video(pk.load_roi(roi_path))

    imgs_bgr = []
    masks_u8 = []
    meta_rows = []
    train_lab_px = []

    for _, row in df.iterrows():
        im_orig = cv2.imread(str(DATA_DIR / row["image_path"]))
        mk_orig = cv2.imread(str(DATA_DIR / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = im_orig.shape[:2]

        # Free spaces computed on original crop aligned via SIFT+RANSAC when ROI json exists
        rec = row["recording_id"]
        free_spaces = np.nan
        if rec in geo_bev:
            H_bev, _ = compute_crop_to_bev(im_orig, geo_bev[rec], sift, bf)
            free_spaces = int(count_free_spaces(mk_orig, H_bev, geo_bev[rec])[0])

        im_960 = cv2.resize(im_orig, (SEG_W, SEG_H), interpolation=cv2.INTER_AREA)
        mk_960 = cv2.resize(mk_orig, (SEG_W, SEG_H), interpolation=cv2.INTER_NEAREST)
        roi_960 = (mk_960 != 255).astype(np.uint8)

        imgs_bgr.append(im_960)
        masks_u8.append(mk_960)

        if row["split"] == "train":
            im_gw = P.shades_of_gray(im_960, roi_960, cfg.get("p", 6))[0] if cfg.get("gray_world") else im_960
            train_lab_px.append(cv2.cvtColor(im_gw, cv2.COLOR_BGR2LAB)[roi_960 > 0])

        roi_px = max(int(np.sum(mk_960 != 255)), 1)
        floor_pct = 100.0 * float(np.sum(mk_960 == 0)) / roi_px
        meta_rows.append({
            "frame": f"{row['frame_id']}.jpg",
            "video_id": rec,
            "date": str(row["date"]).replace("-", ""),
            "time": row["time_of_day"],
            "weather": row["weather"],
            "who": row["cameraman"],
            "t_sec": ts_to_sec(str(row["timestamp"])),
            "split": row["split"],
            "floor % of ROI": floor_pct,
            "free spaces": free_spaces,
            "orig_w": orig_w,
            "orig_h": orig_h,
        })

    X_seg = np.stack(imgs_bgr, axis=0)
    Y = np.stack(masks_u8, axis=0)
    meta = pd.DataFrame(meta_rows)
    print(f"Stacked X_seg={X_seg.shape} {X_seg.dtype}, Y={Y.shape} {Y.dtype}", flush=True)
    print(f"Splits: {meta.groupby('split').size().to_dict()}", flush=True)

    # Save 1: dataset_ours111_seg.pkl (raw BGR 960x540, identical schema to Dinda's dataset_seg.pkl)
    seg_path = OUT_DIR / "dataset_ours111_seg.pkl"
    seg_payload = {
        "X": X_seg,
        "Y": Y,
        "meta": meta,
        "geo": geo_dinda,
        "fit": pk.FIT,
        "classes": CLASSES,
        "info": (
            "Ours 111-frame crop dataset resized to 960x540 BGR uint8 (no PCD), compatible with "
            "Dinda's dataset_seg.pkl schema. Y: 0=floor, 1=motorcycle, 255=ignore. Split strictly by day."
        ),
    }
    with open(seg_path, "wb") as f:
        pickle.dump(seg_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] Saved {seg_path} ({seg_path.stat().st_size / 1e6:.2f} MB)", flush=True)

    # Save 2: dataset_ours111_pcd.pkl (HSV 960x540 after Dinda run_pcd fitted ONLY on our 67 train frames)
    train_lab_all = np.concatenate(train_lab_px, axis=0)
    lab_ref = [P.channel_cdf(train_lab_all[:, c]) for c in range(3)] if cfg.get("hist_match") else None

    X_pcd_list, pcd_masks_list = [], []
    seg_arg = {
        **seg,
        "adaptive": seg["adaptive"] and tuple(seg["adaptive"]),
        "edge": seg["edge"] and tuple(seg["edge"]),
    }
    for im_960, mk_960 in zip(imgs_bgr, masks_u8):
        roi_960 = (mk_960 != 255).astype(np.uint8)
        hsv = P.run_pcd(im_960, roi_960, cfg, lab_ref)[1]
        X_pcd_list.append(hsv)
        pcd_masks_list.append(P.segment(hsv, roi_960, seg_arg, t)[0])

    X_pcd = np.stack(X_pcd_list, axis=0)
    pcd_masks = np.stack(pcd_masks_list, axis=0)

    pcd_path = OUT_DIR / "dataset_ours111_pcd.pkl"
    pcd_payload = {
        "X": X_pcd,
        "Y": Y,
        "meta": meta,
        "geo": geo_dinda,
        "fit": pk.FIT,
        "classes": CLASSES,
        "pcd": pcd_name,
        "pcd_config": cfg,
        "lab_reference_cdf": lab_ref,
        "pcd_mask": pcd_masks,
        "pcd_seg": seg,
        "pcd_t": t,
        "info": (
            f"Ours 111-frame crop dataset resized to 960x540 HSV uint8 after Dinda PCD '{pcd_name}' "
            "(Lab reference CDF fitted strictly on 67 train frames), compatible with dataset_pcd.pkl schema."
        ),
    }
    with open(pcd_path, "wb") as f:
        pickle.dump(pcd_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] Saved {pcd_path} ({pcd_path.stat().st_size / 1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()

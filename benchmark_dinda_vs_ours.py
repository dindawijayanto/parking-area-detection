import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "pcd"))
sys.path.insert(0, str(ROOT / "src" / "ml"))
import parking as pk
import pipeline as P

from train_rf_vs_xgb import apply_morphology, predict_prob_map
from train_rf_vs_xgb_v2 import (
    FEATURE_NAMES_V2,
    compute_fast_metrics,
    extract_pcd_features_v2,
    overlay_vis,
    sample_pixels,
    smooth_prob_roi,
)

DATA_DIR = ROOT / "colab_export"
PKL_SEG = ROOT / "data" / "processed" / "dataset_seg.pkl"
PKL_PCD = ROOT / "data" / "processed" / "dataset_pcd.pkl"
V2_DIR = ROOT / "results" / "rf_vs_xgb_v2"
OUT_DIR = ROOT / "results" / "dinda_vs_ours"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def apply_dinda_run_pcd_to_bgr(img_bgr: np.ndarray, roi_u8: np.ndarray, cfg: dict, ref) -> np.ndarray:
    """Apply Dinda's exact run_pcd (gw+hm+median+close) and return BGR uint8 for feature extraction."""
    _, hsv = P.run_pcd(img_bgr, roi_u8, cfg, ref)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def build_lab_ref_from_ours(train_df: pd.DataFrame, gray_world: bool = True, p: int = 6):
    """Compute Dinda-style Lab CDF reference strictly from our 67 train frames (zero leakage)."""
    px = []
    for _, row in train_df.iterrows():
        im = cv2.imread(str(DATA_DIR / row["image_path"]))
        mk = cv2.imread(str(DATA_DIR / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        roi = (mk != 255).astype(np.uint8)
        if gray_world:
            im = P.shades_of_gray(im, roi, p)[0]
        px.append(cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[roi > 0])
    px = np.concatenate(px, axis=0)
    return [P.channel_cdf(px[:, c]) for c in range(3)]


def tune_postproc_on_val(model, val_feats, val_masks, val_rois):
    """Tune blur_sigma, threshold, and morph_k strictly on validation split."""
    probs = [predict_prob_map(model, f, r) for f, r in zip(val_feats, val_rois)]
    best_miou = -1.0
    best_cfg = {"blur_sigma": 1.5, "thr": 0.5, "morph_k": 15}
    for sigma in (0.0, 1.5, 2.5):
        sm_probs = [smooth_prob_roi(p, r, sigma) for p, r in zip(probs, val_rois)]
        for thr in (0.38, 0.42, 0.46, 0.50, 0.54):
            for mk_k in (7, 11, 15, 19):
                yt_all, yp_all = [], []
                for sp, mk, r in zip(sm_probs, val_masks, val_rois):
                    pred_bin = ((sp >= thr) & r).astype(np.uint8)
                    pred_clean = apply_morphology(pred_bin, r, mk_k)
                    yt_all.append(mk[r])
                    yp_all.append(pred_clean[r])
                m = compute_fast_metrics(yt_all, yp_all)["mIoU"]
                if m > best_miou:
                    best_miou = m
                    best_cfg = {"blur_sigma": sigma, "thr": thr, "morph_k": mk_k}
    return best_cfg, float(best_miou)


def evaluate_on_dinda_test(
    name: str,
    model,
    test_feats,
    test_Y,
    test_meta: pd.DataFrame,
    geo: dict,
    cfg: dict,
    direct_masks=None,
):
    """Evaluate predictions on Dinda's exact 26-frame test set & parking.free_spaces BEV."""
    yt_all, yp_all = [], []
    bev_abs_errs, bev_sq_errs = [], []
    per_vid_bucket = {}
    pred_masks = []
    t_inf0 = time.time()

    for idx in range(len(test_Y)):
        mk = test_Y[idx]
        roi = mk != 255
        vid = test_meta.iloc[idx]["video_id"]
        gt_free = int(test_meta.iloc[idx]["free spaces"])

        if direct_masks is not None:
            pred = direct_masks[idx].copy()
        else:
            prob = predict_prob_map(model, test_feats[idx], roi)
            sp = smooth_prob_roi(prob, roi, cfg["blur_sigma"])
            pred_bin = ((sp >= cfg["thr"]) & roi).astype(np.uint8)
            pred_clean = apply_morphology(pred_bin, roi, cfg["morph_k"])
            pred = np.where(roi, pred_clean, 255).astype(np.uint8)

        pred_masks.append(pred)
        pred_free = int(pk.free_spaces(pred, geo[vid])["free"])
        err = pred_free - gt_free
        bev_abs_errs.append(abs(err))
        bev_sq_errs.append(err * err)

        yt_all.append(mk[roi])
        yp_all.append(pred[roi])

        b = per_vid_bucket.setdefault(
            vid, {"yt": [], "yp": [], "gt_free": [], "pred_free": [], "abs_e": [], "sq_e": []}
        )
        b["yt"].append(mk[roi])
        b["yp"].append(pred[roi])
        b["gt_free"].append(gt_free)
        b["pred_free"].append(pred_free)
        b["abs_e"].append(abs(err))
        b["sq_e"].append(err * err)

    infer_ms = (time.time() - t_inf0) * 1000.0 / max(len(test_Y), 1)
    overall = compute_fast_metrics(yt_all, yp_all)
    per_vid = {}
    for vid, b in per_vid_bucket.items():
        vm = compute_fast_metrics(b["yt"], b["yp"])
        vm["gt_free_mean"] = float(np.mean(b["gt_free"]))
        vm["pred_free_mean"] = float(np.mean(b["pred_free"]))
        vm["bev_mae"] = float(np.mean(b["abs_e"]))
        vm["bev_rmse"] = float(np.sqrt(np.mean(b["sq_e"])))
        per_vid[vid] = vm

    dusk_miou = per_vid.get("20261104_evening_keiza", {}).get("mIoU", float("nan"))
    res = {
        "name": name,
        "config": cfg,
        "overall": overall,
        "dusk_mIoU": float(dusk_miou),
        "bev_mae": float(np.mean(bev_abs_errs)),
        "bev_rmse": float(np.sqrt(np.mean(bev_sq_errs))),
        "infer_ms": float(infer_ms),
        "per_video": per_vid,
    }
    print(
        f"[{name:30s}] mIoU={overall['mIoU']:.4f} | Floor={overall['IoU_Floor']:.4f} | "
        f"Motor={overall['IoU_Motor']:.4f} | Dusk={dusk_miou:.4f} | "
        f"BEV MAE={res['bev_mae']:.2f} | RMSE={res['bev_rmse']:.2f}",
        flush=True,
    )
    return res, pred_masks


def train_rf_and_xgb(X_train, y_train, X_val, y_val, tag: str):
    """Train identical RF v2 and XGBoost v2 architectures for controlled ablation."""
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=24,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_sec = time.time() - t0
    print(f"  [{tag}] RF trained on {X_train.shape} in {rf_sec:.1f}s", flush=True)

    t1 = time.time()
    xgb = XGBClassifier(
        n_estimators=900,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=15,
        reg_lambda=3.0,
        reg_alpha=0.5,
        tree_method="hist",
        eval_metric="logloss",
        early_stopping_rounds=40,
        n_jobs=-1,
        random_state=42,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_sec = time.time() - t1
    print(
        f"  [{tag}] XGB trained on {X_train.shape} in {xgb_sec:.1f}s (best_iter={xgb.best_iteration})",
        flush=True,
    )
    return rf, rf_sec, xgb, xgb_sec


def main():
    assert PKL_SEG.exists() and PKL_PCD.exists(), "Run src/pcd/export_dataset.py first."
    print("Loading Dinda datasets (dataset_seg.pkl & dataset_pcd.pkl)...", flush=True)
    with open(PKL_SEG, "rb") as f:
        d_seg = pickle.load(f)
    with open(PKL_PCD, "rb") as f:
        d_pcd = pickle.load(f)

    meta = d_pcd["meta"].reset_index(drop=True)
    geo = d_pcd["geo"]
    pcd_cfg = d_pcd["pcd_config"]
    train_idx = meta.index[meta["split"] == "train"].tolist()
    val_idx = meta.index[meta["split"] == "val"].tolist()
    test_idx = meta.index[meta["split"] == "test"].tolist()
    print(
        f"Dinda split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} | "
        f"PCD={d_pcd['pcd']}",
        flush=True,
    )

    test_Y = [d_seg["Y"][i] for i in test_idx]
    test_meta = meta.iloc[test_idx].reset_index(drop=True)
    test_X_raw_bgr = [d_seg["X"][i] for i in test_idx]
    test_X_dinda_bgr = [cv2.cvtColor(d_pcd["X"][i], cv2.COLOR_HSV2BGR) for i in test_idx]
    test_pcd_masks = [d_pcd["pcd_mask"][i] for i in test_idx]

    results = []
    vis_preds = {}

    # -------------------------------------------------------------------------
    # Baseline 0: Dinda Pure PCD (no ML) on Dinda 26-frame test set
    # -------------------------------------------------------------------------
    print("\n=== Baseline 0: Dinda Pure PCD (No ML) ===", flush=True)
    r_pcd, m_pcd = evaluate_on_dinda_test(
        "Dinda_Pure_PCD", None, None, test_Y, test_meta, geo, {"method": d_pcd["pcd"]}, direct_masks=test_pcd_masks
    )
    r_pcd["variant"] = "Baseline Dinda PCD (No ML)"
    r_pcd["train_sec"] = 0.0
    results.append(r_pcd)
    vis_preds["Dinda_Pure_PCD"] = m_pcd

    # -------------------------------------------------------------------------
    # VARIANT 1: Data Dinda Sepenuhnya (dataset_pcd.pkl, 92 frames)
    # -------------------------------------------------------------------------
    print("\n=== Variant 1: Data Dinda Sepenuhnya (dataset_pcd.pkl, 53 train / 13 val) ===", flush=True)
    rng1 = np.random.default_rng(42)
    X_tr1, y_tr1 = [], []
    for i in train_idx:
        bgr = cv2.cvtColor(d_pcd["X"][i], cv2.COLOR_HSV2BGR)
        feats = extract_pcd_features_v2(bgr)
        xs, ys = sample_pixels(feats, d_pcd["Y"][i], n_per_frame=9000, rng=rng1)
        X_tr1.extend(xs)
        y_tr1.extend(ys)
    X_tr1 = np.concatenate(X_tr1, axis=0)
    y_tr1 = np.concatenate(y_tr1, axis=0)

    rng1v = np.random.default_rng(123)
    X_va1, y_va1 = [], []
    val1_feats, val1_masks, val1_rois = [], [], []
    for i in val_idx:
        bgr = cv2.cvtColor(d_pcd["X"][i], cv2.COLOR_HSV2BGR)
        mk = d_pcd["Y"][i]
        feats = extract_pcd_features_v2(bgr)
        val1_feats.append(feats)
        val1_masks.append(mk)
        val1_rois.append(mk != 255)
        xs, ys = sample_pixels(feats, mk, n_per_frame=8000, rng=rng1v)
        X_va1.extend(xs)
        y_va1.extend(ys)
    X_va1 = np.concatenate(X_va1, axis=0)
    y_va1 = np.concatenate(y_va1, axis=0)

    test1_feats = [extract_pcd_features_v2(bgr) for bgr in test_X_dinda_bgr]
    rf1, rf1_s, xgb1, xgb1_s = train_rf_and_xgb(X_tr1, y_tr1, X_va1, y_va1, "V1_Dinda_Full")

    rf1_cfg, rf1_vm = tune_postproc_on_val(rf1, val1_feats, val1_masks, val1_rois)
    xgb1_cfg, xgb1_vm = tune_postproc_on_val(xgb1, val1_feats, val1_masks, val1_rois)
    print(f"  [V1] Val tuned RF={rf1_cfg} (val mIoU={rf1_vm:.4f}) | XGB={xgb1_cfg} (val mIoU={xgb1_vm:.4f})", flush=True)

    r_v1_rf, m_v1_rf = evaluate_on_dinda_test("V1_Dinda_Full_RF", rf1, test1_feats, test_Y, test_meta, geo, rf1_cfg)
    r_v1_rf["variant"] = "V1: Data Dinda Sepenuhnya"
    r_v1_rf["train_sec"] = float(rf1_s)
    r_v1_rf["val_mIoU"] = rf1_vm
    results.append(r_v1_rf)
    vis_preds["V1_Dinda_Full_RF"] = m_v1_rf

    r_v1_xgb, m_v1_xgb = evaluate_on_dinda_test("V1_Dinda_Full_XGB", xgb1, test1_feats, test_Y, test_meta, geo, xgb1_cfg)
    r_v1_xgb["variant"] = "V1: Data Dinda Sepenuhnya"
    r_v1_xgb["train_sec"] = float(xgb1_s)
    r_v1_xgb["val_mIoU"] = xgb1_vm
    results.append(r_v1_xgb)
    vis_preds["V1_Dinda_Full_XGB"] = m_v1_xgb

    joblib.dump({"model": rf1, "features": FEATURE_NAMES_V2, "cfg": rf1_cfg}, OUT_DIR / "v1_dinda_rf.joblib", compress=3)
    joblib.dump({"model": xgb1, "features": FEATURE_NAMES_V2, "cfg": xgb1_cfg}, OUT_DIR / "v1_dinda_xgb.joblib", compress=3)

    # -------------------------------------------------------------------------
    # VARIANT 2: Data Kita (111 frame) + Preprocessing Dinda (gw+hm+median+close)
    # -------------------------------------------------------------------------
    print("\n=== Variant 2: Data Kita 111 Frame + Preprocessing Dinda ===", flush=True)
    df_ours = pd.read_csv(DATA_DIR / "manifest.csv")
    ours_train = df_ours[df_ours["split"] == "train"].reset_index(drop=True)
    ours_val = df_ours[df_ours["split"] == "val"].reset_index(drop=True)

    print("  Computing Lab reference CDF strictly on our 67 train frames...", flush=True)
    ours_lab_ref = build_lab_ref_from_ours(
        ours_train, gray_world=bool(pcd_cfg.get("gray_world", True)), p=int(pcd_cfg.get("p", 6))
    )

    rng2 = np.random.default_rng(42)
    X_tr2, y_tr2 = [], []
    for _, row in ours_train.iterrows():
        im = cv2.imread(str(DATA_DIR / row["image_path"]))
        mk = cv2.imread(str(DATA_DIR / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        roi_u8 = (mk != 255).astype(np.uint8)
        bgr_pcd = apply_dinda_run_pcd_to_bgr(im, roi_u8, pcd_cfg, ours_lab_ref)
        feats = extract_pcd_features_v2(bgr_pcd)
        xs, ys = sample_pixels(feats, mk, n_per_frame=9000, rng=rng2)
        X_tr2.extend(xs)
        y_tr2.extend(ys)
    X_tr2 = np.concatenate(X_tr2, axis=0)
    y_tr2 = np.concatenate(y_tr2, axis=0)

    rng2v = np.random.default_rng(123)
    X_va2, y_va2 = [], []
    val2_feats, val2_masks, val2_rois = [], [], []
    for _, row in ours_val.iterrows():
        im = cv2.imread(str(DATA_DIR / row["image_path"]))
        mk = cv2.imread(str(DATA_DIR / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        roi_u8 = (mk != 255).astype(np.uint8)
        bgr_pcd = apply_dinda_run_pcd_to_bgr(im, roi_u8, pcd_cfg, ours_lab_ref)
        feats = extract_pcd_features_v2(bgr_pcd)
        val2_feats.append(feats)
        val2_masks.append(mk)
        val2_rois.append(mk != 255)
        xs, ys = sample_pixels(feats, mk, n_per_frame=8000, rng=rng2v)
        X_va2.extend(xs)
        y_va2.extend(ys)
    X_va2 = np.concatenate(X_va2, axis=0)
    y_va2 = np.concatenate(y_va2, axis=0)

    # Apply the exact same train-fitted Dinda preproc to Dinda's 26 test frames
    test2_feats = []
    for im_raw, mk in zip(test_X_raw_bgr, test_Y):
        roi_u8 = (mk != 255).astype(np.uint8)
        bgr_pcd = apply_dinda_run_pcd_to_bgr(im_raw, roi_u8, pcd_cfg, ours_lab_ref)
        test2_feats.append(extract_pcd_features_v2(bgr_pcd))

    rf2, rf2_s, xgb2, xgb2_s = train_rf_and_xgb(X_tr2, y_tr2, X_va2, y_va2, "V2_Ours111_DindaPreproc")

    rf2_cfg, rf2_vm = tune_postproc_on_val(rf2, val2_feats, val2_masks, val2_rois)
    xgb2_cfg, xgb2_vm = tune_postproc_on_val(xgb2, val2_feats, val2_masks, val2_rois)
    print(f"  [V2] Val tuned RF={rf2_cfg} (val mIoU={rf2_vm:.4f}) | XGB={xgb2_cfg} (val mIoU={xgb2_vm:.4f})", flush=True)

    r_v2_rf, m_v2_rf = evaluate_on_dinda_test("V2_Ours111_DindaPreproc_RF", rf2, test2_feats, test_Y, test_meta, geo, rf2_cfg)
    r_v2_rf["variant"] = "V2: Data Kita + Preproc Dinda"
    r_v2_rf["train_sec"] = float(rf2_s)
    r_v2_rf["val_mIoU"] = rf2_vm
    results.append(r_v2_rf)
    vis_preds["V2_Ours111_DindaPreproc_RF"] = m_v2_rf

    r_v2_xgb, m_v2_xgb = evaluate_on_dinda_test("V2_Ours111_DindaPreproc_XGB", xgb2, test2_feats, test_Y, test_meta, geo, xgb2_cfg)
    r_v2_xgb["variant"] = "V2: Data Kita + Preproc Dinda"
    r_v2_xgb["train_sec"] = float(xgb2_s)
    r_v2_xgb["val_mIoU"] = xgb2_vm
    results.append(r_v2_xgb)
    vis_preds["V2_Ours111_DindaPreproc_XGB"] = m_v2_xgb

    joblib.dump({"model": rf2, "features": FEATURE_NAMES_V2, "cfg": rf2_cfg}, OUT_DIR / "v2_ours_dindapre_rf.joblib", compress=3)
    joblib.dump({"model": xgb2, "features": FEATURE_NAMES_V2, "cfg": xgb2_cfg}, OUT_DIR / "v2_ours_dindapre_xgb.joblib", compress=3)

    # -------------------------------------------------------------------------
    # VARIANT 3: Murni Kita (RF_v2 & XGB_v2) Diuji Langsung pada Data Dinda
    # -------------------------------------------------------------------------
    print("\n=== Variant 3: Murni Kita (v2) Diuji Langsung pada Data Test Dinda ===", flush=True)
    v2_summary = json.loads((V2_DIR / "v2_benchmark_summary.json").read_text(encoding="utf-8"))
    rf_pure = joblib.load(V2_DIR / "rf_v2_model.joblib")["model"]
    xgb_pure = joblib.load(V2_DIR / "xgb_v2_model.joblib")["model"]
    test3_feats = [extract_pcd_features_v2(im_raw) for im_raw in test_X_raw_bgr]

    r_v3_rf, m_v3_rf = evaluate_on_dinda_test(
        "V3_Ours_Pure_v2_RF", rf_pure, test3_feats, test_Y, test_meta, geo, v2_summary["RF_v2"]["config"]
    )
    r_v3_rf["variant"] = "V3: Murni Kita -> Test Dinda"
    r_v3_rf["train_sec"] = float(v2_summary["RF_v2"]["train_sec"])
    results.append(r_v3_rf)
    vis_preds["V3_Ours_Pure_v2_RF"] = m_v3_rf

    r_v3_xgb, m_v3_xgb = evaluate_on_dinda_test(
        "V3_Ours_Pure_v2_XGB", xgb_pure, test3_feats, test_Y, test_meta, geo, v2_summary["XGB_v2"]["config"]
    )
    r_v3_xgb["variant"] = "V3: Murni Kita -> Test Dinda"
    r_v3_xgb["train_sec"] = float(v2_summary["XGB_v2"]["train_sec"])
    results.append(r_v3_xgb)
    vis_preds["V3_Ours_Pure_v2_XGB"] = m_v3_xgb

    # -------------------------------------------------------------------------
    # Save CSV + JSON summary
    # -------------------------------------------------------------------------
    rows = []
    for r in results:
        pv = r["per_video"]
        rows.append({
            "Model": r["name"],
            "Variant": r["variant"],
            "mIoU": round(r["overall"]["mIoU"], 4),
            "IoU_Floor": round(r["overall"]["IoU_Floor"], 4),
            "IoU_Motor": round(r["overall"]["IoU_Motor"], 4),
            "Macro_F1": round(r["overall"]["Macro_F1"], 4),
            "Pixel_Acc": round(r["overall"]["Pixel_Acc"], 4),
            "Afternoon_mIoU": round(pv["20261104_afternoon_dinda"]["mIoU"], 4),
            "Dusk_mIoU": round(pv["20261104_evening_keiza"]["mIoU"], 4),
            "Morning_mIoU": round(pv["20261112_morning_keiza"]["mIoU"], 4),
            "BEV_MAE": round(r["bev_mae"], 2),
            "BEV_RMSE": round(r["bev_rmse"], 2),
            "Dusk_BEV_MAE": round(pv["20261104_evening_keiza"]["bev_mae"], 2),
            "Train_s": round(r["train_sec"], 1),
            "Infer_ms": round(r["infer_ms"], 1),
        })
    df_res = pd.DataFrame(rows)
    df_res.to_csv(OUT_DIR / "comparison_table.csv", index=False)
    with open(OUT_DIR / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY TABLE (ALL ON DINDA'S EXACT 26-FRAME TEST SET) ===", flush=True)
    print(df_res.to_string(index=False), flush=True)

    # -------------------------------------------------------------------------
    # Plot 1: Grouped Bar Chart (mIoU, Dusk mIoU, BEV MAE)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), facecolor="white")
    labels = [r["Model"].replace("V2_Ours111_DindaPreproc_", "V2_Ours+DP_").replace("V3_Ours_Pure_v2_", "V3_OursPure_").replace("V1_Dinda_Full_", "V1_Dinda_") for r in rows]
    x = np.arange(len(labels))
    colors = ["#7f8c8d", "#2980b9", "#1f618d", "#27ae60", "#1e8449", "#e67e22", "#d35400"]

    for ax, col, title, fmt in [
        (axes[0], "mIoU", "Overall Test mIoU (Higher = Better)", "{:.3f}"),
        (axes[1], "Dusk_mIoU", "Dusk Test mIoU (Higher = Better)", "{:.3f}"),
        (axes[2], "BEV_MAE", "BEV Free Spaces MAE (Lower = Better)", "{:.2f}"),
    ]:
        vals = df_res[col].values
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=8.5)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (max(vals) * 0.015), fmt.format(v),
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(vals) * 1.15)

    fig.suptitle("3-Way Ablation on Dinda's Exact 26-Frame Test Set", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison_bar_chart.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 2: Visual Comparison Grid (3 representative test frames: Afternoon, Dusk, Morning)
    # -------------------------------------------------------------------------
    rep_vids = ["20261104_afternoon_dinda", "20261104_evening_keiza", "20261112_morning_keiza"]
    rep_indices = []
    for vid in rep_vids:
        idxs = test_meta.index[test_meta["video_id"] == vid].tolist()
        rep_indices.append(idxs[len(idxs) // 2])

    cols_vis = [
        ("GT", None),
        ("Dinda Pure PCD", "Dinda_Pure_PCD"),
        ("V1: Dinda Full (XGB)", "V1_Dinda_Full_XGB"),
        ("V2: Ours+DP (XGB)", "V2_Ours111_DindaPreproc_XGB"),
        ("V3: Ours Pure (RF)", "V3_Ours_Pure_v2_RF"),
        ("V3: Ours Pure (XGB)", "V3_Ours_Pure_v2_XGB"),
    ]
    fig, axes = plt.subplots(3, len(cols_vis), figsize=(22, 8.5), facecolor="white")
    for r_i, t_idx in enumerate(rep_indices):
        im_raw = test_X_raw_bgr[t_idx]
        mk_gt = test_Y[t_idx]
        vid = test_meta.iloc[t_idx]["video_id"]
        gt_free = int(test_meta.iloc[t_idx]["free spaces"])
        for c_i, (col_title, key) in enumerate(cols_vis):
            ax = axes[r_i, c_i]
            if key is None:
                vis = overlay_vis(im_raw, mk_gt)
                sub = f"{vid}\nGT (free={gt_free})"
            else:
                pm = vis_preds[key][t_idx]
                vis = overlay_vis(im_raw, pm)
                roi = mk_gt != 255
                fm = compute_fast_metrics([mk_gt[roi]], [pm[roi]])["mIoU"]
                pf = int(pk.free_spaces(pm, geo[vid])["free"])
                sub = f"{col_title}\nmIoU={fm:.3f}, free={pf}"
            ax.imshow(vis)
            ax.set_title(sub, fontsize=8.5)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison_visual_grid.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n[+] Saved all outputs to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

import json
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
from sklearn.metrics import confusion_matrix, f1_score
from xgboost import XGBClassifier

DATA_DIR = Path("colab_export")
OUT_DIR = Path("results/rf_vs_xgb")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "H_shifted", "H_sin", "H_cos", "S", "V_clahe",
    "L_lab", "a_lab", "b_lab",
    "V_rel_31", "L_rel_31",
    "std_5", "std_11", "std_25",
    "std_11_a", "std_11_b", "std_11_S",
    "sobel_5", "sobel_15",
    "canny_9", "canny_21",
    "lap_9", "lap_21",
]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def local_std(arr: np.ndarray, k: int) -> np.ndarray:
    """Compute local standard deviation over a k x k box window."""
    m = cv2.boxFilter(arr, -1, (k, k))
    m2 = cv2.boxFilter(arr * arr, -1, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def extract_pcd_features(img_bgr: np.ndarray) -> np.ndarray:
    """Extract 22 classical PCD color, contrast, and multi-scale texture features."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_raw = hsv[:, :, 0].astype(np.float32)  # 0..179
    h_shift = ((h_raw + 90.0) % 180.0) / 180.0
    h_rad = h_raw * (2.0 * np.pi / 180.0)
    h_sin = np.sin(h_rad)
    h_cos = np.cos(h_rad)
    s = hsv[:, :, 1].astype(np.float32) / 255.0

    v_clahe_u8 = cv2.medianBlur(clahe.apply(hsv[:, :, 2]), 5)
    v_clahe = v_clahe_u8.astype(np.float32) / 255.0

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    L_lab, a_lab, b_lab = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Illumination-invariant local contrast
    v_rel_31 = v_clahe - cv2.boxFilter(v_clahe, -1, (31, 31))
    L_rel_31 = L_lab - cv2.boxFilter(L_lab, -1, (31, 31))

    # Multi-scale luminance & chromaticity texture
    std_5 = local_std(v_clahe, 5)
    std_11 = local_std(v_clahe, 11)
    std_25 = local_std(v_clahe, 25)
    std_11_a = local_std(a_lab, 11)
    std_11_b = local_std(b_lab, 11)
    std_11_S = local_std(s, 11)

    # Gradient, edge density & Laplacian energy
    gx = cv2.Sobel(v_clahe, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(v_clahe, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)
    sobel_5 = cv2.boxFilter(gmag, -1, (5, 5))
    sobel_15 = cv2.boxFilter(gmag, -1, (15, 15))

    edges = (cv2.Canny(v_clahe_u8, 60, 150) > 0).astype(np.float32)
    canny_9 = cv2.boxFilter(edges, -1, (9, 9))
    canny_21 = cv2.boxFilter(edges, -1, (21, 21))

    lap = np.abs(cv2.Laplacian(v_clahe, cv2.CV_32F, ksize=3))
    lap_9 = cv2.boxFilter(lap, -1, (9, 9))
    lap_21 = cv2.boxFilter(lap, -1, (21, 21))

    return np.stack([
        h_shift, h_sin, h_cos, s, v_clahe,
        L_lab, a_lab, b_lab,
        v_rel_31, L_rel_31,
        std_5, std_11, std_25,
        std_11_a, std_11_b, std_11_S,
        sobel_5, sobel_15,
        canny_9, canny_21,
        lap_9, lap_21,
    ], axis=-1).astype(np.float32)


def sample_split(df_sub: pd.DataFrame, pixels_per_frame: int, seed: int):
    """Sample balanced (floor=0, motor=1) pixel feature vectors from a split."""
    rng = np.random.default_rng(seed)
    Xs, ys_out = [], []
    for _, row in df_sub.iterrows():
        im = cv2.imread(str(DATA_DIR / row["image_path"]))
        mk = cv2.imread(str(DATA_DIR / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        feats = extract_pcd_features(im)
        for cls_id in (0, 1):
            ry, rx = np.where(mk == cls_id)
            if len(ry) == 0:
                continue
            n_take = min(pixels_per_frame // 2, len(ry))
            pick = rng.choice(len(ry), size=n_take, replace=False)
            Xs.append(feats[ry[pick], rx[pick]])
            ys_out.append(np.full(n_take, cls_id, dtype=np.uint8))
    return np.vstack(Xs), np.concatenate(ys_out)


def predict_prob_map(model, feats: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Predict positive class (motor) probability inside ROI in chunks."""
    prob = np.zeros(roi_mask.shape, dtype=np.float32)
    ry, rx = np.where(roi_mask)
    if len(ry) == 0:
        return prob
    X_valid = feats[ry, rx]
    probs = np.empty(len(ry), dtype=np.float32)
    chunk = 150_000
    for s in range(0, len(ry), chunk):
        probs[s:s + chunk] = model.predict_proba(X_valid[s:s + chunk])[:, 1]
    prob[ry, rx] = probs
    return prob


def apply_morphology(pred_bin: np.ndarray, roi_mask: np.ndarray, ksize: int) -> np.ndarray:
    """Apply elliptical morphological open+close constrained to roi_mask."""
    if ksize <= 0:
        return ((pred_bin > 0) & roi_mask).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    cleaned = cv2.morphologyEx(pred_bin.astype(np.uint8), cv2.MORPH_OPEN, k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k)
    return ((cleaned > 0) & roi_mask).astype(np.uint8)


def compute_metrics(yt_list, yp_list):
    """Compute IoU_Floor, IoU_Motor, mIoU, Macro_F1, and Pixel_Acc."""
    yt = np.concatenate(yt_list)
    yp = np.concatenate(yp_list)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    iou_floor = tn / max(tn + fp + fn, 1)
    iou_motor = tp / max(tp + fp + fn, 1)
    miou = 0.5 * (iou_floor + iou_motor)
    acc = (tn + tp) / max(cm.sum(), 1)
    mf1 = f1_score(yt, yp, average="macro")
    return {
        "IoU_Floor": float(iou_floor),
        "IoU_Motor": float(iou_motor),
        "mIoU": float(miou),
        "Macro_F1": float(mf1),
        "Pixel_Acc": float(acc),
    }


def overlay_vis(im_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Overlay RGB label colors onto BGR frame for visualization."""
    out = im_bgr[:, :, ::-1].astype(np.float32)
    colors = {0: (60, 230, 151), 1: (255, 86, 111), 255: (112, 146, 224)}
    for v, rgb in colors.items():
        sel = mask_u8 == v
        if np.any(sel):
            out[sel] = out[sel] * 0.55 + np.array(rgb, dtype=np.float32) * 0.45
    return out.astype(np.uint8)


def main():
    manifest_path = DATA_DIR / "manifest.csv"
    assert manifest_path.exists(), f"Missing {manifest_path}"
    df = pd.read_csv(manifest_path)
    assert len(df) >= 111, f"Expected >= 111 frames, got {len(df)}"
    dates = {s: set(g["date"]) for s, g in df.groupby("split")}
    assert dates["train"].isdisjoint(dates["val"])
    assert dates["train"].isdisjoint(dates["test"])
    assert dates["val"].isdisjoint(dates["test"])
    print(f"Dataset: {len(df)} frames | splits: {df['split'].value_counts().to_dict()}")
    print("Anti-leakage check PASSED (0 overlapping dates across train/val/test).")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    t0 = time.time()
    X_train, y_train = sample_split(train_df, pixels_per_frame=8000, seed=42)
    X_val_s, y_val_s = sample_split(val_df, pixels_per_frame=6000, seed=123)
    print(
        f"Sampled X_train={X_train.shape} (floor={(y_train==0).sum():,}, motor={(y_train==1).sum():,}), "
        f"X_val_s={X_val_s.shape} in {time.time()-t0:.1f}s"
    )

    # Pre-extract features for val & test frames once (fair timing & fast tuning)
    t_fe = time.time()
    val_frames = []
    for _, row in val_df.iterrows():
        im = cv2.imread(str(DATA_DIR / row["image_path"]))
        mk = cv2.imread(str(DATA_DIR / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        val_frames.append((row, im, mk, mk != 255, extract_pcd_features(im)))
    test_frames = []
    for _, row in test_df.iterrows():
        im = cv2.imread(str(DATA_DIR / row["image_path"]))
        mk = cv2.imread(str(DATA_DIR / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        test_frames.append((row, im, mk, mk != 255, extract_pcd_features(im)))
    print(f"Extracted 22 PCD features for {len(val_frames)} val + {len(test_frames)} test frames in {time.time()-t_fe:.1f}s")

    # 1. Train Random Forest
    print("\n--- Training Random Forest ---")
    t_rf = time.time()
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
    rf_train_s = time.time() - t_rf
    print(f"Random Forest trained in {rf_train_s:.1f}s")

    # 2. Train XGBoost (hist, early stopping on val sample)
    print("\n--- Training XGBoost ---")
    t_xgb = time.time()
    xgb = XGBClassifier(
        n_estimators=900,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=2.0,
        reg_alpha=0.2,
        tree_method="hist",
        eval_metric="logloss",
        early_stopping_rounds=40,
        n_jobs=-1,
        random_state=42,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val_s, y_val_s)], verbose=False)
    xgb_train_s = time.time() - t_xgb
    print(f"XGBoost trained in {xgb_train_s:.1f}s (best_iteration={xgb.best_iteration})")

    # Save models to measure size
    rf_path = OUT_DIR / "rf_model.joblib"
    xgb_path = OUT_DIR / "xgb_model.joblib"
    joblib.dump({"model": rf, "features": FEATURE_NAMES}, rf_path, compress=3)
    joblib.dump({"model": xgb, "features": FEATURE_NAMES}, xgb_path, compress=3)
    rf_mb = rf_path.stat().st_size / (1024 * 1024)
    xgb_mb = xgb_path.stat().st_size / (1024 * 1024)

    models = {
        "RandomForest": (rf, rf_train_s, rf_mb),
        "XGBoost": (xgb, xgb_train_s, xgb_mb),
    }

    summary = {}
    test_preds_cache = {}

    for name, (mdl, train_s, size_mb) in models.items():
        val_probs = [predict_prob_map(mdl, feats, roi) for _, _, _, roi, feats in val_frames]
        best_val = None
        best_raw_val = None
        for thr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
            for ksize in [0, 7, 11, 15]:
                yt_l, yp_l = [], []
                for (_, _, mk, roi, _), prob in zip(val_frames, val_probs):
                    pred = apply_morphology((prob >= thr).astype(np.uint8), roi, ksize)
                    yt_l.append(mk[roi])
                    yp_l.append(pred[roi])
                m = compute_metrics(yt_l, yp_l)
                cfg = {"thr": thr, "morph_k": ksize, **m}
                if ksize == 0 and (best_raw_val is None or m["mIoU"] > best_raw_val["mIoU"]):
                    best_raw_val = cfg
                if best_val is None or m["mIoU"] > best_val["mIoU"]:
                    best_val = cfg

        t_inf = time.time()
        test_probs = [predict_prob_map(mdl, feats, roi) for _, _, _, roi, feats in test_frames]
        inf_ms = (time.time() - t_inf) * 1000.0 / len(test_frames)

        # Raw (no morphology, best val raw threshold)
        yt_r, yp_r = [], []
        for (_, _, mk, roi, _), prob in zip(test_frames, test_probs):
            pred_r = ((prob >= best_raw_val["thr"]) & roi).astype(np.uint8)
            yt_r.append(mk[roi])
            yp_r.append(pred_r[roi])
        test_raw = compute_metrics(yt_r, yp_r)

        # Morphed (best val threshold + morph_k)
        yt_m, yp_m = [], []
        per_rec = {}
        vis_store = {}
        for (row, im, mk, roi, _), prob in zip(test_frames, test_probs):
            pred_m = apply_morphology((prob >= best_val["thr"]).astype(np.uint8), roi, best_val["morph_k"])
            yt_m.append(mk[roi])
            yp_m.append(pred_m[roi])
            rec = row["recording_id"]
            per_rec.setdefault(rec, ([], []))
            per_rec[rec][0].append(mk[roi])
            per_rec[rec][1].append(pred_m[roi])
            if row["timestamp"] == "00-45":
                vis_store[rec] = (row["frame_id"], im, mk, pred_m)

        test_morph = compute_metrics(yt_m, yp_m)
        per_rec_metrics = {rec: compute_metrics(a, b) for rec, (a, b) in sorted(per_rec.items())}
        test_preds_cache[name] = vis_store

        summary[name] = {
            "train_sec": round(train_s, 2),
            "infer_ms_per_frame": round(inf_ms, 1),
            "model_size_mb": round(size_mb, 2),
            "best_val_raw": best_raw_val,
            "best_val_morph": best_val,
            "test_raw": test_raw,
            "test_morph": test_morph,
            "test_per_video": per_rec_metrics,
        }

    print("\n================ FINAL BENCHMARK (HELD-OUT TEST: 4 NOV & 12 NOV, 26 FRAMES) ================")
    rows_out = [
        {
            "Model": "Naive Bayes (Baseline Repo Dinda)",
            "mIoU": 0.6470, "IoU_Floor": 0.5370, "IoU_Motor": 0.7570,
            "Macro_F1": np.nan, "Pixel_Acc": 0.8110,
            "Train_s": np.nan, "Infer_ms": np.nan, "Size_MB": np.nan,
        },
    ]
    for name in ("RandomForest", "XGBoost"):
        s = summary[name]
        for tag, m in [("raw", s["test_raw"]), ("+morph", s["test_morph"])]:
            rows_out.append({
                "Model": f"{name} ({tag})",
                "mIoU": round(m["mIoU"], 4),
                "IoU_Floor": round(m["IoU_Floor"], 4),
                "IoU_Motor": round(m["IoU_Motor"], 4),
                "Macro_F1": round(m["Macro_F1"], 4),
                "Pixel_Acc": round(m["Pixel_Acc"], 4),
                "Train_s": s["train_sec"],
                "Infer_ms": s["infer_ms_per_frame"],
                "Size_MB": s["model_size_mb"],
            })
    res_df = pd.DataFrame(rows_out)
    print(res_df.to_string(index=False))
    res_df.to_csv(OUT_DIR / "benchmark_table.csv", index=False)

    print("\n--- Per-Video Held-Out Test (+morph) ---")
    for rec in sorted(summary["RandomForest"]["test_per_video"].keys()):
        rf_m = summary["RandomForest"]["test_per_video"][rec]
        xg_m = summary["XGBoost"]["test_per_video"][rec]
        print(
            f"  {rec:38s} | RF mIoU={rf_m['mIoU']:.4f} (F={rf_m['IoU_Floor']:.3f}, M={rf_m['IoU_Motor']:.3f}) "
            f"| XGB mIoU={xg_m['mIoU']:.4f} (F={xg_m['IoU_Floor']:.3f}, M={xg_m['IoU_Motor']:.3f})"
        )

    (OUT_DIR / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Side-by-side Feature Importance chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (name, mdl, col) in zip(axes, [("Random Forest", rf, "#2563eb"), ("XGBoost", xgb, "#ea580c")]):
        imp = pd.Series(mdl.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=True)
        imp.plot(kind="barh", ax=ax, color=col)
        ax.set_title(f"{name} Feature Importance")
        ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    # Visual test comparison grid (Original | GT | Random Forest | XGBoost)
    recs = sorted(test_preds_cache["RandomForest"].keys())
    fig, axes = plt.subplots(len(recs), 4, figsize=(20, 4.2 * len(recs)))
    if len(recs) == 1:
        axes = [axes]
    for row_ax, rec in zip(axes, recs):
        fid, im, mk, rf_pred = test_preds_cache["RandomForest"][rec]
        _, _, _, xgb_pred = test_preds_cache["XGBoost"][rec]
        rf_full = np.where(mk == 255, 255, rf_pred).astype(np.uint8)
        xgb_full = np.where(mk == 255, 255, xgb_pred).astype(np.uint8)
        rf_miou = summary["RandomForest"]["test_per_video"][rec]["mIoU"]
        xg_miou = summary["XGBoost"]["test_per_video"][rec]["mIoU"]

        row_ax[0].imshow(im[:, :, ::-1]); row_ax[0].set_title(f"Original: {fid}"); row_ax[0].axis("off")
        row_ax[1].imshow(overlay_vis(im, mk)); row_ax[1].set_title("Ground Truth"); row_ax[1].axis("off")
        row_ax[2].imshow(overlay_vis(im, rf_full)); row_ax[2].set_title(f"Random Forest (mIoU={rf_miou:.3f})"); row_ax[2].axis("off")
        row_ax[3].imshow(overlay_vis(im, xgb_full)); row_ax[3].set_title(f"XGBoost (mIoU={xg_miou:.3f})"); row_ax[3].axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_comparison_grid.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved plots & models to {OUT_DIR}/")


if __name__ == "__main__":
    main()

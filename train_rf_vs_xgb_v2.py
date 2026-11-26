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
from xgboost import XGBClassifier

from train_rf_vs_xgb import apply_morphology, extract_pcd_features as extract_pcd_features_v1, predict_prob_map

DATA_DIR = Path("colab_export")
ROI_DIR = Path("data/annotations/roi")
REF_DIR = Path("data/annotations/references")
V1_DIR = Path("results/rf_vs_xgb")
OUT_DIR = Path("results/rf_vs_xgb_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEV_SIZE = 600
FIT = 0.7

BEV_MAP = {
    "03-04-11-2026-afternoon-top-dinda": "20261104_afternoon_dinda",
    "04-04-11-2026-evening-top-keiza": "20261104_evening_keiza",
    "09-12-11-2026-morning-top-keiza": "20261112_morning_keiza",
    "08-10-11-2026-afternoon-rainy-top-tobi": "20261110_afternoonrain_tobi",
}

FEATURE_NAMES_V2 = [
    "H_shifted", "H_sin", "H_cos", "S", "V_clahe",
    "a_lab", "b_lab", "r_norm", "g_norm", "b_norm", "r_minus_b",
    "msr_v", "V_rel_31", "b_rel_31",
    "std_5", "std_11", "std_25", "std_11_a", "std_11_b", "std_11_S",
    "sobel_5", "sobel_15", "canny_11", "canny_21", "lap_11", "local_range_15",
]

CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
K_ELLIPSE_15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


def local_std(arr: np.ndarray, k: int) -> np.ndarray:
    """Compute local standard deviation over a k x k box window."""
    m = cv2.boxFilter(arr, -1, (k, k))
    m2 = cv2.boxFilter(arr * arr, -1, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def extract_pcd_features_v2(img_bgr: np.ndarray) -> np.ndarray:
    """Extract 26 occupancy-invariant, dusk-resilient classical PCD features."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_raw = hsv[:, :, 0].astype(np.float32)
    h_shift = ((h_raw + 90.0) % 180.0) / 180.0
    h_rad = h_raw * (2.0 * np.pi / 180.0)
    h_sin = np.sin(h_rad)
    h_cos = np.cos(h_rad)

    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v_raw = hsv[:, :, 2].astype(np.float32)
    v_clahe_u8 = cv2.medianBlur(CLAHE.apply(hsv[:, :, 2]), 5)
    v_clahe = v_clahe_u8.astype(np.float32) / 255.0

    # Purely chromatic Lab channels (exclude raw L_lab to avoid brightness shift)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    a_lab, b_lab = lab[:, :, 1], lab[:, :, 2]

    # Normalized RGB chromaticity
    bgr_f = img_bgr.astype(np.float32)
    bgr_sum = bgr_f.sum(axis=2) + 1.0
    r_norm = bgr_f[:, :, 2] / bgr_sum
    g_norm = bgr_f[:, :, 1] / bgr_sum
    b_norm = bgr_f[:, :, 0] / bgr_sum
    r_minus_b = r_norm - b_norm

    # Multi-Scale Retinex on log(V+1) + local spatial contrast
    lv = np.log1p(v_raw)
    msr_v = lv - (
        cv2.GaussianBlur(lv, (0, 0), 7)
        + cv2.GaussianBlur(lv, (0, 0), 21)
        + cv2.GaussianBlur(lv, (0, 0), 45)
    ) / 3.0
    V_rel_31 = v_clahe - cv2.boxFilter(v_clahe, -1, (31, 31))
    b_rel_31 = b_lab - cv2.boxFilter(b_lab, -1, (31, 31))

    # Multi-scale texture on CLAHE V and chromaticity
    std_5 = local_std(v_clahe, 5)
    std_11 = local_std(v_clahe, 11)
    std_25 = local_std(v_clahe, 25)
    std_11_a = local_std(a_lab, 11)
    std_11_b = local_std(b_lab, 11)
    std_11_S = local_std(s, 11)

    # Gradient, edges, Laplacian & morphological local range
    gx = cv2.Sobel(v_clahe, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(v_clahe, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)
    sobel_5 = cv2.boxFilter(gmag, -1, (5, 5))
    sobel_15 = cv2.boxFilter(gmag, -1, (15, 15))

    edges = (cv2.Canny(v_clahe_u8, 60, 150) > 0).astype(np.float32)
    canny_11 = cv2.boxFilter(edges, -1, (11, 11))
    canny_21 = cv2.boxFilter(edges, -1, (21, 21))

    lap = np.abs(cv2.Laplacian(v_clahe, cv2.CV_32F, ksize=3))
    lap_11 = cv2.boxFilter(lap, -1, (11, 11))

    local_range_15 = cv2.dilate(v_clahe, K_ELLIPSE_15) - cv2.erode(v_clahe, K_ELLIPSE_15)

    return np.stack([
        h_shift, h_sin, h_cos, s, v_clahe,
        a_lab, b_lab, r_norm, g_norm, b_norm, r_minus_b,
        msr_v, V_rel_31, b_rel_31,
        std_5, std_11, std_25, std_11_a, std_11_b, std_11_S,
        sobel_5, sobel_15, canny_11, canny_21, lap_11, local_range_15,
    ], axis=-1).astype(np.float32)


def synthesize_dusk(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Create a physics-matched neutral-chroma low-light dusk variant without altering labels."""
    gamma = float(rng.uniform(1.05, 1.35))
    alpha = float(rng.uniform(0.45, 0.68))

    f = np.power(np.clip(img_bgr.astype(np.float32) / 255.0, 0.0, 1.0), gamma) * alpha
    dark = np.clip(f * 255.0, 0.0, 255.0).astype(np.uint8)

    # Compress Lab chroma toward neutral 128 to match real dusk physics
    lab = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 1] = 128.0 + (lab[:, :, 1] - 128.0) * float(rng.uniform(0.55, 0.85))
    lab[:, :, 2] = 128.0 + (lab[:, :, 2] - 128.0) * float(rng.uniform(0.45, 0.75))
    dark = cv2.cvtColor(np.clip(lab, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

    noise = rng.normal(0.0, float(rng.uniform(1.0, 3.0)), size=dark.shape).astype(np.float32)
    return np.clip(dark + noise, 0.0, 255.0).astype(np.uint8)


def sample_pixels(feats: np.ndarray, mk: np.ndarray, n_per_frame: int, rng: np.random.Generator):
    """Sample balanced floor/motor pixel vectors from precomputed feature map."""
    Xs, ys_out = [], []
    for cls_id in (0, 1):
        ry, rx = np.where(mk == cls_id)
        if len(ry) == 0:
            continue
        n_take = min(n_per_frame // 2, len(ry))
        pick = rng.choice(len(ry), size=n_take, replace=False)
        Xs.append(feats[ry[pick], rx[pick]])
        ys_out.append(np.full(n_take, cls_id, dtype=np.uint8))
    return Xs, ys_out


def build_augmented_samples(df_sub: pd.DataFrame, orig_per_frame: int, dusk_per_frame: int, seed: int):
    """Sample balanced original + synthetic dusk pixel features from a split."""
    rng = np.random.default_rng(seed)
    Xs, ys_out = [], []
    for _, row in df_sub.iterrows():
        im = cv2.imread(str(DATA_DIR / row["image_path"]))
        mk = cv2.imread(str(DATA_DIR / row["mask_path"]), cv2.IMREAD_GRAYSCALE)

        feats_orig = extract_pcd_features_v2(im)
        x_o, y_o = sample_pixels(feats_orig, mk, orig_per_frame, rng)
        Xs.extend(x_o)
        ys_out.extend(y_o)

        if dusk_per_frame > 0:
            im_dusk = synthesize_dusk(im, rng)
            feats_dusk = extract_pcd_features_v2(im_dusk)
            x_d, y_d = sample_pixels(feats_dusk, mk, dusk_per_frame, rng)
            Xs.extend(x_d)
            ys_out.extend(y_d)

    return np.vstack(Xs), np.concatenate(ys_out)


def bev_homography(roi_data: dict):
    """Exact match to notebooks/01_roi_and_masks.ipynb bev_homography(roi)."""
    square = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    ground = np.array(roi_data["ground"], dtype=np.float32)
    H = cv2.getPerspectiveTransform(ground, square)
    pts = np.concatenate([np.array(p, dtype=np.float32) for p in roi_data["roi"]]).reshape(-1, 1, 2)
    bev = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    x0, y0 = bev.min(axis=0)
    x1, y1 = bev.max(axis=0)
    scale = BEV_SIZE / max(float(x1 - x0), float(y1 - y0))
    T = np.float64([[scale, 0, -x0 * scale], [0, scale, -y0 * scale], [0, 0, 1]])
    size = (int(np.ceil((x1 - x0) * scale)) + 1, int(np.ceil((y1 - y0) * scale)) + 1)
    return T @ H, size


def prepare_bev_geometries():
    """Prepare BEV homographies, motor footprint kernels, and SIFT reference descriptors."""
    sift = cv2.SIFT_create(nfeatures=5000)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    geo = {}
    for rec, vid in BEV_MAP.items():
        roi_data = json.loads((ROI_DIR / f"{vid}.json").read_text(encoding="utf-8"))
        H_ref_to_bev, size = bev_homography(roi_data)
        motor_poly = np.array(roi_data["motor"], dtype=np.float32).reshape(-1, 1, 2)
        motor_bev = cv2.perspectiveTransform(motor_poly, H_ref_to_bev).reshape(-1, 2)
        motor_area = float(cv2.contourArea(motor_bev))

        shape = (motor_bev - motor_bev.mean(axis=0)) * FIT
        shape = shape - shape.min(axis=0)
        kh = int(np.ceil(shape[:, 1].max())) + 1
        kw = int(np.ceil(shape[:, 0].max())) + 1
        kernel = np.zeros((kh, kw), dtype=np.uint8)
        cv2.fillPoly(kernel, [np.int32(np.round(shape))], 1)

        ref_img = cv2.imread(str(REF_DIR / f"{vid}.jpg"))
        assert ref_img is not None, f"Missing reference image {vid}.jpg"
        ref_gray = CLAHE.apply(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY))
        kp_r, des_r = sift.detectAndCompute(ref_gray, None)

        geo[rec] = {
            "vid": vid,
            "H_ref_to_bev": H_ref_to_bev,
            "size": size,
            "motor_area": motor_area,
            "kernel": kernel,
            "kp_r": kp_r,
            "des_r": des_r,
        }
    return geo, sift, bf


def compute_crop_to_bev(img_bgr: np.ndarray, g: dict, sift, bf) -> tuple[np.ndarray, int]:
    """Align crop frame to reference image via SIFT+RANSAC and return H_crop_to_bev."""
    gray = CLAHE.apply(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    kp_i, des_i = sift.detectAndCompute(gray, None)
    matches = bf.knnMatch(des_i, g["des_r"], k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    assert len(good) >= 12, f"Too few SIFT matches ({len(good)}) for {g['vid']}"
    src = np.float32([kp_i[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([g["kp_r"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_crop_to_ref, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    n_inl = int(inl.sum()) if inl is not None else 0
    assert n_inl >= 200, f"Low RANSAC inliers ({n_inl}) for {g['vid']}"
    return g["H_ref_to_bev"] @ H_crop_to_ref, n_inl


def count_free_spaces(mask_u8: np.ndarray, H_crop_to_bev: np.ndarray, g: dict):
    """Count feasible motorcycle slots in BEV (exact match to 01_roi_and_masks.ipynb)."""
    floor = (mask_u8 == 0).astype(np.uint8)
    floor_bev = cv2.warpPerspective(floor, H_crop_to_bev, g["size"], flags=cv2.INTER_NEAREST)
    usable = cv2.morphologyEx(floor_bev, cv2.MORPH_OPEN, g["kernel"])
    n, _, comp_stats, _ = cv2.connectedComponentsWithStats(usable)
    free = 0
    for c in range(1, n):
        free += max(1, int(round(comp_stats[c, cv2.CC_STAT_AREA] / g["motor_area"])))
    return free, floor_bev, usable


def smooth_prob_roi(prob: np.ndarray, roi_mask: np.ndarray, sigma: float) -> np.ndarray:
    """Apply normalized Gaussian blur inside roi_mask so Ignore never bleeds in."""
    if sigma <= 0.0:
        return prob
    roi_f = roi_mask.astype(np.float32)
    num = cv2.GaussianBlur(prob * roi_f, (0, 0), sigmaX=sigma)
    den = cv2.GaussianBlur(roi_f, (0, 0), sigmaX=sigma)
    out = np.zeros_like(prob, dtype=np.float32)
    out[roi_mask] = num[roi_mask] / np.maximum(den[roi_mask], 1e-6)
    return out


def compute_fast_metrics(yt_list, yp_list):
    """Vectorized confusion-matrix metrics (~50x faster than sklearn f1_score)."""
    tn = fp = fn = tp = 0
    for yt, yp in zip(yt_list, yp_list):
        yt_b = yt == 1
        yp_b = yp == 1
        tp += int(np.count_nonzero(yt_b & yp_b))
        fp += int(np.count_nonzero((~yt_b) & yp_b))
        fn += int(np.count_nonzero(yt_b & (~yp_b)))
        tn += int(np.count_nonzero((~yt_b) & (~yp_b)))
    iou_floor = tn / max(tn + fp + fn, 1)
    iou_motor = tp / max(tp + fp + fn, 1)
    miou = 0.5 * (iou_floor + iou_motor)
    acc = (tn + tp) / max(tn + fp + fn + tp, 1)
    f1_0 = (2.0 * tn) / max(2 * tn + fp + fn, 1)
    f1_1 = (2.0 * tp) / max(2 * tp + fp + fn, 1)
    mf1 = 0.5 * (f1_0 + f1_1)
    return {
        "IoU_Floor": float(iou_floor),
        "IoU_Motor": float(iou_motor),
        "mIoU": float(miou),
        "Macro_F1": float(mf1),
        "Pixel_Acc": float(acc),
    }


def overlay_vis(im_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Overlay RGB label colors onto BGR frame for display."""
    out = im_bgr[:, :, ::-1].astype(np.float32)
    colors = {0: (60, 230, 151), 1: (255, 86, 111), 255: (112, 146, 224)}
    for v, rgb in colors.items():
        sel = mask_u8 == v
        if np.any(sel):
            out[sel] = out[sel] * 0.55 + np.array(rgb, dtype=np.float32) * 0.45
    return out.astype(np.uint8)


def render_bev_tile(floor_bev: np.ndarray, usable: np.ndarray, g: dict) -> np.ndarray:
    """Render BEV floor + green usable motorcycle regions + component count badges."""
    canvas = np.full((floor_bev.shape[0], floor_bev.shape[1], 3), 24, dtype=np.uint8)
    canvas[floor_bev > 0] = (110, 118, 128)
    canvas[usable > 0] = (60, 230, 151)
    n, _, comp_stats, centroids = cv2.connectedComponentsWithStats(usable)
    for c in range(1, n):
        x, y, w, h, area = comp_stats[c]
        slots = max(1, int(round(area / g["motor_area"])))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 220, 80), 2)
        cx, cy = int(round(centroids[c][0])), int(round(centroids[c][1]))
        cv2.putText(canvas, f"+{slots}", (max(4, cx - 18), max(22, cy + 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main():
    manifest_path = DATA_DIR / "manifest.csv"
    assert manifest_path.exists(), f"Missing {manifest_path}"
    df = pd.read_csv(manifest_path)
    assert len(df) >= 111, f"Expected >= 111 frames, got {len(df)}"
    dates = {s: set(g["date"]) for s, g in df.groupby("split")}
    assert dates["train"].isdisjoint(dates["val"])
    assert dates["train"].isdisjoint(dates["test"])
    assert dates["val"].isdisjoint(dates["test"])
    print(f"Dataset: {len(df)} frames | splits: {df['split'].value_counts().to_dict()}", flush=True)
    print("Anti-leakage check PASSED (0 overlapping dates).", flush=True)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print("\n--- Preparing BEV Geometries & SIFT Alignment ---", flush=True)
    geo, sift, bf = prepare_bev_geometries()

    def load_frames_and_features(sub_df, add_dusk: bool = False):
        items = []
        inls_log = []
        rng_dusk = np.random.default_rng(777) if add_dusk else None
        for _, row in sub_df.iterrows():
            rec = row["recording_id"]
            im = cv2.imread(str(DATA_DIR / row["image_path"]))
            mk = cv2.imread(str(DATA_DIR / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
            roi = mk != 255
            H_bev = None
            gt_free = None
            if rec in geo:
                H_bev, n_inl = compute_crop_to_bev(im, geo[rec], sift, bf)
                inls_log.append((rec, row["timestamp"], n_inl))
                gt_free, _, _ = count_free_spaces(mk, H_bev, geo[rec])
            items.append({
                "row": row,
                "im": im,
                "mk": mk,
                "roi": roi,
                "feats_v2": extract_pcd_features_v2(im),
                "feats_v2_dusk": extract_pcd_features_v2(synthesize_dusk(im, rng_dusk)) if add_dusk else None,
                "H_bev": H_bev,
                "gt_free": gt_free,
            })
        return items, inls_log

    t0 = time.time()
    val_items, val_inls = load_frames_and_features(val_df, add_dusk=True)
    test_items, test_inls = load_frames_and_features(test_df, add_dusk=False)
    min_inl = min(x[2] for x in (val_inls + test_inls))
    print(f"Aligned BEV for {len(val_inls)} val + {len(test_inls)} test frames (min_inliers={min_inl}) "
          f"& extracted {len(FEATURE_NAMES_V2)} v2 features in {time.time()-t0:.1f}s", flush=True)

    # Sample train (6k orig + 3k dusk/frame) & val early-stopping (6k orig + 2k dusk/frame)
    t_s = time.time()
    X_train, y_train = build_augmented_samples(train_df, orig_per_frame=6000, dusk_per_frame=3000, seed=42)
    X_val_s, y_val_s = build_augmented_samples(val_df, orig_per_frame=6000, dusk_per_frame=2000, seed=123)
    print(f"Sampled X_train={X_train.shape} (floor={(y_train==0).sum():,}, motor={(y_train==1).sum():,}), "
          f"X_val_s={X_val_s.shape} in {time.time()-t_s:.1f}s", flush=True)

    # Train Random Forest v2
    print("\n--- Training Random Forest v2 ---", flush=True)
    t_rf = time.time()
    rf_v2 = RandomForestClassifier(
        n_estimators=250,
        max_depth=24,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    rf_v2.fit(X_train, y_train)
    rf_train_s = time.time() - t_rf
    print(f"Random Forest v2 trained in {rf_train_s:.1f}s", flush=True)

    # Train XGBoost v2
    print("\n--- Training XGBoost v2 ---", flush=True)
    t_xgb = time.time()
    xgb_v2 = XGBClassifier(
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
    xgb_v2.fit(X_train, y_train, eval_set=[(X_val_s, y_val_s)], verbose=False)
    xgb_train_s = time.time() - t_xgb
    print(f"XGBoost v2 trained in {xgb_train_s:.1f}s (best_iteration={xgb_v2.best_iteration})", flush=True)

    rf_v2_path = OUT_DIR / "rf_v2_model.joblib"
    xgb_v2_path = OUT_DIR / "xgb_v2_model.joblib"
    joblib.dump({"model": rf_v2, "features": FEATURE_NAMES_V2}, rf_v2_path, compress=3)
    joblib.dump({"model": xgb_v2, "features": FEATURE_NAMES_V2}, xgb_v2_path, compress=3)

    # Tune v2 on val (0.5 * orig + 0.5 * synthetic dusk) & evaluate on held-out test
    v2_models = {
        "RF_v2": (rf_v2, rf_train_s, rf_v2_path.stat().st_size / (1024 * 1024)),
        "XGB_v2": (xgb_v2, xgb_train_s, xgb_v2_path.stat().st_size / (1024 * 1024)),
    }

    all_results = {}
    vis_masks = {"RF_v2": {}, "XGB_v2": {}, "RF_v1": {}, "XGB_v1": {}}
    bev_tiles = {"GT": {}, "RF_v2": {}, "XGB_v2": {}}

    for name, (mdl, train_s, size_mb) in v2_models.items():
        val_probs_orig = [predict_prob_map(mdl, it["feats_v2"], it["roi"]) for it in val_items]
        val_probs_dusk = [predict_prob_map(mdl, it["feats_v2_dusk"], it["roi"]) for it in val_items]
        best_cfg = None
        for sigma in [0.0, 1.5]:
            b_orig = [smooth_prob_roi(p, it["roi"], sigma) for p, it in zip(val_probs_orig, val_items)]
            b_dusk = [smooth_prob_roi(p, it["roi"], sigma) for p, it in zip(val_probs_dusk, val_items)]
            for thr in [0.40, 0.45, 0.50, 0.55, 0.60]:
                for ksize in [7, 11, 15]:
                    yt_l, yp_o_l, yp_d_l = [], [], []
                    for it, po, pd_ in zip(val_items, b_orig, b_dusk):
                        yt_l.append(it["mk"][it["roi"]])
                        yp_o_l.append(apply_morphology((po >= thr).astype(np.uint8), it["roi"], ksize)[it["roi"]])
                        yp_d_l.append(apply_morphology((pd_ >= thr).astype(np.uint8), it["roi"], ksize)[it["roi"]])
                    m_orig = compute_fast_metrics(yt_l, yp_o_l)
                    m_dusk = compute_fast_metrics(yt_l, yp_d_l)
                    comb_miou = 0.5 * m_orig["mIoU"] + 0.5 * m_dusk["mIoU"]
                    if best_cfg is None or comb_miou > best_cfg["comb_mIoU"]:
                        best_cfg = {
                            "blur_sigma": sigma, "thr": thr, "morph_k": ksize,
                            "orig_mIoU": m_orig["mIoU"], "dusk_mIoU": m_dusk["mIoU"],
                            "comb_mIoU": comb_miou,
                        }

        # Check BEV MAE on val 08-10-11 with best_cfg
        val_errs = []
        for it, prob in zip(val_items, val_probs_orig):
            if it["H_bev"] is None:
                continue
            ps = smooth_prob_roi(prob, it["roi"], best_cfg["blur_sigma"])
            pred = apply_morphology((ps >= best_cfg["thr"]).astype(np.uint8), it["roi"], best_cfg["morph_k"])
            pred_full = np.where(it["roi"], pred, 255).astype(np.uint8)
            p_free, _, _ = count_free_spaces(pred_full, it["H_bev"], geo[it["row"]["recording_id"]])
            val_errs.append(abs(p_free - it["gt_free"]))
        val_mae = float(np.mean(val_errs)) if val_errs else float("nan")
        print(f"\n[{name}] Best Val cfg: sigma={best_cfg['blur_sigma']}, thr={best_cfg['thr']}, "
              f"morph_k={best_cfg['morph_k']} | Val orig={best_cfg['orig_mIoU']:.4f} dusk={best_cfg['dusk_mIoU']:.4f} "
              f"comb={best_cfg['comb_mIoU']:.4f} | Val BEV MAE={val_mae:.2f}", flush=True)

        # Locked held-out test evaluation
        t_inf = time.time()
        test_probs = [predict_prob_map(mdl, it["feats_v2"], it["roi"]) for it in test_items]
        inf_ms = (time.time() - t_inf) * 1000.0 / len(test_items)

        all_results[name] = evaluate_test_predictions(
            test_items, test_probs, best_cfg["blur_sigma"], best_cfg["thr"], best_cfg["morph_k"],
            geo, train_s, inf_ms, size_mb, vis_masks[name], bev_tiles[name], bev_tiles["GT"],
        )

    # Evaluate v1 models (loaded from disk) on identical 26 test frames for BOTH mIoU and BEV MAE
    print("\n--- Evaluating v1 Models (loaded from disk) for BEV MAE & direct v1 vs v2 comparison ---", flush=True)
    v1_summary = json.loads((V1_DIR / "benchmark_summary.json").read_text(encoding="utf-8"))
    rf_v1 = joblib.load(V1_DIR / "rf_model.joblib")["model"]
    xgb_v1 = joblib.load(V1_DIR / "xgb_model.joblib")["model"]

    t_v1_fe = time.time()
    test_feats_v1 = [extract_pcd_features_v1(it["im"]) for it in test_items]
    _ = time.time() - t_v1_fe

    for v1_key, v1_sum_key, mdl in [("RF_v1", "RandomForest", rf_v1), ("XGB_v1", "XGBoost", xgb_v1)]:
        cfg_v1 = v1_summary[v1_sum_key]["best_val_morph"]
        t_inf = time.time()
        probs_v1 = [predict_prob_map(mdl, fv1, it["roi"]) for fv1, it in zip(test_feats_v1, test_items)]
        inf_ms = (time.time() - t_inf) * 1000.0 / len(test_items)
        all_results[v1_key] = evaluate_test_predictions(
            test_items, probs_v1, 0.0, cfg_v1["thr"], cfg_v1["morph_k"],
            geo, v1_summary[v1_sum_key]["train_sec"], v1_summary[v1_sum_key]["infer_ms_per_frame"],
            v1_summary[v1_sum_key]["model_size_mb"], vis_masks[v1_key], None, None,
        )

    # Build final benchmark table
    table_rows = [
        {
            "Model": "Naive Bayes (Baseline Repo Dinda)",
            "mIoU": 0.6470, "IoU_Floor": 0.5370, "IoU_Motor": 0.7570,
            "Macro_F1": np.nan, "Pixel_Acc": 0.8110,
            "Dusk_mIoU": np.nan, "MAE_Free": 12.81, "RMSE_Free": np.nan,
            "Train_s": np.nan, "Infer_ms": np.nan, "Size_MB": np.nan,
        },
    ]
    order = [
        ("RF_v1", "Random Forest v1 (22 feat)"),
        ("XGB_v1", "XGBoost v1 (22 feat)"),
        ("RF_v2", "Random Forest v2 (+dusk+norm)"),
        ("XGB_v2", "XGBoost v2 (+dusk+norm)"),
    ]
    for key, label in order:
        r = all_results[key]
        m = r["overall"]
        dusk_m = r["per_video"]["04-04-11-2026-evening-top-keiza"]["mIoU"]
        table_rows.append({
            "Model": label,
            "mIoU": round(m["mIoU"], 4),
            "IoU_Floor": round(m["IoU_Floor"], 4),
            "IoU_Motor": round(m["IoU_Motor"], 4),
            "Macro_F1": round(m["Macro_F1"], 4),
            "Pixel_Acc": round(m["Pixel_Acc"], 4),
            "Dusk_mIoU": round(dusk_m, 4),
            "MAE_Free": round(r["bev_mae"], 2),
            "RMSE_Free": round(r["bev_rmse"], 2),
            "Train_s": round(r["train_sec"], 2),
            "Infer_ms": round(r["infer_ms"], 1),
            "Size_MB": round(r["size_mb"], 2),
        })

    res_df = pd.DataFrame(table_rows)
    print("\n================ FINAL BENCHMARK: SEGMENTATION + BEV FREE SPACES (26 HELD-OUT TEST FRAMES) ================")
    print(res_df.to_string(index=False))
    res_df.to_csv(OUT_DIR / "v2_benchmark_table.csv", index=False)

    print("\n--- Per-Video Test Breakdown (mIoU & BEV Free Spaces) ---")
    for rec in sorted(all_results["RF_v2"]["per_video"].keys()):
        parts = []
        gt_f = all_results["RF_v2"]["per_video"][rec]["gt_free_mean"]
        for key in ("RF_v1", "RF_v2", "XGB_v1", "XGB_v2"):
            pv = all_results[key]["per_video"][rec]
            parts.append(f"{key}: mIoU={pv['mIoU']:.4f} pred={pv['pred_free_mean']:.1f} MAE={pv['bev_mae']:.2f}")
        print(f"  {rec:38s} (GT free avg={gt_f:4.1f}) | " + " | ".join(parts))

    (OUT_DIR / "v2_benchmark_summary.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # Visual Plots
    save_feature_importance(rf_v2, xgb_v2)
    save_test_comparison_grid(test_items, vis_masks, all_results)
    save_bev_grid(test_items, bev_tiles, all_results)
    print(f"\nSaved all v2 outputs to {OUT_DIR}/", flush=True)


def evaluate_test_predictions(
    test_items, probs, sigma, thr, morph_k, geo, train_s, inf_ms, size_mb, vis_store, bev_store, gt_bev_store
):
    yt_all, yp_all = [], []
    bev_errors = []
    per_rec = {}

    for it, prob in zip(test_items, probs):
        ps = smooth_prob_roi(prob, it["roi"], sigma)
        pred = apply_morphology((ps >= thr).astype(np.uint8), it["roi"], morph_k)
        pred_full = np.where(it["roi"], pred, 255).astype(np.uint8)

        rec = it["row"]["recording_id"]
        g = geo[rec]
        p_free, floor_bev, usable = count_free_spaces(pred_full, it["H_bev"], g)
        gt_free = it["gt_free"]
        abs_err = abs(p_free - gt_free)

        yt_all.append(it["mk"][it["roi"]])
        yp_all.append(pred[it["roi"]])
        bev_errors.append(abs_err)

        bucket = per_rec.setdefault(rec, {"yt": [], "yp": [], "gt_f": [], "pr_f": [], "err": []})
        bucket["yt"].append(it["mk"][it["roi"]])
        bucket["yp"].append(pred[it["roi"]])
        bucket["gt_f"].append(gt_free)
        bucket["pr_f"].append(p_free)
        bucket["err"].append(abs_err)

        if it["row"]["timestamp"] == "00-45":
            vis_store[rec] = (it["row"]["frame_id"], it["im"], it["mk"], pred_full, p_free, gt_free)
            if bev_store is not None:
                bev_store[rec] = (p_free, render_bev_tile(floor_bev, usable, g))
            if gt_bev_store is not None and rec not in gt_bev_store:
                _, gt_fb, gt_us = count_free_spaces(it["mk"], it["H_bev"], g)
                gt_bev_store[rec] = (gt_free, render_bev_tile(gt_fb, gt_us, g))

    overall = compute_fast_metrics(yt_all, yp_all)
    per_video_out = {}
    for rec, b in sorted(per_rec.items()):
        pm = compute_fast_metrics(b["yt"], b["yp"])
        errs = np.array(b["err"], dtype=np.float32)
        per_video_out[rec] = {
            **pm,
            "gt_free_mean": float(np.mean(b["gt_f"])),
            "pred_free_mean": float(np.mean(b["pr_f"])),
            "bev_mae": float(np.mean(errs)),
            "bev_rmse": float(np.sqrt(np.mean(errs * errs))),
        }

    errs_all = np.array(bev_errors, dtype=np.float32)
    return {
        "train_sec": float(train_s),
        "infer_ms": float(inf_ms),
        "size_mb": float(size_mb),
        "config": {"blur_sigma": float(sigma), "thr": float(thr), "morph_k": int(morph_k)},
        "overall": overall,
        "bev_mae": float(np.mean(errs_all)),
        "bev_rmse": float(np.sqrt(np.mean(errs_all * errs_all))),
        "per_video": per_video_out,
    }


def save_feature_importance(rf_v2, xgb_v2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (title, mdl, col) in zip(axes, [("Random Forest v2", rf_v2, "#2563eb"), ("XGBoost v2", xgb_v2, "#ea580c")]):
        imp = pd.Series(mdl.feature_importances_, index=FEATURE_NAMES_V2).sort_values(ascending=True).tail(20)
        imp.plot(kind="barh", ax=ax, color=col)
        ax.set_title(f"{title} Top-20 Features")
        ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "v2_feature_importance.png", dpi=150)
    plt.close(fig)


def save_test_comparison_grid(test_items, vis_masks, all_results):
    recs = sorted(vis_masks["RF_v2"].keys())
    fig, axes = plt.subplots(len(recs), 5, figsize=(24, 4.3 * len(recs)))
    if len(recs) == 1:
        axes = [axes]
    for row_ax, rec in zip(axes, recs):
        fid, im, mk, rf2_m, rf2_free, gt_free = vis_masks["RF_v2"][rec]
        _, _, _, rf1_m, rf1_free, _ = vis_masks["RF_v1"][rec]
        _, _, _, xgb2_m, xgb2_free, _ = vis_masks["XGB_v2"][rec]

        rf1_miou = all_results["RF_v1"]["per_video"][rec]["mIoU"]
        rf2_miou = all_results["RF_v2"]["per_video"][rec]["mIoU"]
        xgb2_miou = all_results["XGB_v2"]["per_video"][rec]["mIoU"]

        row_ax[0].imshow(im[:, :, ::-1]); row_ax[0].set_title(f"Original: {fid}"); row_ax[0].axis("off")
        row_ax[1].imshow(overlay_vis(im, mk)); row_ax[1].set_title(f"Ground Truth (free={gt_free})"); row_ax[1].axis("off")
        row_ax[2].imshow(overlay_vis(im, rf1_m)); row_ax[2].set_title(f"RF v1 (mIoU={rf1_miou:.3f}, free={rf1_free})"); row_ax[2].axis("off")
        row_ax[3].imshow(overlay_vis(im, rf2_m)); row_ax[3].set_title(f"RF v2 (mIoU={rf2_miou:.3f}, free={rf2_free})"); row_ax[3].axis("off")
        row_ax[4].imshow(overlay_vis(im, xgb2_m)); row_ax[4].set_title(f"XGB v2 (mIoU={xgb2_miou:.3f}, free={xgb2_free})"); row_ax[4].axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "v2_test_comparison_grid.png", dpi=150)
    plt.close(fig)


def save_bev_grid(test_items, bev_tiles, all_results):
    recs = sorted(bev_tiles["GT"].keys())
    fig, axes = plt.subplots(len(recs), 3, figsize=(15, 4.8 * len(recs)))
    if len(recs) == 1:
        axes = [axes]
    for row_ax, rec in zip(axes, recs):
        gt_free, gt_img = bev_tiles["GT"][rec]
        rf_free, rf_img = bev_tiles["RF_v2"][rec]
        xg_free, xg_img = bev_tiles["XGB_v2"][rec]
        short_rec = rec[:24]
        row_ax[0].imshow(gt_img); row_ax[0].set_title(f"{short_rec} GT BEV (free={gt_free})"); row_ax[0].axis("off")
        row_ax[1].imshow(rf_img); row_ax[1].set_title(f"RF v2 BEV (free={rf_free})"); row_ax[1].axis("off")
        row_ax[2].imshow(xg_img); row_ax[2].set_title(f"XGB v2 BEV (free={xg_free})"); row_ax[2].axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "v2_bev_free_spaces_grid.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

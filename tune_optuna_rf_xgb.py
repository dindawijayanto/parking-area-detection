import json
import time
from pathlib import Path
import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from train_rf_vs_xgb_v2 import (
    DATA_DIR,
    FEATURE_NAMES_V2,
    build_augmented_samples,
    compute_crop_to_bev,
    compute_fast_metrics,
    count_free_spaces,
    extract_pcd_features_v2,
    overlay_vis,
    predict_prob_map,
    prepare_bev_geometries,
    render_bev_tile,
    sample_pixels,
    smooth_prob_roi,
    synthesize_dusk,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

V2_SUMMARY_PATH = Path("results/rf_vs_xgb_v2/v2_benchmark_summary.json")
OUT_DIR = Path("results/optuna_rf_xgb")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fast_pixel_miou(y_true: np.ndarray, probs: np.ndarray, thr: float) -> float:
    """Compute vectorized binary mIoU (floor=0, motor=1) in <1ms."""
    yp = probs >= thr
    yt = y_true == 1
    tp = int(np.count_nonzero(yt & yp))
    fp = int(np.count_nonzero((~yt) & yp))
    fn = int(np.count_nonzero(yt & (~yp)))
    tn = len(y_true) - tp - fp - fn
    iou0 = tn / max(tn + fp + fn, 1)
    iou1 = tp / max(tp + fp + fn, 1)
    return 0.5 * (iou0 + iou1)


def best_balanced_miou(y_vo: np.ndarray, p_vo: np.ndarray, y_vd: np.ndarray, p_vd: np.ndarray) -> float:
    """Best mean of original-val and dusk-val mIoU over a lightweight threshold sweep."""
    best = 0.0
    for thr in (0.40, 0.45, 0.50, 0.55):
        m_o = fast_pixel_miou(y_vo, p_vo, thr)
        m_d = fast_pixel_miou(y_vd, p_vd, thr)
        score = 0.5 * (m_o + m_d)
        if score > best:
            best = score
    return best


def apply_asym_morphology(pred_bin: np.ndarray, roi_mask: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    """Apply elliptical open then close morphology inside roi_mask."""
    out = pred_bin.astype(np.uint8)
    if open_k > 0:
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, ko)
    if close_k > 0:
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kc)
    return ((out > 0) & roi_mask).astype(np.uint8)


def load_eval_frames(sub_df: pd.DataFrame, geo: dict, sift, bf, add_dusk: bool = False):
    """Load frames, align BEV homographies, and precompute 26 v2 features."""
    items = []
    inls = []
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
            inls.append(n_inl)
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
    return items, inls


def sample_val_matrices(val_items: list[dict]):
    """Build separate orig and dusk validation pixel arrays directly from precomputed val features."""
    rng_o = np.random.default_rng(123)
    rng_d = np.random.default_rng(456)
    Xo_l, yo_l, Xd_l, yd_l = [], [], [], []
    for it in val_items:
        xo, yo = sample_pixels(it["feats_v2"], it["mk"], 6000, rng_o)
        xd, yd = sample_pixels(it["feats_v2_dusk"], it["mk"], 6000, rng_d)
        Xo_l.extend(xo)
        yo_l.extend(yo)
        Xd_l.extend(xd)
        yd_l.extend(yd)
    return np.vstack(Xo_l), np.concatenate(yo_l), np.vstack(Xd_l), np.concatenate(yd_l)


def tune_postproc_on_val(model, val_items: list[dict], geo: dict):
    """Tune Gaussian smoothing, probability threshold, and asymmetric morphology strictly on val."""
    val_probs_orig = [predict_prob_map(model, it["feats_v2"], it["roi"]) for it in val_items]
    val_probs_dusk = [predict_prob_map(model, it["feats_v2_dusk"], it["roi"]) for it in val_items]

    best = None
    morph_pairs = [(11, 11), (15, 11), (15, 15), (19, 11), (19, 15)]
    for sigma in (0.0, 1.5, 2.5):
        b_orig = [smooth_prob_roi(p, it["roi"], sigma) for p, it in zip(val_probs_orig, val_items)]
        b_dusk = [smooth_prob_roi(p, it["roi"], sigma) for p, it in zip(val_probs_dusk, val_items)]
        for thr in (0.42, 0.46, 0.50, 0.54, 0.58):
            for open_k, close_k in morph_pairs:
                yt_l, yp_o_l, yp_d_l = [], [], []
                val_errs = []
                for it, po, pd_ in zip(val_items, b_orig, b_dusk):
                    pred_o = apply_asym_morphology(po >= thr, it["roi"], open_k, close_k)
                    pred_d = apply_asym_morphology(pd_ >= thr, it["roi"], open_k, close_k)
                    yt_l.append(it["mk"][it["roi"]])
                    yp_o_l.append(pred_o[it["roi"]])
                    yp_d_l.append(pred_d[it["roi"]])
                    if it["H_bev"] is not None:
                        pred_full = np.where(it["roi"], pred_o, 255).astype(np.uint8)
                        p_free, _, _ = count_free_spaces(pred_full, it["H_bev"], geo[it["row"]["recording_id"]])
                        val_errs.append(abs(p_free - it["gt_free"]))
                m_orig = compute_fast_metrics(yt_l, yp_o_l)
                m_dusk = compute_fast_metrics(yt_l, yp_d_l)
                val_mae = float(np.mean(val_errs)) if val_errs else 0.0
                joint = 0.5 * m_orig["mIoU"] + 0.5 * m_dusk["mIoU"] - 0.004 * val_mae
                if best is None or joint > best["joint_score"]:
                    best = {
                        "blur_sigma": float(sigma),
                        "thr": float(thr),
                        "open_k": int(open_k),
                        "close_k": int(close_k),
                        "orig_mIoU": m_orig["mIoU"],
                        "dusk_mIoU": m_dusk["mIoU"],
                        "val_bev_mae": val_mae,
                        "joint_score": float(joint),
                    }
    return best


def evaluate_on_test(model, test_items: list[dict], cfg: dict, geo: dict, train_s: float, size_mb: float):
    """Evaluate locked model + post-proc config once on 26 held-out test frames."""
    t_inf = time.time()
    probs = [predict_prob_map(model, it["feats_v2"], it["roi"]) for it in test_items]
    preds = []
    for it, prob in zip(test_items, probs):
        ps = smooth_prob_roi(prob, it["roi"], cfg["blur_sigma"])
        pred = apply_asym_morphology(ps >= cfg["thr"], it["roi"], cfg["open_k"], cfg["close_k"])
        preds.append(pred)
    infer_ms = (time.time() - t_inf) * 1000.0 / len(test_items)

    yt_all, yp_all, bev_errors = [], [], []
    per_rec = {}
    vis_store = {}
    bev_store = {}
    gt_bev_store = {}

    for it, pred in zip(test_items, preds):
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
            bev_store[rec] = (p_free, render_bev_tile(floor_bev, usable, g))
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
    result = {
        "train_sec": float(train_s),
        "infer_ms": float(infer_ms),
        "size_mb": float(size_mb),
        "config": cfg,
        "overall": overall,
        "bev_mae": float(np.mean(errs_all)),
        "bev_rmse": float(np.sqrt(np.mean(errs_all * errs_all))),
        "per_video": per_video_out,
    }
    return result, vis_store, bev_store, gt_bev_store


def save_optuna_history(xgb_study: optuna.Study, rf_study: optuna.Study):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, title, study, col in zip(
        axes,
        ["XGBoost Optuna (35 Trials)", "Random Forest Optuna (18 Trials)"],
        [xgb_study, rf_study],
        ["#ea580c", "#2563eb"],
    ):
        vals = [t.value for t in study.trials if t.value is not None]
        xs = list(range(1, len(vals) + 1))
        best_curve = np.maximum.accumulate(vals)
        ax.scatter(xs, vals, color=col, alpha=0.65, label="Trial Val mIoU")
        ax.plot(xs, best_curve, color="#111827", linewidth=2.2, label="Best So Far")
        ax.set_title(f"{title} — Best={study.best_value:.4f}")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Balanced Val mIoU (Orig+Dusk)")
        ax.grid(alpha=0.25)
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "optuna_history.png", dpi=150)
    plt.close(fig)


def save_test_comparison_grid(vis_rf: dict, vis_xgb: dict, res_rf: dict, res_xgb: dict):
    recs = sorted(vis_rf.keys())
    fig, axes = plt.subplots(len(recs), 4, figsize=(20, 4.3 * len(recs)))
    if len(recs) == 1:
        axes = [axes]
    for row_ax, rec in zip(axes, recs):
        fid, im, mk, rf_m, rf_free, gt_free = vis_rf[rec]
        _, _, _, xgb_m, xgb_free, _ = vis_xgb[rec]
        rf_miou = res_rf["per_video"][rec]["mIoU"]
        xg_miou = res_xgb["per_video"][rec]["mIoU"]

        row_ax[0].imshow(im[:, :, ::-1]); row_ax[0].set_title(f"Original: {fid}"); row_ax[0].axis("off")
        row_ax[1].imshow(overlay_vis(im, mk)); row_ax[1].set_title(f"Ground Truth (free={gt_free})"); row_ax[1].axis("off")
        row_ax[2].imshow(overlay_vis(im, rf_m)); row_ax[2].set_title(f"RF v3 Optuna (mIoU={rf_miou:.3f}, free={rf_free})"); row_ax[2].axis("off")
        row_ax[3].imshow(overlay_vis(im, xgb_m)); row_ax[3].set_title(f"XGB v3 Optuna (mIoU={xg_miou:.3f}, free={xgb_free})"); row_ax[3].axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "optuna_test_comparison_grid.png", dpi=150)
    plt.close(fig)


def save_bev_grid(bev_gt: dict, bev_rf: dict, bev_xgb: dict):
    recs = sorted(bev_gt.keys())
    fig, axes = plt.subplots(len(recs), 3, figsize=(15, 4.8 * len(recs)))
    if len(recs) == 1:
        axes = [axes]
    for row_ax, rec in zip(axes, recs):
        gt_free, gt_img = bev_gt[rec]
        rf_free, rf_img = bev_rf[rec]
        xg_free, xg_img = bev_xgb[rec]
        short_rec = rec[:24]
        row_ax[0].imshow(gt_img); row_ax[0].set_title(f"{short_rec} GT BEV (free={gt_free})"); row_ax[0].axis("off")
        row_ax[1].imshow(rf_img); row_ax[1].set_title(f"RF v3 Optuna BEV (free={rf_free})"); row_ax[1].axis("off")
        row_ax[2].imshow(xg_img); row_ax[2].set_title(f"XGB v3 Optuna BEV (free={xg_free})"); row_ax[2].axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "optuna_bev_free_spaces_grid.png", dpi=150)
    plt.close(fig)


def main():
    # Stage 1: Load & Precompute
    df = pd.read_csv(DATA_DIR / "manifest.csv")
    dates = {s: set(g["date"]) for s, g in df.groupby("split")}
    assert dates["train"].isdisjoint(dates["val"])
    assert dates["train"].isdisjoint(dates["test"])
    assert dates["val"].isdisjoint(dates["test"])
    print(f"=== STAGE 1: Precompute & Anti-Leakage Check ({len(df)} frames) ===", flush=True)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    t0 = time.time()
    geo, sift, bf = prepare_bev_geometries()
    val_items, val_inls = load_eval_frames(val_df, geo, sift, bf, add_dusk=True)
    test_items, test_inls = load_eval_frames(test_df, geo, sift, bf, add_dusk=False)
    assert min(val_inls + test_inls) >= 200
    print(f"BEV aligned (min_inliers={min(val_inls + test_inls)}) & 26 features extracted for val/test in {time.time()-t0:.1f}s", flush=True)

    t_s = time.time()
    X_tr_tune, y_tr_tune = build_augmented_samples(train_df, orig_per_frame=4500, dusk_per_frame=2500, seed=42)
    X_tr_full, y_tr_full = build_augmented_samples(train_df, orig_per_frame=6000, dusk_per_frame=3000, seed=99)
    X_vo, y_vo, X_vd, y_vd = sample_val_matrices(val_items)
    X_val_es = np.vstack([X_vo, X_vd])
    y_val_es = np.concatenate([y_vo, y_vd])
    print(f"Sampled X_tr_tune={X_tr_tune.shape}, X_tr_full={X_tr_full.shape}, X_val_es={X_val_es.shape} in {time.time()-t_s:.1f}s", flush=True)

    # Stage 2: Optuna Study 1 — XGBoost (35 trials)
    print("\n=== STAGE 2: Optuna XGBoost (35 trials) ===", flush=True)
    t_xgb_study = time.time()
    best_xgb_score = [-1.0]

    def xgb_objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.14, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 60.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.85, 1.15),
        }
        mdl = XGBClassifier(
            **params,
            n_estimators=700,
            tree_method="hist",
            eval_metric="logloss",
            early_stopping_rounds=35,
            n_jobs=-1,
            random_state=42,
        )
        mdl.fit(X_tr_tune, y_tr_tune, eval_set=[(X_val_es, y_val_es)], verbose=False)
        p_vo = mdl.predict_proba(X_vo)[:, 1]
        p_vd = mdl.predict_proba(X_vd)[:, 1]
        score = best_balanced_miou(y_vo, p_vo, y_vd, p_vd)
        if score > best_xgb_score[0] or (trial.number + 1) % 5 == 0:
            is_best = score > best_xgb_score[0]
            if is_best:
                best_xgb_score[0] = score
            tag = " *NEW BEST*" if is_best else ""
            print(f"  [XGB trial {trial.number+1:2d}/35] val_bal_mIoU={score:.4f} (best={best_xgb_score[0]:.4f}) "
                  f"iter={mdl.best_iteration} depth={params['max_depth']} lr={params['learning_rate']:.3f}{tag}", flush=True)
        return score

    xgb_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    xgb_study.optimize(xgb_objective, n_trials=35)
    print(f"XGBoost Optuna done in {time.time()-t_xgb_study:.1f}s | best={xgb_study.best_value:.4f} | params={xgb_study.best_params}", flush=True)

    # Stage 3: Optuna Study 2 — Random Forest (18 trials)
    print("\n=== STAGE 3: Optuna Random Forest (18 trials) ===", flush=True)
    t_rf_study = time.time()
    best_rf_score = [-1.0]

    def rf_objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 14, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 4, 32),
            "max_features": trial.suggest_float("max_features", 0.15, 0.55),
            "max_samples": trial.suggest_float("max_samples", 0.50, 0.95),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }
        mdl = RandomForestClassifier(
            **params,
            n_estimators=90,
            class_weight="balanced_subsample",
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
        )
        mdl.fit(X_tr_tune, y_tr_tune)
        p_vo = mdl.predict_proba(X_vo)[:, 1]
        p_vd = mdl.predict_proba(X_vd)[:, 1]
        score = best_balanced_miou(y_vo, p_vo, y_vd, p_vd)
        if score > best_rf_score[0] or (trial.number + 1) % 3 == 0:
            is_best = score > best_rf_score[0]
            if is_best:
                best_rf_score[0] = score
            tag = " *NEW BEST*" if is_best else ""
            print(f"  [RF  trial {trial.number+1:2d}/18] val_bal_mIoU={score:.4f} (best={best_rf_score[0]:.4f}) "
                  f"depth={params['max_depth']} leaf={params['min_samples_leaf']} feat={params['max_features']:.2f}{tag}", flush=True)
        return score

    rf_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    rf_study.optimize(rf_objective, n_trials=18)
    print(f"RF Optuna done in {time.time()-t_rf_study:.1f}s | best={rf_study.best_value:.4f} | params={rf_study.best_params}", flush=True)

    # Stage 4: Full Refit on X_tr_full (603k)
    print("\n=== STAGE 4: Full Refit of Tuned Winners (603k samples) ===", flush=True)
    t_rf_fit = time.time()
    rf_opt = RandomForestClassifier(
        **rf_study.best_params,
        n_estimators=300,
        class_weight="balanced_subsample",
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
    )
    rf_opt.fit(X_tr_full, y_tr_full)
    rf_fit_s = time.time() - t_rf_fit

    t_xgb_fit = time.time()
    xgb_opt = XGBClassifier(
        **xgb_study.best_params,
        n_estimators=1200,
        tree_method="hist",
        eval_metric="logloss",
        early_stopping_rounds=45,
        n_jobs=-1,
        random_state=42,
    )
    xgb_opt.fit(X_tr_full, y_tr_full, eval_set=[(X_val_es, y_val_es)], verbose=False)
    xgb_fit_s = time.time() - t_xgb_fit
    print(f"Refit done | RF_Optuna={rf_fit_s:.1f}s | XGB_Optuna={xgb_fit_s:.1f}s (best_iter={xgb_opt.best_iteration})", flush=True)

    rf_path = OUT_DIR / "rf_optuna_model.joblib"
    xgb_path = OUT_DIR / "xgb_optuna_model.joblib"
    joblib.dump({"model": rf_opt, "features": FEATURE_NAMES_V2, "params": rf_study.best_params}, rf_path, compress=3)
    joblib.dump({"model": xgb_opt, "features": FEATURE_NAMES_V2, "params": xgb_study.best_params}, xgb_path, compress=3)
    rf_mb = rf_path.stat().st_size / (1024 * 1024)
    xgb_mb = xgb_path.stat().st_size / (1024 * 1024)

    # Stage 5: Asymmetric Morphology + Threshold Tuning on Full-Frame Val
    print("\n=== STAGE 5: Asymmetric Morphology & Threshold Tuning (Val) ===", flush=True)
    rf_val_cfg = tune_postproc_on_val(rf_opt, val_items, geo)
    xgb_val_cfg = tune_postproc_on_val(xgb_opt, val_items, geo)
    print(f"  [RF_Optuna  Val] {rf_val_cfg}", flush=True)
    print(f"  [XGB_Optuna Val] {xgb_val_cfg}", flush=True)

    # Stage 6: Locked Held-Out Test Evaluation & Deliverables
    print("\n=== STAGE 6: Held-Out Test Evaluation (26 frames) ===", flush=True)
    res_rf, vis_rf, bev_rf, bev_gt = evaluate_on_test(rf_opt, test_items, rf_val_cfg, geo, rf_fit_s, rf_mb)
    res_xgb, vis_xgb, bev_xgb, _ = evaluate_on_test(xgb_opt, test_items, xgb_val_cfg, geo, xgb_fit_s, xgb_mb)

    v2_prev = json.loads(V2_SUMMARY_PATH.read_text(encoding="utf-8"))
    all_models = {
        "RF_v1": ("Random Forest v1 (22 feat)", v2_prev["RF_v1"]),
        "XGB_v1": ("XGBoost v1 (22 feat)", v2_prev["XGB_v1"]),
        "RF_v2": ("Random Forest v2 (26 feat manual)", v2_prev["RF_v2"]),
        "XGB_v2": ("XGBoost v2 (26 feat manual)", v2_prev["XGB_v2"]),
        "RF_v3_Optuna": ("Random Forest v3 (26 feat Optuna)", res_rf),
        "XGB_v3_Optuna": ("XGBoost v3 (26 feat Optuna)", res_xgb),
    }

    rows = [
        {
            "Model": "Naive Bayes (Baseline Repo Dinda)",
            "mIoU": 0.6470, "IoU_Floor": 0.5370, "IoU_Motor": 0.7570,
            "Macro_F1": np.nan, "Pixel_Acc": 0.8110,
            "Dusk_mIoU": np.nan, "MAE_Free": 12.81, "RMSE_Free": np.nan,
            "Train_s": np.nan, "Infer_ms": np.nan, "Size_MB": np.nan,
        },
    ]
    for _, label, r in [(k, lbl, d) for k, (lbl, d) in all_models.items()]:
        m = r["overall"]
        dusk_m = r["per_video"]["04-04-11-2026-evening-top-keiza"]["mIoU"]
        rows.append({
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

    table_df = pd.DataFrame(rows)
    print("\n================ COMPLETE 7-ROW BENCHMARK (26 HELD-OUT TEST FRAMES) ================")
    print(table_df.to_string(index=False))
    table_df.to_csv(OUT_DIR / "optuna_benchmark_table.csv", index=False)

    print("\n--- Per-Video Held-Out Test Breakdown (v2 manual vs v3 Optuna) ---")
    for rec in sorted(res_rf["per_video"].keys()):
        gt_f = res_rf["per_video"][rec]["gt_free_mean"]
        rf2 = v2_prev["RF_v2"]["per_video"][rec]
        xg2 = v2_prev["XGB_v2"]["per_video"][rec]
        rf3 = res_rf["per_video"][rec]
        xg3 = res_xgb["per_video"][rec]
        print(
            f"  {rec:36s} (GT={gt_f:4.1f}) | "
            f"RF_v2={rf2['mIoU']:.4f} MAE={rf2['bev_mae']:.2f} | RF_Opt={rf3['mIoU']:.4f} pred={rf3['pred_free_mean']:.1f} MAE={rf3['bev_mae']:.2f} | "
            f"XGB_v2={xg2['mIoU']:.4f} MAE={xg2['bev_mae']:.2f} | XGB_Opt={xg3['mIoU']:.4f} pred={xg3['pred_free_mean']:.1f} MAE={xg3['bev_mae']:.2f}"
        )

    summary_payload = {
        "xgb_optuna": {
            "n_trials": 35,
            "best_val_bal_miou": float(xgb_study.best_value),
            "best_params": xgb_study.best_params,
            "best_iteration": int(xgb_opt.best_iteration),
            "test_results": res_xgb,
        },
        "rf_optuna": {
            "n_trials": 18,
            "best_val_bal_miou": float(rf_study.best_value),
            "best_params": rf_study.best_params,
            "test_results": res_rf,
        },
    }
    (OUT_DIR / "optuna_benchmark_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    save_optuna_history(xgb_study, rf_study)
    save_test_comparison_grid(vis_rf, vis_xgb, res_rf, res_xgb)
    save_bev_grid(bev_gt, bev_rf, bev_xgb)
    print(f"\nSaved all Optuna outputs to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()

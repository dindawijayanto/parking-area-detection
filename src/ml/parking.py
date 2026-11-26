# Shared code for the classical ML part: pixel features, prediction + post-processing, the bird's-eye geometry,
# the free-space counter and the free-space overlay. Used by notebooks/03_ml_classical.ipynb and
# src/ml/predict.py, so training, evaluation and prediction all compute features and free spaces the same way.

import json
from pathlib import Path

import cv2 as cv
import numpy as np

FRAME_W, FRAME_H = 1920, 1080   # aligned frames and ROI clicks
SEG_W, SEG_H = 960, 540         # masks, dataset and model input
FLOOR, MOTOR, IGNORE = 0, 1, 255
BEV_SIZE = 600
FIT = 0.7
SCALES = [1, 2, 4, 8]           # Gaussian sigmas of the filter bank

_clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ----------------------------------------------------------------------------------------------- features

def feature_names():
    names = ['B', 'G', 'R', 'H cos*S', 'H sin*S', 'S', 'V', 'V clahe', 'L', 'a', 'b', 'r chroma', 'g chroma']
    for s in SCALES:
        names += [f'Vc blur s{s}', f'Vc gradient s{s}', f'Vc LoG s{s}', f'Vc local std s{s}',
                  f'a blur s{s}', f'b blur s{s}', f'S blur s{s}']
    return names


def pixel_features(img):
    # BGR uint8 (h, w, 3) -> float32 (h, w, n_features): colour of the pixel + a multi-scale filter bank
    # (the same idea as ilastik / Trainable Weka Segmentation: every pixel is described by its neighbourhood)
    f = img.astype(np.float32)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB).astype(np.float32)
    h = hsv[:, :, 0].astype(np.float32) * (2 * np.pi / 180)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    vc = _clahe.apply(hsv[:, :, 2]).astype(np.float32)                 # brightness after CLAHE (notebooks/02)
    total = f.sum(axis=2) + 1
    feats = [f[:, :, 0], f[:, :, 1], f[:, :, 2],
             np.cos(h) * s / 255, np.sin(h) * s / 255,                    # hue on a circle, weighted by saturation
             s, v, vc, lab[:, :, 0], lab[:, :, 1], lab[:, :, 2],
             f[:, :, 2] / total, f[:, :, 1] / total]                     # chromaticity: brightness removed
    for sigma in SCALES:
        k = (0, 0)
        vb = cv.GaussianBlur(vc, k, sigma)
        gx, gy = cv.Sobel(vb, cv.CV_32F, 1, 0), cv.Sobel(vb, cv.CV_32F, 0, 1)
        mean_sq = cv.GaussianBlur(vc * vc, k, sigma * 2)
        mean = cv.GaussianBlur(vc, k, sigma * 2)
        feats += [vb, np.sqrt(gx * gx + gy * gy), cv.Laplacian(vb, cv.CV_32F) * sigma ** 2,
                  np.sqrt(np.maximum(mean_sq - mean * mean, 0)),
                  cv.GaussianBlur(lab[:, :, 1], k, sigma), cv.GaussianBlur(lab[:, :, 2], k, sigma),
                  cv.GaussianBlur(s, k, sigma)]
    return np.stack(feats, axis=2).astype(np.float32)


# ----------------------------------------------------------------------------------------------- prediction

def floor_probability(model, img, roi):
    # P(floor) for every pixel inside the ROI (0 outside); the model is trained with target 1 = floor
    feats = pixel_features(img)
    inside = roi > 0
    prob = np.zeros(roi.shape, np.float32)
    if inside.any():
        prob[inside] = model.predict_proba(feats[inside])[:, 1]
    return prob


def superpixels(img, roi, size=12):
    # SLIC superpixels of about size x size pixels (same size as the labeling tool), 0 = outside the ROI.
    # SLIC runs on the bounding box of the ROI without a mask: the same result inside, ~15x faster.
    from skimage.segmentation import slic
    sp = np.zeros(roi.shape, np.int32)
    ys, xs = np.where(roi > 0)
    if len(ys) == 0:
        return sp
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    n_segments = max(1, int((y1 - y0) * (x1 - x0) / size ** 2))
    sp[y0:y1, x0:x1] = slic(np.ascontiguousarray(img[y0:y1, x0:x1, ::-1]), n_segments=n_segments,
                            compactness=10, start_label=1)
    sp[roi == 0] = 0
    return sp


def superpixel_smooth(prob, sp):
    # average the probability inside every superpixel (only over its pixels inside the ROI)
    inside = (sp > 0).ravel()
    sums = np.bincount(sp.ravel()[inside], weights=prob.ravel()[inside], minlength=sp.max() + 1)
    counts = np.maximum(np.bincount(sp.ravel()[inside], minlength=sp.max() + 1), 1)
    out = (sums / counts)[sp].astype(np.float32)
    out[sp == 0] = 0
    return out


def morphology(mask, size=7):
    # opening removes small floor specks on motorcycles, closing fills small holes in the floor (notebooks/02)
    floor = np.uint8(mask == FLOOR)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))
    floor = cv.morphologyEx(floor, cv.MORPH_OPEN, kernel)
    floor = cv.morphologyEx(floor, cv.MORPH_CLOSE, kernel)
    out = np.where(floor == 1, FLOOR, MOTOR).astype(np.uint8)
    out[mask == IGNORE] = IGNORE
    return out


def to_mask(prob, roi, threshold=0.5, morph=False):
    mask = np.where(prob > threshold, FLOOR, MOTOR).astype(np.uint8)
    mask[roi == 0] = IGNORE
    return morphology(mask) if morph else mask


def predict_mask(bundle, img, roi):
    # bundle = what the notebook saves with joblib: model, threshold and post-processing
    prob = floor_probability(bundle['model'], img, roi)
    if bundle.get('superpixels'):
        prob = superpixel_smooth(prob, superpixels(img, roi))
    return to_mask(prob, roi, bundle['threshold'], bundle.get('morphology', False))


# ----------------------------------------------------------------------------------------------- geometry

def load_roi(path):
    # ROI json clicked on the 1920 reference -> coordinates on the 960 masks (same as notebooks/01)
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    s = SEG_W / FRAME_W
    return {'roi': [np.float32(p) * s for p in data['roi']],
            'ground': np.float32(data['ground']) * s,
            'motor': np.float32(data['motor']) * s}


def roi_mask(roi):
    mask = np.zeros((SEG_H, SEG_W), np.uint8)
    for poly in roi['roi']:
        cv.fillPoly(mask, [np.int32(poly)], 1)
    return mask


def bev_homography(roi):
    square = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    H = cv.getPerspectiveTransform(roi['ground'], square)
    pts = np.concatenate(roi['roi']).reshape(-1, 1, 2)
    bev = cv.perspectiveTransform(pts, H).reshape(-1, 2)
    x0, y0 = bev.min(axis=0)
    x1, y1 = bev.max(axis=0)
    scale = BEV_SIZE / max(x1 - x0, y1 - y0)
    T = np.float64([[scale, 0, -x0 * scale], [0, scale, -y0 * scale], [0, 0, 1]])
    size = (int(np.ceil((x1 - x0) * scale)) + 1, int(np.ceil((y1 - y0) * scale)) + 1)
    return T @ H, size


def prepare_video(roi, fit=FIT):
    # same as prepare_video in notebooks/01, so geo from the pickle and geo from a ROI json are identical
    H, size = bev_homography(roi)
    roi_bev = cv.warpPerspective(roi_mask(roi), H, size, flags=cv.INTER_NEAREST)
    motor_bev = cv.perspectiveTransform(roi['motor'].reshape(-1, 1, 2), H).reshape(-1, 2)
    motor_area = cv.contourArea(motor_bev)
    shape = (motor_bev - motor_bev.mean(axis=0)) * fit
    shape = shape - shape.min(axis=0)
    kernel = np.zeros((int(np.ceil(shape[:, 1].max())) + 1, int(np.ceil(shape[:, 0].max())) + 1), np.uint8)
    cv.fillPoly(kernel, [np.int32(np.round(shape))], 1)
    return {'roi_mask': roi_mask(roi), 'H': H, 'size': size, 'roi_bev': roi_bev,
            'motor_bev': motor_bev, 'motor_area': motor_area, 'kernel': kernel}


def free_spaces(mask, g):
    # mask -> bird's-eye floor -> opening with one motorcycle -> every area / one motorcycle
    # returns the count plus everything needed to draw it
    floor = (mask == FLOOR).astype(np.uint8)
    floor_bev = cv.warpPerspective(floor, g['H'], g['size'], flags=cv.INTER_NEAREST) * g['roi_bev']
    usable = cv.morphologyEx(floor_bev, cv.MORPH_OPEN, g['kernel'])
    n, labels, stats, centroids = cv.connectedComponentsWithStats(usable)
    areas = []
    for c in range(1, n):
        k = max(1, int(round(stats[c, cv.CC_STAT_AREA] / g['motor_area'])))
        areas.append({'label': c, 'spaces': k, 'centroid_bev': centroids[c]})
    total = sum(a['spaces'] for a in areas)
    return {'free': total, 'areas': areas, 'labels_bev': labels, 'floor_bev': floor_bev, 'usable_bev': usable,
            'floor_area': floor_bev.sum() / g['motor_area'], 'roi_area': g['roi_bev'].sum() / g['motor_area']}


def count_free_spaces(mask, g):
    return free_spaces(mask, g)['free']


def draw_free_spaces(img, mask, g, result=None):
    # camera view: red = motorcycle, green = free parking area that fits a motorcycle, number = spaces per area
    if result is None:
        result = free_spaces(mask, g)
    usable = cv.warpPerspective(result['usable_bev'], g['H'], (SEG_W, SEG_H),
                                flags=cv.INTER_NEAREST | cv.WARP_INVERSE_MAP) * g['roi_mask']
    color = img.copy()
    color[mask == MOTOR] = (0, 0, 230)
    color[usable > 0] = (0, 220, 0)
    vis = cv.addWeighted(color, 0.45, img, 0.55, 0)
    contours, _ = cv.findContours(g['roi_mask'], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(vis, contours, -1, (255, 200, 0), 2)
    contours, _ = cv.findContours(np.uint8(usable > 0), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(vis, contours, -1, (0, 255, 0), 2)

    H_inv = np.linalg.inv(g['H'])
    for a in result['areas']:
        p = cv.perspectiveTransform(np.float32(a['centroid_bev']).reshape(1, 1, 2), H_inv)[0, 0]
        org = (int(p[0]) - 8, int(p[1]) + 8)
        cv.putText(vis, str(a['spaces']), org, cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(vis, str(a['spaces']), org, cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)
    text = f"free: {result['free']} motorcycles"
    cv.putText(vis, text, (12, 36), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv.LINE_AA)
    cv.putText(vis, text, (12, 36), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
    return vis

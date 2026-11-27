# PCD pipeline used by the web demo (app/pcd_demo.py), the same steps as notebooks/02 step 10-15:
#   BGR -> gray world (shades of gray) -> histogram matching in Lab to the train days -> HSV -> median on V
#   -> grayscale closing on V (removes the thin dark paving joints).  CLAHE was dropped: it lowered mIoU everywhere.
# plus segmentations that use PCD only (no machine learning):
#   threshold_segment: threshold on V (fixed from train, or Otsu) + opening / closing      (old baseline)
#   segment:           score channel (V, inverted S or V - w*S) -> fixed / Otsu threshold
#                      -> optional adaptive threshold (local mean) and Canny edge density as extra 'motor' evidence
#                      -> opening / closing -> remove small blobs and fill small holes
#   the best settings are searched by src/pcd/search_best.py -> results/pcd_eval/best_config.json
# and the evaluation helpers (overlap, confusion matrix, image statistics).

import json
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FRAMES_DIR = ROOT / 'data' / 'raw' / 'frames'
MASK_DIR = ROOT / 'data' / 'annotations' / 'masks'
ROI_DIR = ROOT / 'data' / 'annotations' / 'roi'
REF_DIR = ROOT / 'data' / 'annotations' / 'references'

SEG_W, SEG_H = 960, 540
FLOOR, MOTOR, IGNORE = 0, 1, 255
TEST_DAYS = ['20261104', '20261112']   # same split as notebooks/01
VAL_DAYS = ['20261105', '20261110']


# ----------------------------------------------------------------------------------------------- dataset

def video_of(frame_name):
    return Path(frame_name).name.rsplit('_t', 1)[0]


def split_of(frame_name):
    day = Path(frame_name).name[:8]
    return 'test' if day in TEST_DAYS else 'val' if day in VAL_DAYS else 'train'


def labeled_frames():
    # frames that exist on disk and have a label mask
    if not FRAMES_DIR.exists():
        return []
    return [p.name for p in sorted(FRAMES_DIR.glob('*.jpg')) if (MASK_DIR / p.with_suffix('.png').name).exists()]


def load_frame(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = cv.imdecode(np.frombuffer(path_or_bytes, np.uint8), cv.IMREAD_COLOR)
    else:
        img = cv.imread(str(path_or_bytes))
    return cv.resize(img, (SEG_W, SEG_H), interpolation=cv.INTER_AREA)


def load_mask(frame_name):
    p = MASK_DIR / Path(frame_name).with_suffix('.png').name
    return cv.imread(str(p), cv.IMREAD_GRAYSCALE) if p.exists() else None


def roi_of(video_id, img):
    data = json.loads((ROI_DIR / f'{video_id}.json').read_text(encoding='utf-8'))
    roi = np.zeros((SEG_H, SEG_W), np.uint8)
    for poly in data['roi']:
        cv.fillPoly(roi, [np.int32(np.float32(poly) * SEG_W / 1920)], 1)
    roi[img.max(axis=2) == 0] = 0          # black border left by the alignment warp
    return roi


# ----------------------------------------------------------------------------------------------- alignment

def align_to_reference(img, video_id, min_inliers=40):
    # same registration as notebooks/00: CLAHE gray -> SIFT -> ratio test -> RANSAC homography -> warp
    ref = load_frame(REF_DIR / f'{video_id}.jpg')
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    sift = cv.SIFT_create(nfeatures=5000)
    kp1, d1 = sift.detectAndCompute(clahe.apply(cv.cvtColor(img, cv.COLOR_BGR2GRAY)), None)
    kp2, d2 = sift.detectAndCompute(clahe.apply(cv.cvtColor(ref, cv.COLOR_BGR2GRAY)), None)
    if d1 is None or d2 is None:
        return img, 0
    pairs = cv.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return img, 0
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv.findHomography(src, dst, cv.RANSAC, 4.0)
    n = int(inl.sum()) if inl is not None else 0
    if H is None or n < min_inliers:
        return img, n
    return cv.warpPerspective(img, H, (SEG_W, SEG_H)), n


# ----------------------------------------------------------------------------------------------- PCD steps

def shades_of_gray(img, mask, p=6):
    # colour constancy: colour of the light = p-norm mean of every channel inside the ROI
    f = img.astype(np.float32)
    illum = np.power(np.mean(f[mask > 0] ** p, axis=0), 1 / p)
    return np.uint8(np.clip(f * illum.mean() / illum, 0, 255)), illum


def channel_cdf(values):
    h = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    return np.cumsum(h) / h.sum()


def lab_reference(gray_world=True, p=6):
    # L, a, b CDFs of all ROI pixels of the train frames (never val / test), as notebooks/02 step 10
    px = []
    for f in labeled_frames():
        if split_of(f) != 'train':
            continue
        img = load_frame(FRAMES_DIR / f)
        roi = roi_of(video_of(f), img)
        if gray_world:
            img = shades_of_gray(img, roi, p)[0]
        px.append(cv.cvtColor(img, cv.COLOR_BGR2LAB)[roi > 0])
    px = np.concatenate(px)
    return [channel_cdf(px[:, c]) for c in range(3)]


def hist_match(img, mask, ref):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    for c in range(3):
        cdf = channel_cdf(lab[:, :, c][mask > 0])
        lut = np.uint8(np.clip(np.round(np.interp(cdf, ref[c], np.arange(256))), 0, 255))
        lab[:, :, c] = lut[lab[:, :, c]]
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)


def run_pcd(img, roi, cfg, ref=None):
    # returns a list of (step name, BGR picture after the step) and the final HSV picture
    steps = [('original', img)]
    out = img
    if cfg.get('gray_world'):
        out, _ = shades_of_gray(out, roi, cfg.get('p', 6))
        steps.append(('gray world', out))
    if cfg.get('hist_match') and ref is not None:
        out = hist_match(out, roi, ref)
        steps.append(('histogram matching', out))
    hsv = cv.cvtColor(out, cv.COLOR_BGR2HSV)
    if cfg.get('median'):
        hsv[:, :, 2] = cv.medianBlur(hsv[:, :, 2], cfg.get('median_k', 5))
        steps.append(('median blur on V', cv.cvtColor(hsv, cv.COLOR_HSV2BGR)))
    if cfg.get('v_close'):
        # grayscale closing = dilate then erode: dark lines thinner than the kernel (paving joints) disappear,
        # big dark objects (motorcycles, tyres) stay
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (cfg.get('close_k', 5),) * 2)
        hsv[:, :, 2] = cv.morphologyEx(hsv[:, :, 2], cv.MORPH_CLOSE, k)
        steps.append(('closing on V', cv.cvtColor(hsv, cv.COLOR_HSV2BGR)))
    return steps, hsv


# ----------------------------------------------------------------------------------------------- PCD-only segmentation

def threshold_segment(hsv, roi, t=None, morph_k=7):
    # floor = bright paving: V > t inside the ROI, then opening / closing.
    # t = None -> Otsu picks a threshold for this picture only; a number -> one fixed threshold for every picture
    v = hsv[:, :, 2]
    if t is None:
        t, _ = cv.threshold(v[roi > 0].reshape(-1, 1), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    floor = np.uint8(v > t)
    if morph_k > 1:
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_k, morph_k))
        floor = cv.morphologyEx(cv.morphologyEx(floor, cv.MORPH_OPEN, k), cv.MORPH_CLOSE, k)
    mask = np.where(floor == 1, FLOOR, MOTOR).astype(np.uint8)
    mask[roi == 0] = IGNORE
    return mask, float(t)


def score_map(s, v, channel='V', w_s=1.0):
    # brightness-like score where floor is high: V, 255 - S (paving is grey = low saturation) or V - w*S
    if channel == 'V':
        return v.copy()
    if channel == 'S':
        return np.uint8(255 - s)
    x = (v.astype(np.float32) - w_s * s.astype(np.float32) + 255 * w_s) / (1 + w_s)
    return np.uint8(np.clip(x, 0, 255))


def otsu(x, roi):
    t, _ = cv.threshold(x[roi > 0].reshape(-1, 1), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return float(t)


def adaptive_mask(x, block, c):
    # floor must also be brighter than (local mean - c): catches dark objects the global threshold lets through
    return cv.adaptiveThreshold(x, 1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, int(block), float(c)) > 0


def edge_mask(x, win, max_density, lo=50, hi=150):
    # Canny edge density in a win x win window: plain paving has few edges, motorcycles have many
    e = cv.Canny(cv.GaussianBlur(x, (5, 5), 0), lo, hi)
    d = cv.boxFilter((e > 0).astype(np.float32), -1, (int(win), int(win)))
    return d < max_density


def refine_mask(x, seg):
    r = None
    if seg.get('adaptive'):
        r = adaptive_mask(x, *seg['adaptive'])
    if seg.get('edge'):
        e = edge_mask(x, *seg['edge'])
        r = e if r is None else r & e
    return r


def cleanup(floor, roi, morph_k=7, min_area=0):
    if morph_k > 1:
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_k, morph_k))
        floor = cv.morphologyEx(cv.morphologyEx(floor, cv.MORPH_OPEN, k), cv.MORPH_CLOSE, k)
    if min_area > 0:
        for val in (1, 0):        # small floor islands -> motorcycle, then small holes in motorcycles -> floor
            comp = ((floor == val) & (roi > 0)).astype(np.uint8)
            n, lab, st, _ = cv.connectedComponentsWithStats(comp, connectivity=8)
            kill = st[:, cv.CC_STAT_AREA] < min_area
            kill[0] = False
            floor[kill[lab]] = 1 - val
    return floor


def segment_score(x, roi, seg, t=None, r=None):
    if seg.get('method', 'fixed') == 'otsu' or t is None:
        t = otsu(x, roi)
    floor = x > t
    if r is None:
        r = refine_mask(x, seg)
    if r is not None:
        floor &= r
    floor = cleanup(floor.astype(np.uint8), roi, seg.get('morph_k', 7), seg.get('min_area', 0))
    mask = np.where(floor == 1, FLOOR, MOTOR).astype(np.uint8)
    mask[roi == 0] = IGNORE
    return mask, float(t)


def segment(hsv, roi, seg, t=None):
    x = score_map(hsv[:, :, 1], hsv[:, :, 2], seg.get('channel', 'V'), seg.get('w_s', 1.0))
    return segment_score(x, roi, seg, t)


def load_best():
    p = ROOT / 'results' / 'pcd_eval' / 'best_config.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def train_threshold(cfg, ref=None, morph_k=7, candidates=range(60, 231, 10)):
    # the fixed threshold on V with the best mIoU over all train frames (val / test are never used)
    data = []
    for f in labeled_frames():
        if split_of(f) != 'train':
            continue
        img = load_frame(FRAMES_DIR / f)
        roi = roi_of(video_of(f), img)
        label = load_mask(f).copy()
        label[roi == 0] = IGNORE
        data.append((run_pcd(img, roi, cfg, ref)[1], roi, label))
    best, best_miou = None, -1
    for t in candidates:
        cm = sum(confusion(threshold_segment(hsv, roi, t, morph_k)[0], label) for hsv, roi, label in data)
        m = cm_metrics(cm)['mIoU']
        if m > best_miou:
            best, best_miou = t, m
    return float(best), best_miou


# ----------------------------------------------------------------------------------------------- evaluation

def overlap(a, b, bins=64, value_range=(0, 256)):
    # overlap of the two normalised histograms: 0 = fully separated, 1 = identical
    ha, _ = np.histogram(a, bins=bins, range=value_range)
    hb, _ = np.histogram(b, bins=bins, range=value_range)
    return float(np.sum(np.minimum(ha / max(ha.sum(), 1), hb / max(hb.sum(), 1))))


def separability(hsv, label):
    rows = {}
    for name, ch, rng in [('H', 0, (0, 180)), ('S', 1, (0, 256)), ('V', 2, (0, 256))]:
        rows[name] = overlap(hsv[:, :, ch][label == FLOOR], hsv[:, :, ch][label == MOTOR], value_range=rng)
    return rows


def image_stats(hsv, roi, label=None):
    v = hsv[:, :, 2]
    inside = v[roi > 0]
    hist = np.bincount(inside, minlength=256) / max(inside.size, 1)
    nz = hist[hist > 0]
    out = {'mean V (ROI)': float(inside.mean()),
           'contrast = std V (ROI)': float(inside.std()),
           'entropy V (bits)': float(-(nz * np.log2(nz)).sum()),
           'sharpness = var(Laplacian)': float(cv.Laplacian(v, cv.CV_64F)[roi > 0].var())}
    if label is not None:
        out['median V floor'] = float(np.median(v[label == FLOOR]))
        out['median V motorcycle'] = float(np.median(v[label == MOTOR]))
    return out


def confusion(pred, label):
    valid = (label != IGNORE) & (pred != IGNORE)
    return np.bincount(label[valid].astype(np.int64) * 2 + pred[valid], minlength=4).reshape(2, 2)


def cm_metrics(cm):
    out = {}
    for cls, n in [(FLOOR, 'floor'), (MOTOR, 'motorcycle')]:
        tp, fp, fn = cm[cls, cls], cm[:, cls].sum() - cm[cls, cls], cm[cls].sum() - cm[cls, cls]
        out[f'precision {n}'] = tp / max(tp + fp, 1)
        out[f'recall {n}'] = tp / max(tp + fn, 1)
        out[f'F1 {n}'] = 2 * tp / max(2 * tp + fp + fn, 1)
        out[f'IoU {n}'] = tp / max(tp + fp + fn, 1)
    out['mIoU'] = (out['IoU floor'] + out['IoU motorcycle']) / 2
    out['pixel accuracy'] = np.trace(cm) / max(cm.sum(), 1)
    return out


def overlay(img, mask, alpha=0.45):
    color = np.zeros_like(img)
    color[mask == FLOOR] = (0, 200, 0)
    color[mask == MOTOR] = (0, 0, 230)
    color[mask == IGNORE] = (150, 150, 150)
    return cv.addWeighted(color, alpha, img, 1 - alpha, 0)

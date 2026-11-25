# Predict the free parking area of new frames (or a video) with the trained classical ML model.
#
#   python src/ml/predict.py <roi video id> <image / folder / video> [--model models/xgboost.joblib] [--every 15]
#
#   python src/ml/predict.py 20261104_evening_keiza data/raw/frames/20261104_evening_keiza_t000.jpg
#   python src/ml/predict.py 20261112_morning_keiza data/raw/frames
#   python src/ml/predict.py 20261112_morning_keiza data/raw/new_video.mp4 --every 5 --model models/random_forest.joblib
#
# The pictures must be taken from the same camera position as the ROI video (aligned like notebooks/00),
# because the ROI, the ground rectangle and the size of one motorcycle come from that video's ROI json.
# Output in results/predictions/: <name>_free.jpg (green = free area, number = motorcycles that fit there)
# and free_spaces.csv with one row per picture.

import argparse
import csv
import sys
from pathlib import Path

import cv2 as cv
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parking as pk  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ROI_DIR = ROOT / 'data' / 'annotations' / 'roi'


def pictures(source, every):
    source = Path(source)
    if source.is_dir():
        for p in sorted(source.glob('*.jpg')) + sorted(source.glob('*.png')):
            yield p.stem, cv.imread(str(p))
    elif source.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
        cap = cv.VideoCapture(str(source))
        fps = cap.get(cv.CAP_PROP_FPS) or 30
        step, k = max(1, int(round(every * fps))), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if k % step == 0:
                yield f'{source.stem}_t{int(k / fps):03d}', frame
            k += 1
        cap.release()
    else:
        yield source.stem, cv.imread(str(source))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video_id', help='video whose ROI json is used, e.g. 20261104_evening_keiza')
    ap.add_argument('source', help='image, folder of images or video')
    ap.add_argument('--model', default=str(ROOT / 'models' / 'xgboost.joblib'))
    ap.add_argument('--every', type=float, default=15, help='seconds between frames for a video')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'predictions'))
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    g = pk.prepare_video(pk.load_roi(ROI_DIR / f'{args.video_id}.json'))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"model {args.model}: {bundle['name']}, threshold {bundle['threshold']:.2f}, "
          f"superpixels {bundle['superpixels']}, morphology {bundle['morphology']}")

    rows = []
    for name, img in pictures(args.source, args.every):
        if img is None:
            print('cannot read', name)
            continue
        img = cv.resize(img, (pk.SEG_W, pk.SEG_H), interpolation=cv.INTER_AREA)
        roi = g['roi_mask'].copy()
        roi[img.max(axis=2) == 0] = 0                      # black border left by the alignment warp
        mask = pk.predict_mask(bundle, img, roi)
        result = pk.free_spaces(mask, g)
        cv.imwrite(str(out / f'{name}_free.jpg'), pk.draw_free_spaces(img, mask, g, result))
        rows.append([name, result['free'], round(result['floor_area'], 1), round(result['roi_area'], 1)])
        print(f"{name}: {result['free']} free spaces (empty floor = {result['floor_area']:.1f} motorcycles of area)")

    with open(out / 'free_spaces.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['picture', 'free_spaces', 'empty_floor_in_motorcycles', 'roi_in_motorcycles'])
        w.writerows(rows)
    print('saved', len(rows), 'predictions in', out)


if __name__ == '__main__':
    main()

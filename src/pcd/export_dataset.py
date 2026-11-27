# Build the dataset pickles for the ML / DL notebooks straight from the frames + annotations, so nobody has to
# send 200 MB pickles around: share data/raw/frames (~77 MB) and let everyone run this script.
#
#   python src/pcd/export_dataset.py              -> data/processed/dataset_seg.pkl + dataset_pcd.pkl
#   python src/pcd/export_dataset.py --only pcd   -> only dataset_pcd.pkl
#   python src/pcd/export_dataset.py --pcd "gw + hm + median"   -> another variant from search_best.VARIANTS
#
# dataset_seg.pkl  X = aligned frames 960x540 BGR uint8 without PCD (same format as notebooks/01)
# dataset_pcd.pkl  X = HSV uint8 960x540 after the best PCD (results/pcd_eval/best_config.json, same format as
#                  notebooks/02) + pcd_mask = the PCD-only segmentation of every frame (0 floor, 1 motor, 255 ignore)
# both: Y = label masks, meta = DataFrame (frame, video_id, date, time, weather, who, t_sec, split, floor % of ROI,
#       free spaces), geo[video_id] = bird's-eye geometry from src/ml/parking.py, classes, info

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'ml'))
import pipeline as P   # noqa: E402
import parking as pk   # noqa: E402

OUT = P.ROOT / 'data' / 'processed'
CLASSES = {P.FLOOR: 'floor', P.MOTOR: 'motorcycle', P.IGNORE: 'ignore'}


def video_info():
    # time / weather per video_id: data/annotations/videos.csv (small, in git)
    p = P.ROOT / 'data' / 'annotations' / 'videos.csv'
    return {r.video_id: r._asdict() for r in pd.read_csv(p).itertuples(index=False)} if p.exists() else {}


def build_meta(frames, labels, geo):
    info = video_info()
    rows = []
    for f, y in zip(frames, labels):
        vid = P.video_of(f)
        date, time, who = vid.split('_')
        i = info.get(vid, {})
        rows.append({'frame': f, 'video_id': vid, 'date': date, 'time': i.get('time', time),
                     'weather': i.get('weather', 'unknown'), 'who': who,
                     't_sec': int(Path(f).stem.rsplit('_t', 1)[1]), 'split': P.split_of(f),
                     'floor % of ROI': 100 * float(np.sum(y == P.FLOOR)) / max(int(np.sum(y != P.IGNORE)), 1),
                     'free spaces': pk.free_spaces(y, geo[vid])['free']})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['seg', 'pcd'])
    ap.add_argument('--pcd', help='variant name from search_best.VARIANTS (default: best_config.json)')
    args = ap.parse_args()

    frames = P.labeled_frames()
    if not frames:
        sys.exit(f'no frames with a mask in {P.FRAMES_DIR}: put the aligned frames (*.jpg) there first')
    best = P.load_best()
    if args.pcd:
        import search_best as S
        cfg, seg, t, name = S.VARIANTS[args.pcd], None, None, args.pcd
    elif best:
        cfg, seg, t, name = best['pcd'], best['seg'], best['t'], best['variant']
    else:
        sys.exit('results/pcd_eval/best_config.json not found: pass --pcd "<variant>" or run search_best.py')

    print(f'{len(frames)} frames, PCD = {name}')
    imgs = [P.load_frame(P.FRAMES_DIR / f) for f in frames]
    rois = [P.roi_of(P.video_of(f), img) for f, img in zip(frames, imgs)]
    Y = np.stack([P.load_mask(f) for f in frames])
    geo = {v: pk.prepare_video(pk.load_roi(P.ROI_DIR / f'{v}.json')) for v in sorted({P.video_of(f) for f in frames})}
    meta = build_meta(frames, Y, geo)
    print(meta.groupby('split').size().to_dict())
    OUT.mkdir(parents=True, exist_ok=True)

    if args.only != 'pcd':
        data = {'X': np.stack(imgs), 'Y': Y, 'meta': meta, 'geo': geo, 'fit': pk.FIT, 'classes': CLASSES,
                'info': 'X = aligned frames 960x540 BGR uint8 (no PCD); Y = masks 0 floor, 1 motorcycle, 255 ignore; '
                        'geo[video_id] = roi_mask, bird\'s-eye homography H, size, roi_bev, motor_area, kernel; '
                        'split by day in meta["split"], never randomly'}
        save(OUT / 'dataset_seg.pkl', data)

    if args.only != 'seg':
        ref = P.lab_reference(bool(cfg.get('gray_world')), cfg.get('p', 6)) if cfg.get('hist_match') else None
        X, masks = [], []
        for img, roi in zip(imgs, rois):
            hsv = P.run_pcd(img, roi, cfg, ref)[1]
            X.append(hsv)
            if seg:
                masks.append(P.segment(hsv, roi, {**seg, 'adaptive': seg['adaptive'] and tuple(seg['adaptive']),
                                                  'edge': seg['edge'] and tuple(seg['edge'])}, t)[0])
        data = {'X': np.stack(X), 'Y': Y, 'meta': meta, 'geo': geo, 'fit': pk.FIT, 'classes': CLASSES,
                'pcd': name, 'pcd_config': cfg, 'lab_reference_cdf': ref,
                'pcd_mask': np.stack(masks) if masks else None, 'pcd_seg': seg, 'pcd_t': t,
                'info': f'X = HSV uint8 960x540 after the PCD "{name}" (src/pcd/pipeline.py run_pcd); '
                        'lab_reference_cdf = train L, a, b CDFs used by the histogram matching (None if not used); '
                        'pcd_mask = PCD-only segmentation (0 floor, 1 motorcycle, 255 ignore), usable as a baseline '
                        'or an extra input feature; Y, meta, geo exactly as in dataset_seg.pkl'}
        save(OUT / 'dataset_pcd.pkl', data)


def save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'saved {path.relative_to(P.ROOT)} ({path.stat().st_size / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()

# Search for the best PCD-only segmentation (no machine learning).
#
#   python src/pcd/search_best.py            (first run builds the .pkl cache, later runs reuse it)
#   python src/pcd/search_best.py --rebuild  (rebuild the cache, e.g. after relabeling)
#
# 1. cache: every preprocessing variant -> S and V of all labeled frames -> results/pcd_eval/cache/*.pkl
# 2. stage 1: variant x score channel x threshold method (fixed / Otsu) x adaptive threshold x Canny edge density
#    (morphology 7, no blob filter). fixed threshold = best train mIoU, ranking = mIoU on train + val ('sel').
#    Otsu learns nothing from train, so train frames are fair selection data (66 frames instead of 13 val frames).
#    test is never used for any choice.
# 3. stage 2: top configs of stage 1 x morphology kernel x minimum blob area, ranked on sel mIoU - 0.01 * sel MAE
# 4. the val winner -> train / val / test mIoU + free-space MAE -> results/pcd_eval/best_config.json,
#    search_stage1.csv, search_stage2.csv, search_final.csv (the app reads best_config.json)

import argparse
import ast
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'ml'))
import pipeline as P   # noqa: E402
import parking as pk   # noqa: E402

OUT = P.ROOT / 'results' / 'pcd_eval'
CACHE = OUT / 'cache'

VARIANTS = {
    'raw': {},
    'median': {'median': True},
    'median + close': {'median': True, 'v_close': True},
    'gray world + median': {'gray_world': True, 'median': True},
    'hist match + median': {'hist_match': True, 'median': True},
    'gw + hm + median': {'gray_world': True, 'hist_match': True, 'median': True},
    'gw + hm + median + close': {'gray_world': True, 'hist_match': True, 'median': True, 'v_close': True},
    'gw + hm + close': {'gray_world': True, 'hist_match': True, 'v_close': True},
}
for _v in VARIANTS.values():
    _v.update({'p': 6, 'median_k': 5, 'close_k': 5, **_v})

CHANNELS = [('V', 0.0), ('S', 0.0), ('V-S', 0.5), ('V-S', 1.0)]
ADAPTIVE = [None, (51, 5), (101, 5), (101, 10), (201, 10)]     # (block size, C)
EDGE = [None, (15, 0.10), (25, 0.15), (25, 0.25)]              # (window, max edge density)
MORPH = [5, 7, 9, 11, 15, 21]
AREA = [0, 2000, 5000, 10000, 20000]
TOP_K = 12
MAE_W = 0.01                                                   # stage 2 score = val mIoU - MAE_W * val MAE


def slug(name):
    return ''.join(c if c.isalnum() else '_' for c in name)


def cached(path, build, rebuild):
    if path.exists() and not rebuild:
        with open(path, 'rb') as f:
            return pickle.load(f)
    obj = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return obj


def build_meta():
    meta = []
    for f in P.labeled_frames():
        img = P.load_frame(P.FRAMES_DIR / f)
        roi = P.roi_of(P.video_of(f), img)
        label = P.load_mask(f).copy()
        label[roi == 0] = P.IGNORE
        meta.append({'frame': f, 'split': P.split_of(f), 'video': P.video_of(f), 'roi': roi, 'label': label})
    return meta


def variant_data(name, meta, rebuild):
    cfg = VARIANTS[name]

    def build():
        ref = None
        if cfg.get('hist_match'):
            gw = bool(cfg.get('gray_world'))
            ref = cached(CACHE / f'lab_ref_gw{int(gw)}.pkl', lambda: P.lab_reference(gw, cfg['p']), rebuild)
        out = []
        for m in meta:
            hsv = P.run_pcd(P.load_frame(P.FRAMES_DIR / m['frame']), m['roi'], cfg, ref)[1]
            out.append((hsv[:, :, 1].copy(), hsv[:, :, 2].copy()))
        return out
    return cached(CACHE / f'{slug(name)}.pkl', build, rebuild)


def fit_threshold(xs, rs, meta, idx, lo=30, hi=240):
    # best fixed threshold on the train frames, straight from histograms (before morphology): fast and exact
    hf, hm, out_f, out_m = np.zeros(256), np.zeros(256), 0, 0
    for i in idx:
        lab, x, r = meta[i]['label'], xs[i], rs[i]
        fl, mo = lab == P.FLOOR, lab == P.MOTOR
        if r is not None:
            out_f += fl.sum() - (fl & r).sum()
            out_m += mo.sum() - (mo & r).sum()
            fl, mo = fl & r, mo & r
        hf += np.bincount(x[fl], minlength=256)
        hm += np.bincount(x[mo], minlength=256)
    F, M = hf.sum() + out_f, hm.sum() + out_m
    tp = np.append(np.cumsum(hf[::-1])[::-1], 0)[1:]     # floor pixels with x > t
    fp = np.append(np.cumsum(hm[::-1])[::-1], 0)[1:]     # motorcycle pixels with x > t
    miou = (tp / np.maximum(F + fp, 1) + (M - fp) / np.maximum(M + F - tp, 1)) / 2
    t = lo + int(np.argmax(miou[lo:hi]))
    return float(t), float(miou[t])


def evaluate(xs, rs, meta, idx, seg, t, with_mae=False, geoms=None):
    cm, errs = np.zeros((2, 2), np.int64), []
    for i in idx:
        m = meta[i]
        pred, _ = P.segment_score(xs[i], m['roi'], seg, t, rs[i])
        cm += P.confusion(pred, m['label'])
        if with_mae:
            g = geoms[m['video']]
            errs.append(abs(pk.free_spaces(pred, g)['free'] - pk.free_spaces(m['label'], g)['free']))
    out = P.cm_metrics(cm)
    if with_mae:
        out['MAE free'] = float(np.mean(errs)) if errs else float('nan')
    return out


def clean_seg(r):
    # pandas turns None into NaN and ints into numpy ints: back to plain python for segment() and json
    tup = lambda v: tuple(float(x) for x in v) if isinstance(v, (tuple, list)) else None
    return {'channel': str(r['channel']), 'w_s': float(r['w_s']), 'method': str(r['method']),
            'adaptive': tup(r['adaptive']), 'edge': tup(r['edge']),
            'morph_k': int(r.get('morph_k', 7)), 'min_area': int(r.get('min_area', 0))}


def combine(a, e):
    return a if e is None else e if a is None else a & e


def scores(sv, ch, w):
    return [P.score_map(s, v, ch, w) for s, v in sv]


def load_stage1():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--stage1', action='store_true', help='redo stage 1 even if search_stage1.csv exists')
    args = ap.parse_args()

    t0 = time.time()
    meta = cached(CACHE / 'meta.pkl', build_meta, args.rebuild)
    split = {s: [i for i, m in enumerate(meta) if m['split'] == s] for s in ('train', 'val', 'test')}
    split['sel'] = split['train'] + split['val']
    print({k: len(v) for k, v in split.items()}, 'frames')

    # ---------------------------------------------------------------------------------- stage 1
    s1_path = OUT / 'search_stage1.csv'
    old = s1_path.exists() and 'sel mIoU' not in pd.read_csv(s1_path, nrows=1).columns
    if s1_path.exists() and not (args.rebuild or args.stage1 or old):
        s1 = pd.read_csv(s1_path)
        for c in ('adaptive', 'edge'):
            s1[c] = [ast.literal_eval(v) if isinstance(v, str) else None for v in s1[c]]
        print(f'stage 1 loaded from {s1_path.name} (use --stage1 to redo it)')
    else:
        s1 = stage1(meta, split, args.rebuild, t0)
        s1.to_csv(s1_path, index=False)
    return s1, meta, split, t0


def stage1(meta, split, rebuild, t0):
    rows = []
    for vname in VARIANTS:
        sv = variant_data(vname, meta, rebuild)
        for ch, w in CHANNELS:
            xs = scores(sv, ch, w)
            A = {a: [P.adaptive_mask(x, *a) for x in xs] if a else [None] * len(xs) for a in ADAPTIVE}
            E = {e: [P.edge_mask(x, *e) for x in xs] if e else [None] * len(xs) for e in EDGE}
            for a, e in itertools.product(ADAPTIVE, EDGE):
                rs = [combine(A[a][i], E[e][i]) for i in range(len(xs))]
                for method in ('fixed', 'otsu'):
                    seg = {'channel': ch, 'w_s': w, 'method': method, 'adaptive': a, 'edge': e,
                           'morph_k': 7, 'min_area': 0}
                    t, train_fit = fit_threshold(xs, rs, meta, split['train']) if method == 'fixed' else (None, None)
                    sel = evaluate(xs, rs, meta, split['sel'], seg, t)
                    rows.append({'variant': vname, **seg, 't': t, 'train mIoU (fit)': train_fit,
                                 'sel mIoU': sel['mIoU']})
            del A, E
        print(f'stage 1  {vname:28s} best sel mIoU so far {max(r["sel mIoU"] for r in rows):.3f}  '
              f'({time.time() - t0:.0f}s)')
    return pd.DataFrame(rows).sort_values('sel mIoU', ascending=False)


def main():
    s1, meta, split, t0 = load_stage1()

    # ---------------------------------------------------------------------------------- stage 2
    geoms = {v: pk.prepare_video(pk.load_roi(P.ROI_DIR / f'{v}.json')) for v in {m['video'] for m in meta}}
    rows = []
    for _, r in s1.head(TOP_K).iterrows():
        seg = clean_seg(r)
        t = None if pd.isna(r['t']) else float(r['t'])
        xs = scores(variant_data(r['variant'], meta, False), seg['channel'], seg['w_s'])
        rs = [P.refine_mask(x, seg) for x in xs]
        for mk, area in itertools.product(MORPH, AREA):
            s = {**seg, 'morph_k': mk, 'min_area': area}
            sel = evaluate(xs, rs, meta, split['sel'], s, t, True, geoms)
            rows.append({'variant': r['variant'], **s, 't': t, 'sel mIoU': sel['mIoU'], 'sel MAE': sel['MAE free'],
                         'score': sel['mIoU'] - MAE_W * sel['MAE free']})
    s2 = pd.DataFrame(rows).sort_values('score', ascending=False)
    s2.to_csv(OUT / 'search_stage2.csv', index=False)
    print(f'stage 2 done ({time.time() - t0:.0f}s)')

    # ---------------------------------------------------------------------------------- final report
    baseline = {'variant': 'raw', 'channel': 'V', 'w_s': 0.0, 'method': 'fixed', 'adaptive': None, 'edge': None,
                'morph_k': 7, 'min_area': 0, 't': None}
    final = [baseline] + s2.head(10).to_dict('records')
    rows = []
    for k, r in enumerate(final):
        seg = clean_seg(r)
        xs = scores(variant_data(r['variant'], meta, False), seg['channel'], seg['w_s'])
        rs = [P.refine_mask(x, seg) for x in xs]
        t = float(r['t']) if r['t'] is not None and not pd.isna(r['t']) else None
        if seg['method'] == 'fixed' and t is None:
            t = fit_threshold(xs, rs, meta, split['train'])[0]
        res = {s: evaluate(xs, rs, meta, split[s], seg, t, True, geoms) for s in ('train', 'val', 'sel', 'test')}
        rows.append({'name': 'baseline (threshold V lama)' if k == 0 else f'rank {k}', 'variant': r['variant'],
                     **seg, 't': t,
                     **{f'{s} mIoU': res[s]['mIoU'] for s in res}, **{f'{s} MAE': res[s]['MAE free'] for s in res},
                     'test IoU floor': res['test']['IoU floor'], 'test IoU motor': res['test']['IoU motorcycle'],
                     'test pixel acc': res['test']['pixel accuracy']})
        if k == 1:
            best = {'variant': r['variant'], 'pcd': VARIANTS[r['variant']], 'seg': seg, 't': t,
                    'sel_mIoU': res['sel']['mIoU'], 'sel_MAE': res['sel']['MAE free'], 'val_mIoU': res['val']['mIoU'], 'test_mIoU': res['test']['mIoU'],
                    'test_MAE': res['test']['MAE free']}
    fin = pd.DataFrame(rows)
    fin.to_csv(OUT / 'search_final.csv', index=False)
    (OUT / 'best_config.json').write_text(json.dumps(best, indent=2, default=float), encoding='utf-8')

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print(fin[['name', 'variant', 'channel', 'w_s', 'method', 'adaptive', 'edge', 'morph_k', 'min_area', 't',
               'sel mIoU', 'sel MAE', 'test mIoU', 'test MAE']].round(3).to_string(index=False))
    print(f'\nbest -> {OUT / "best_config.json"}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()

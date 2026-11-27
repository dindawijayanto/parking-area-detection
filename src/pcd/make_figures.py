# Figures for the presentation: every preprocessing variant with ITS OWN best segmentation, every step saved as a
# separate full-size picture (960 x 540) so it can go on a slide on its own.
#
#   python src/pcd/make_figures.py            (needs the cache + search_stage1.csv from src/pcd/search_best.py)
#   python src/pcd/make_figures.py --research (redo the per-variant search instead of reusing best_per_variant.csv)
#
# per-variant best: the variant's best stage-1 config (sel = train + val mIoU), then morphology x blob area tuned
# on sel mIoU - 0.01 * sel MAE, exactly like stage 2 of search_best.py. test is only reported, never used to choose.
#
# results/pcd_eval/figures/
#   best_per_variant.csv                       config + train/val/test numbers of every variant
#   summary_mIoU.png, summary_MAE.png          test mIoU / test MAE per variant
#   NN_<variant>/
#     info.txt                                 preprocessing, segmentation, metrics, per-camera numbers
#     confusion_test.png                       test confusion matrix
#     <camera>/01_original.jpg ... NN_error.jpg   one test frame per camera, one picture per step

import argparse
import ast
import itertools
import sys
from pathlib import Path

import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'ml'))
import pipeline as P       # noqa: E402
import search_best as S    # noqa: E402
import parking as pk       # noqa: E402

FIG = S.OUT / 'figures'
INK, INK2, GRID = '#191919', '#666666', '#e5e5e5'
BAR, BAR_BEST = '#84b9f3', '#257de0'          # reference palette: blue step 250 / series-1
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.titlecolor': INK})
BASELINE = 'baseline (raw + threshold V lama)'
STEP_FILE = {'original': 'original', 'gray world': 'gray_world', 'histogram matching': 'histogram_matching',
             'median blur on V': 'median_V', 'closing on V': 'closing_V'}


def parse_tuples(df):
    for c in ('adaptive', 'edge'):
        df[c] = [ast.literal_eval(v) if isinstance(v, str) else None for v in df[c]]
    return df


def tune(name, r, meta, split, geoms):
    seg = S.clean_seg(r)
    t = None if pd.isna(r['t']) else float(r['t'])
    xs = S.scores(S.variant_data(name, meta, False), seg['channel'], seg['w_s'])
    rs = [P.refine_mask(x, seg) for x in xs]
    best = None
    for mk, area in itertools.product(S.MORPH, S.AREA):
        s = {**seg, 'morph_k': mk, 'min_area': area}
        sel = S.evaluate(xs, rs, meta, split['sel'], s, t, True, geoms)
        score = sel['mIoU'] - S.MAE_W * sel['MAE free']
        if best is None or score > best[0]:
            best = (score, s)
    return best[1], t, xs, rs


def per_variant_best(meta, split, geoms, research):
    path = FIG / 'best_per_variant.csv'
    if path.exists() and not research:
        print(f'per-variant configs loaded from {path.name} (use --research to redo)')
        return parse_tuples(pd.read_csv(path))
    s1 = parse_tuples(pd.read_csv(S.OUT / 'search_stage1.csv')).sort_values('sel mIoU', ascending=False)
    base_seg = {'channel': 'V', 'w_s': 0.0, 'method': 'fixed', 'adaptive': None, 'edge': None,
                'morph_k': 7, 'min_area': 0}
    xs = S.scores(S.variant_data('raw', meta, False), 'V', 0.0)
    jobs = [(BASELINE, 'raw', base_seg, S.fit_threshold(xs, [None] * len(xs), meta, split['train'])[0], xs,
             [None] * len(xs))]
    for name in S.VARIANTS:
        print(f'tuning {name} ...')
        seg, t, xs, rs = tune(name, s1[s1['variant'] == name].iloc[0], meta, split, geoms)
        jobs.append((name, name, seg, t, xs, rs))
    rows = []
    for name, data, seg, t, xs, rs in jobs:
        res = {s: S.evaluate(xs, rs, meta, split[s], seg, t, True, geoms) for s in ('train', 'val', 'sel', 'test')}
        rows.append({'variant': name, 'data': data, **seg, 't': t,
                     **{f'{s} mIoU': res[s]['mIoU'] for s in res}, **{f'{s} MAE': res[s]['MAE free'] for s in res},
                     'test IoU floor': res['test']['IoU floor'], 'test IoU motor': res['test']['IoU motorcycle'],
                     'test pixel acc': res['test']['pixel accuracy']})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def pick_samples(meta):
    # one test frame per camera (the middle one), any split if a camera has no test frame
    out = []
    for v in sorted({m['video'] for m in meta}):
        cand = [i for i, m in enumerate(meta) if m['video'] == v and m['split'] == 'test'] or \
               [i for i, m in enumerate(meta) if m['video'] == v]
        out.append(cand[len(cand) // 2])
    return out


def gray3(x):
    return cv.cvtColor(np.uint8(np.clip(x, 0, 255)), cv.COLOR_GRAY2BGR)


def binary(b, roi):
    out = np.where(b, 255, 0).astype(np.uint8)
    pic = gray3(out)
    pic[roi == 0] = (60, 60, 60)            # outside the ROI: dark grey
    return pic


def error_map(img, pred, label):
    err = np.zeros_like(img)
    err[(pred == P.FLOOR) & (label == P.MOTOR)] = (0, 165, 255)     # orange: motorcycle taken as floor
    err[(pred == P.MOTOR) & (label == P.FLOOR)] = (255, 0, 255)     # magenta: floor taken as motorcycle
    return cv.addWeighted(err, 0.7, img, 0.3, 0)


def export_frame(out_dir, m, cfg, seg, t, ref, geoms):
    out_dir.mkdir(parents=True, exist_ok=True)
    img = P.load_frame(P.FRAMES_DIR / m['frame'])
    roi = m['roi']
    steps, hsv = P.run_pcd(img, roi, cfg, ref)
    pics = [(STEP_FILE.get(n, n.replace(' ', '_')), p) for n, p in steps]
    pics += [('hsv_H', gray3(hsv[:, :, 0].astype(np.float32) * 255 / 179)),
             ('hsv_S', gray3(hsv[:, :, 1])), ('hsv_V', gray3(hsv[:, :, 2]))]

    x = P.score_map(hsv[:, :, 1], hsv[:, :, 2], seg['channel'], seg['w_s'])
    tt = P.otsu(x, roi) if seg['method'] == 'otsu' or t is None else t
    thr = (x > tt) & (roi > 0)
    if seg['channel'] != 'V':
        pics.append((f'skor_{seg["channel"].replace("-", "_")}', gray3(x)))
    pics.append((f'threshold_{seg["method"]}_t{tt:.0f}', binary(thr, roi)))
    cur = thr
    if seg.get('adaptive'):
        a = P.adaptive_mask(x, *seg['adaptive'])
        pics.append((f'adaptive_b{int(seg["adaptive"][0])}', binary(a, roi)))
        cur = cur & a
    if seg.get('edge'):
        win, dmax = seg['edge']
        e = cv.Canny(cv.GaussianBlur(x, (5, 5), 0), 50, 150)
        dens = cv.boxFilter((e > 0).astype(np.float32), -1, (int(win), int(win)))
        pics.append(('canny_edges', binary(e > 0, roi)))
        pics.append((f'edge_density_w{int(win)}', gray3(dens / (2 * dmax) * 255)))
        cur = cur & (dens < dmax)
    if seg.get('adaptive') or seg.get('edge'):
        pics.append(('gabungan', binary(cur, roi)))
    morph = P.cleanup(cur.astype(np.uint8), roi, seg['morph_k'], 0)
    pics.append((f'morfologi_k{seg["morph_k"]}', binary(morph == 1, roi)))
    if seg['min_area'] > 0:
        morph = P.cleanup(cur.astype(np.uint8), roi, seg['morph_k'], seg['min_area'])
        pics.append((f'buang_blob_{seg["min_area"]}', binary(morph == 1, roi)))

    pred, _ = P.segment(hsv, roi, seg, t)
    g = geoms[m['video']]
    free_p, free_l = pk.free_spaces(pred, g), pk.free_spaces(m['label'], g)
    pics += [('prediksi', P.overlay(img, pred)), ('label', P.overlay(img, m['label'])),
             ('error', error_map(img, pred, m['label'])),
             ('slot_prediksi', pk.draw_free_spaces(img, pred, g, free_p)),
             ('slot_label', pk.draw_free_spaces(img, m['label'], g, free_l))]
    for k, (name, pic) in enumerate(pics, 1):
        cv.imwrite(str(out_dir / f'{k:02d}_{name}.jpg'), pic, [cv.IMWRITE_JPEG_QUALITY, 95])
    miou = P.cm_metrics(P.confusion(pred, m['label']))['mIoU']
    return f'{m["video"]:24s} {m["frame"]}  mIoU {miou:.3f}  slot kosong prediksi {free_p["free"]} / label {free_l["free"]}'


def confusion_figure(cm, title, path):
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    for a in range(2):
        for b in range(2):
            ax.text(b, a, f'{100 * norm[a, b]:.1f}%\n({cm[a, b]:,} px)', ha='center', va='center',
                    color='white' if norm[a, b] > 0.5 else INK, fontsize=9)
    ax.set_xticks([0, 1], ['lantai', 'motor'])
    ax.set_yticks([0, 1], ['lantai', 'motor'])
    ax.set_xlabel('prediksi')
    ax.set_ylabel('label')
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bar_chart(df, col, title, fname, fmt, best_name):
    d = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(d) + 1.2))
    ax.barh(d['label'], d[col], color=[BAR_BEST if n == best_name else BAR for n in d['variant']],
            height=0.6, edgecolor='white', linewidth=2)
    for y, v in enumerate(d[col]):
        ax.text(v, y, ' ' + fmt.format(v), va='center', fontsize=8, color=INK)
    ax.set_title(title, loc='left', fontsize=11)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, d[col].max() * 1.15)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--research', action='store_true')
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    meta = S.cached(S.CACHE / 'meta.pkl', S.build_meta, False)
    split = {s: [i for i, m in enumerate(meta) if m['split'] == s] for s in ('train', 'val', 'test')}
    split['sel'] = split['train'] + split['val']
    geoms = {v: pk.prepare_video(pk.load_roi(P.ROI_DIR / f'{v}.json')) for v in {m['video'] for m in meta}}
    refs = {gw: S.cached(S.CACHE / f'lab_ref_gw{int(gw)}.pkl', lambda: P.lab_reference(gw, 6), False)
            for gw in (False, True)}
    samples = pick_samples(meta)

    df = per_variant_best(meta, split, geoms, args.research)
    df['label'] = [f'{k}. {n}' for k, n in enumerate(df['variant'])]
    winner = df.iloc[1:].assign(score=lambda d: d['sel mIoU'] - S.MAE_W * d['sel MAE']) \
        .sort_values('score', ascending=False)['variant'].iloc[0]

    for k, r in df.iterrows():
        cfg = S.VARIANTS[r['data']]
        seg = S.clean_seg(r)
        t = None if pd.isna(r['t']) else float(r['t'])
        ref = refs[bool(cfg.get('gray_world'))] if cfg.get('hist_match') else None
        vdir = FIG / f'{k:02d}_{S.slug(r["variant"])}'
        lines = [export_frame(vdir / meta[i]['video'], meta[i], cfg, seg, t, ref, geoms) for i in samples]

        xs = S.scores(S.variant_data(r['data'], meta, False), seg['channel'], seg['w_s'])
        cm = np.zeros((2, 2), np.int64)
        for i in split['test']:
            cm += P.confusion(P.segment_score(xs[i], meta[i]['roi'], seg, t)[0], meta[i]['label'])
        confusion_figure(cm, f'confusion matrix test: {r["variant"]}', vdir / 'confusion_test.png')

        pcd_steps = [n for n in ('gray_world', 'hist_match', 'median', 'v_close') if cfg.get(n)] or ['(tanpa PCD)']
        info = [f'varian        : {r["variant"]}{"   <- TERBAIK" if r["variant"] == winner else ""}',
                f'preprocessing : {" -> ".join(pcd_steps)}   {cfg}',
                f'segmentasi    : {seg}   t = {"Otsu per gambar" if t is None or seg["method"] == "otsu" else t}',
                '',
                f'train  mIoU {r["train mIoU"]:.3f}   MAE slot {r["train MAE"]:.2f}',
                f'val    mIoU {r["val mIoU"]:.3f}   MAE slot {r["val MAE"]:.2f}',
                f'test   mIoU {r["test mIoU"]:.3f}   MAE slot {r["test MAE"]:.2f}   IoU lantai {r["test IoU floor"]:.3f}'
                f'   IoU motor {r["test IoU motor"]:.3f}   pixel acc {r["test pixel acc"]:.3f}',
                '', 'contoh frame (satu per kamera):'] + lines
        (vdir / 'info.txt').write_text('\n'.join(info), encoding='utf-8')
        print(f'{k}. {r["variant"]:34s} test mIoU {r["test mIoU"]:.3f}  test MAE {r["test MAE"]:.2f}  -> {vdir.name}')

    bar_chart(df, 'test mIoU', 'Test mIoU per varian (setelan terbaik masing-masing, lebih tinggi = lebih baik)',
              'summary_mIoU.png', '{:.3f}', winner)
    bar_chart(df, 'test MAE', 'Test MAE slot kosong per varian (lebih rendah = lebih baik)',
              'summary_MAE.png', '{:.1f}', winner)
    print(f'\nterbaik (dipilih dari train + val): {winner}\nfigures -> {FIG}')


if __name__ == '__main__':
    main()

# Web demo of the PCD pipeline (no machine learning):
#   input picture -> PCD step by step -> PCD-only segmentation (threshold on V + morphology) -> evaluation against the label
#
#   python -m streamlit run app/pcd_demo.py

import sys
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'pcd'))
sys.path.insert(0, str(ROOT / 'src' / 'ml'))
import pipeline as pcd   # noqa: E402
import parking as pk     # noqa: E402  (only the bird's-eye geometry and the free-space counter, no model)

st.set_page_config(page_title='Demo PCD - Parkir Motor', layout='wide')


def rgb(bgr):
    return cv.cvtColor(bgr, cv.COLOR_BGR2RGB)


@st.cache_resource(show_spinner='menghitung histogram referensi dari frame train ...')
def reference(gray_world, p):
    return pcd.lab_reference(gray_world, p)


@st.cache_resource(show_spinner='memilih threshold terbaik dari frame train ...')
def fixed_threshold(cfg_items, morph_k):
    cfg = dict(cfg_items)
    ref = reference(cfg['gray_world'], cfg['p']) if cfg['hist_match'] else None
    return pcd.train_threshold(cfg, ref, morph_k)


@st.cache_resource
def geometry(video_id):
    return pk.prepare_video(pk.load_roi(pcd.ROI_DIR / f'{video_id}.json'))


def plot_cm(cm, title):
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    for a in range(2):
        for b in range(2):
            ax.text(b, a, f'{100 * norm[a, b]:.1f}%\n({cm[a, b]:,} px)', ha='center', va='center',
                    color='white' if norm[a, b] > 0.5 else 'black', fontsize=9)
    ax.set_xticks([0, 1], ['lantai', 'motor'])
    ax.set_yticks([0, 1], ['lantai', 'motor'])
    ax.set_xlabel('prediksi')
    ax.set_ylabel('label')
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_hist(hsv_before, hsv_after, label, ch, name, rng):
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True)
    for ax, hsv, title in [(axes[0], hsv_before, 'sebelum PCD'), (axes[1], hsv_after, 'sesudah PCD')]:
        c = hsv[:, :, ch]
        ax.hist(c[label == pcd.FLOOR], bins=64, range=rng, density=True, alpha=0.6, color='green', label='lantai')
        ax.hist(c[label == pcd.MOTOR], bins=64, range=rng, density=True, alpha=0.6, color='red', label='motor')
        ov = pcd.overlap(c[label == pcd.FLOOR], c[label == pcd.MOTOR], value_range=rng)
        ax.set_title(f'{name} {title}: overlap {ov:.2f}', fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------------------------- sidebar: input

st.sidebar.title('1. Input')
frames = pcd.labeled_frames()
videos = sorted(p.stem for p in pcd.ROI_DIR.glob('*.json'))
source = st.sidebar.radio('sumber gambar', ['pilih dari dataset', 'upload gambar'])

img, label, video_id, name = None, None, None, None
if source == 'pilih dari dataset':
    if not frames:
        st.error('folder data/raw/frames kosong: jalankan notebooks/00 dulu atau pakai "upload gambar".')
        st.stop()
    split = st.sidebar.selectbox('split', ['train', 'val', 'test', 'semua'])
    options = [f for f in frames if split == 'semua' or pcd.split_of(f) == split]
    name = st.sidebar.selectbox('frame', options)
    video_id = pcd.video_of(name)
    img = pcd.load_frame(pcd.FRAMES_DIR / name)
    label = pcd.load_mask(name)
else:
    up = st.sidebar.file_uploader('gambar (jpg / png) dari posisi kamera yang sama', type=['jpg', 'jpeg', 'png'])
    if up is None:
        st.info('upload satu gambar di sidebar, atau pilih "pilih dari dataset".')
        st.stop()
    name = up.name
    guess = next((v for v in videos if Path(name).name.startswith(v)), videos[0])
    video_id = st.sidebar.selectbox('posisi kamera (ROI video)', videos, index=videos.index(guess))
    img = pcd.load_frame(up.getvalue())
    if st.sidebar.checkbox('align ke referensi (SIFT + RANSAC)', value=not Path(name).stem.startswith(video_id)):
        img, n_inliers = pcd.align_to_reference(img, video_id)
        st.sidebar.caption(f'alignment: {n_inliers} inlier' + ('' if n_inliers >= 40 else ' (gagal, dipakai tanpa align)'))
    label = pcd.load_mask(name)

st.sidebar.title('2. Langkah PCD')
best = pcd.load_best()
d = best['pcd'] if best else {'gray_world': True, 'hist_match': True, 'median': True}
cfg = {'gray_world': st.sidebar.checkbox('gray world (color constancy)', bool(d.get('gray_world'))),
       'hist_match': st.sidebar.checkbox('histogram matching ke frame train', bool(d.get('hist_match'))),
       'median': st.sidebar.checkbox('median blur pada V', bool(d.get('median'))),
       'v_close': st.sidebar.checkbox('closing pada V (hapus garis nat paving)', bool(d.get('v_close')))}
with st.sidebar.expander('parameter'):
    cfg['p'] = st.slider('p shades of gray (1 = gray world)', 1, 12, int(d.get('p', 6)))
    cfg['median_k'] = st.select_slider('median kernel', [3, 5, 7, 9], int(d.get('median_k', 5)))
    cfg['close_k'] = st.select_slider('kernel closing V', [3, 5, 7, 9, 11], int(d.get('close_k', 5)))
    morph_k = st.select_slider('kernel morfologi segmentasi', [1, 3, 5, 7, 9, 11], 7)
st.sidebar.caption('default = preprocessing terbaik hasil src/pcd/search_best.py' if best else
                   'jalankan python src/pcd/search_best.py untuk mencari pipeline terbaik.')
st.sidebar.title('3. Segmentasi (tanpa ML)')
methods = (['terbaik dari search (best_config.json)'] if best else []) + ['tetap (dipilih dari frame train)', 'Otsu (per gambar)']
method = st.sidebar.radio('metode segmentasi', methods)

# ----------------------------------------------------------------------------------------------- processing

roi = pcd.roi_of(video_id, img)
ref = reference(cfg['gray_world'], cfg['p']) if cfg['hist_match'] else None
steps, hsv_after = pcd.run_pcd(img, roi, cfg, ref)
hsv_before = cv.cvtColor(img, cv.COLOR_BGR2HSV)
if label is not None:
    label = label.copy()
    label[roi == 0] = pcd.IGNORE

st.title('Demo Pipeline PCD: Deteksi Area Parkir Motor Kosong')
st.caption(f'gambar: **{name}** | posisi kamera: **{video_id}** | split: **{pcd.split_of(name)}** | '
           f'label mask: **{"ada" if label is not None else "tidak ada (evaluasi label dilewati)"}** | '
           'tanpa ML / DL: segmentasi memakai threshold pada V + morfologi')

c1, c2 = st.columns(2)
roi_vis = img.copy()
contours, _ = cv.findContours(roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(roi_vis, contours, -1, (255, 200, 0), 3)
c1.image(rgb(roi_vis), caption='input + ROI (biru = area parkir yang dianalisis)', width='stretch')
if label is not None:
    c2.image(rgb(pcd.overlay(img, label)), caption='label: hijau = lantai, merah = motor, abu = ignore',
             width='stretch')

# ----------------------------------------------------------------------------------------------- PCD steps

st.header('Langkah PCD')
cols = st.columns(len(steps))
for col, (step, pic) in zip(cols, steps):
    col.image(rgb(pic), caption=step, width='stretch')

st.subheader('Kanal HSV sebelum vs sesudah PCD')
cols = st.columns(3)
for col, (ch, nm) in zip(cols, [(0, 'H'), (1, 'S'), (2, 'V')]):
    col.image(np.hstack([hsv_before[:, :, ch], hsv_after[:, :, ch]]), caption=f'{nm}: kiri sebelum, kanan sesudah',
              width='stretch', clamp=True)

final = cv.cvtColor(hsv_after, cv.COLOR_HSV2BGR)
st.download_button('download gambar hasil PCD', cv.imencode('.png', final)[1].tobytes(),
                   file_name=f'{Path(name).stem}_pcd.png', mime='image/png')

# ----------------------------------------------------------------------------------------------- evaluation

st.header('Evaluasi')
st.subheader('a. Statistik citra (di dalam ROI)')
stats = pd.DataFrame({'sebelum PCD': pcd.image_stats(hsv_before, roi, label),
                      'sesudah PCD': pcd.image_stats(hsv_after, roi, label)})
stats['perubahan'] = stats['sesudah PCD'] - stats['sebelum PCD']
st.dataframe(stats.style.format('{:.2f}'), width='stretch')
st.caption('contrast = simpangan baku V, entropy = banyaknya informasi tingkat keabuan, sharpness = varians Laplacian. '
           'median V lantai yang mirip di semua kondisi cuaca berarti cahaya sudah lebih seragam.')

if label is None:
    st.info('gambar ini tidak punya label mask, jadi evaluasi b dan c dilewati. pilih frame dari dataset untuk evaluasi lengkap.')
    st.stop()

st.subheader('b. Seberapa mudah lantai dibedakan dari motor')
sep = pd.DataFrame({'overlap sebelum': pcd.separability(hsv_before, label),
                    'overlap sesudah': pcd.separability(hsv_after, label)})
sep['perubahan'] = sep['overlap sesudah'] - sep['overlap sebelum']
st.dataframe(sep.style.format('{:.3f}'), width='stretch')
st.caption('overlap = luas bagian histogram lantai dan motor yang tumpang tindih. 0 = terpisah sempurna, 1 = sama persis. '
           'nilai negatif di kolom perubahan berarti PCD membuat lantai lebih mudah dibedakan.')
st.pyplot(plot_hist(hsv_before, hsv_after, label, 2, 'V', (0, 256)))

st.subheader('c. Segmentasi PCD murni (threshold pada V + morfologi) vs label')
raw_cfg = {'gray_world': False, 'hist_match': False, 'median': False, 'v_close': False, 'p': cfg['p']}
if method.startswith('terbaik'):
    t_before, _ = fixed_threshold(tuple(sorted(raw_cfg.items())), morph_k)
    st.caption(f"tanpa PCD = threshold V lama (t = {t_before:.0f}). dengan PCD = hasil search: {best['variant']} | "
               f"{best['seg']} | val mIoU {best['val_mIoU']:.3f}, test mIoU {best['test_mIoU']:.3f}")
elif method.startswith('tetap'):
    t_before, miou_before = fixed_threshold(tuple(sorted(raw_cfg.items())), morph_k)
    t_after, miou_after = fixed_threshold(tuple(sorted(cfg.items())), morph_k)
    st.caption(f'threshold dipilih dari semua frame train: tanpa PCD t = {t_before:.0f} (train mIoU {miou_before:.3f}), '
               f'dengan PCD t = {t_after:.0f} (train mIoU {miou_after:.3f}). angka yang sama dipakai untuk semua gambar.')
else:
    t_before, t_after = None, None
pred_before, t_before = pcd.threshold_segment(hsv_before, roi, t_before, morph_k)
if method.startswith('terbaik'):
    pred_after, t_after = pcd.segment(hsv_after, roi, best['seg'], best['t'])
else:
    pred_after, t_after = pcd.threshold_segment(hsv_after, roi, t_after, morph_k)
g = geometry(video_id)
free_label = pk.free_spaces(label, g)
res_before, res_after = pk.free_spaces(pred_before, g), pk.free_spaces(pred_after, g)

c1, c2, c3 = st.columns(3)
c1.image(rgb(pk.draw_free_spaces(img, label, g, free_label)), caption=f'label: {free_label["free"]} slot kosong',
         width='stretch')
c2.image(rgb(pk.draw_free_spaces(img, pred_before, g, res_before)),
         caption=f'tanpa PCD (t = {t_before:.0f}): {res_before["free"]} slot kosong', width='stretch')
c3.image(rgb(pk.draw_free_spaces(img, pred_after, g, res_after)),
         caption=f'dengan PCD (t = {t_after:.0f}): {res_after["free"]} slot kosong', width='stretch')

cm_before, cm_after = pcd.confusion(pred_before, label), pcd.confusion(pred_after, label)
c1, c2, c3 = st.columns([1, 1, 1.3])
c1.pyplot(plot_cm(cm_before, 'confusion matrix tanpa PCD'))
c2.pyplot(plot_cm(cm_after, 'confusion matrix dengan PCD'))
metrics = pd.DataFrame({'tanpa PCD': pcd.cm_metrics(cm_before), 'dengan PCD': pcd.cm_metrics(cm_after)})
metrics.loc['slot kosong (label ' + str(free_label['free']) + ')'] = [res_before['free'], res_after['free']]
metrics.loc['error slot kosong'] = [abs(res_before['free'] - free_label['free']), abs(res_after['free'] - free_label['free'])]
c3.dataframe(metrics.style.format('{:.3f}'), width='stretch')

err = np.zeros_like(img)
err[(pred_after == pcd.FLOOR) & (label == pcd.MOTOR)] = (0, 165, 255)
err[(pred_after == pcd.MOTOR) & (label == pcd.FLOOR)] = (255, 0, 255)
st.image(rgb(cv.addWeighted(err, 0.7, img, 0.3, 0)), width='stretch',
         caption='piksel salah (dengan PCD): oranye = motor dikira lantai, magenta = lantai dikira motor')
st.caption('segmentasi ini sengaja tanpa machine learning: lantai paving lebih terang, jadi piksel dengan V > t dianggap '
           'lantai, lalu opening dan closing. threshold tetap hanya bekerja kalau cahaya semua gambar sudah seragam, '
           'dan itulah gunanya normalisasi cahaya (gray world + histogram matching). Otsu memilih threshold sendiri '
           'per gambar, jadi normalisasi cahaya tidak banyak berpengaruh di sana. jumlah slot kosong dihitung dengan bird\'s-eye view + opening '
           'dengan tapak satu motor, sama seperti notebooks/01.')

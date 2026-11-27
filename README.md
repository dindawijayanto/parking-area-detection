# Deteksi Area Parkir Motor Kosong

Segmentasi lantai vs motor dari kamera atas, lalu menghitung jumlah slot parkir kosong.
Bagian **PCD** (pengolahan citra digital, tanpa ML) ada di `src/pcd/`. Bagian **ML / DL** memakai dataset `.pkl` yang dibuat dari PCD.

## Struktur folder

```
data/
  raw/frames/            frame hasil alignment (*.jpg, ~77 MB)       <- TIDAK di git, minta ke tim
  annotations/
    masks/               label mask per frame (0 lantai, 1 motor, 255 ignore)
    roi/                 ROI per video (json)
    references/          frame referensi untuk alignment
    videos.csv           waktu & cuaca per video
  processed/             dataset_seg.pkl, dataset_pcd.pkl              <- dibuat sendiri, TIDAK di git
notebooks/
  00_video_to_frames     video -> frame + alignment
  01_roi_and_masks       ROI, label mask, dataset_seg.pkl
  02_pcd_color_space     eksperimen PCD awal
  03_ml_classical        segmentasi ML klasik (random forest, xgboost)
src/
  pcd/pipeline.py        semua langkah PCD + segmentasi PCD + evaluasi
  pcd/export_dataset.py  bikin dataset_seg.pkl & dataset_pcd.pkl untuk ML / DL
  pcd/search_best.py     mencari kombinasi PCD + segmentasi terbaik
  pcd/make_figures.py    gambar per langkah per varian (untuk presentasi)
  pcd/draw_roi.py, label_masks.py   tool anotasi
  ml/                    kode ML (parking.py, predict.py)
app/pcd_demo.py          demo web (streamlit)
results/pcd_eval/        hasil evaluasi PCD (best_config.json, search_*.csv)
```

## Setup

```
pip install -r requirements.txt
```

Taruh frame (`*.jpg`) di `data/raw/frames/`. Frame tidak ada di git, jadi minta zip-nya ke tim
(~77 MB, jauh lebih kecil daripada mengirim `.pkl` yang 200+ MB). Label, ROI, dan referensi sudah ada di git.

## Bikin dataset untuk ML / DL (cukup sekali)

```
python src/pcd/export_dataset.py
```

Hasilnya di `data/processed/`:

| File | `X` | Isi lain |
|---|---|---|
| `dataset_seg.pkl` | frame BGR 960×540 uint8, **tanpa** PCD | `Y`, `meta`, `geo`, `classes` |
| `dataset_pcd.pkl` | HSV 960×540 uint8 **setelah PCD terbaik** | sama + `pcd_config`, `lab_reference_cdf`, `pcd_mask` |

- `Y`: label mask (0 lantai, 1 motor, 255 ignore)
- `meta`: DataFrame `frame, video_id, date, time, weather, who, t_sec, split, floor % of ROI, free spaces`.
  Split dibagi **per hari** (test 4 & 12 Nov, val 5 & 10 Nov, sisanya train). Jangan diacak.
- `geo[video_id]`: geometri bird's-eye untuk `parking.free_spaces(mask, geo[vid])` (hitung slot kosong)
- `pcd_mask`: hasil segmentasi PCD murni tiap frame. Bisa dipakai sebagai baseline atau fitur tambahan.
- `dataset_seg.pkl` identik dengan buatan notebook 01, jadi `notebooks/03_ml_classical` jalan tanpa diubah.

Opsi:
```
python src/pcd/export_dataset.py --only pcd                    # hanya dataset_pcd.pkl
python src/pcd/export_dataset.py --pcd "gw + hm + median"      # varian PCD lain (lihat VARIANTS di search_best.py)
```

Memakainya:
```python
import pickle
d = pickle.load(open('data/processed/dataset_pcd.pkl', 'rb'))
X, Y, meta, geo = d['X'], d['Y'], d['meta'], d['geo']
train = meta.index[meta['split'] == 'train']
```

### Di Google Colab
Upload zip frame ke Drive, lalu:
```python
!git clone https://github.com/dindawijayanto/parking-area-detection.git
%cd parking-area-detection
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p data/raw && unzip -q /content/drive/MyDrive/frames.zip -d data/raw/   # hasilnya data/raw/frames/*.jpg
!python src/pcd/export_dataset.py
```

## Pipeline PCD terbaik

```
frame -> resize 960x540 -> ROI
  -> gray world (shades of gray, p=6) -> histogram matching Lab ke frame train
  -> HSV -> median V (5) -> closing V (5)                          = preprocessing (X di dataset_pcd.pkl)
  -> skor V - S -> Otsu -> kerapatan tepi Canny
  -> opening + closing -> buang blob kecil                         = segmentasi PCD (pcd_mask)
```

Setelan persisnya ada di `results/pcd_eval/best_config.json`. Hasil test: mIoU 0.743, MAE slot kosong 3.7
(metode lama, threshold V tanpa PCD: 0.582 / 12.2). CLAHE sudah tidak dipakai karena menurunkan hasil.

## Script PCD lain (opsional)

```
python src/pcd/search_best.py          # cari ulang kombinasi terbaik (~20-30 menit, cache .pkl ~800 MB)
python src/pcd/make_figures.py         # gambar per langkah per varian -> results/pcd_eval/figures/
python -m streamlit run app/pcd_demo.py
```

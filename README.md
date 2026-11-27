# Deteksi Area Parkir Motor Kosong

Segmentasi lantai vs motor dari kamera atas, lalu menghitung jumlah slot parkir yang masih kosong.

- **PCD** (pengolahan citra digital, tanpa ML): `src/pcd/`
- **ML / DL**: memakai dataset `.pkl` yang dibangun dari hasil PCD

## Mulai cepat

```bash
git clone https://github.com/dindawijayanto/parking-area-detection.git
cd parking-area-detection
pip install -r requirements.txt

# 1. Ekstrak frame -> data/raw/frames/*.jpg (93 frame)
unzip -q frames.zip -d data/raw/

# 2. Bangun dataset ML / DL -> data/processed/*.pkl
python src/pcd/export_dataset.py
```

Di Windows (PowerShell), langkah 1 bisa diganti dengan:

```powershell
Expand-Archive frames.zip -DestinationPath data\raw
```

Label mask, ROI, dan frame referensi sudah tersedia di repo, jadi dua langkah di atas sudah cukup.

### Google Colab

```python
!git clone https://github.com/dindawijayanto/parking-area-detection.git
%cd parking-area-detection
!pip install -q -r requirements.txt
!mkdir -p data/raw && unzip -q frames.zip -d data/raw/
!python src/pcd/export_dataset.py
```

## Dataset untuk ML / DL

`python src/pcd/export_dataset.py` menghasilkan dua file di `data/processed/`:

| File | `X` | Isi lain |
|---|---|---|
| `dataset_seg.pkl` | frame BGR 960×540 uint8, **tanpa** PCD | `Y`, `meta`, `geo`, `classes` |
| `dataset_pcd.pkl` | HSV 960×540 uint8 **setelah pipeline PCD terbaik** | sama + `pcd_config`, `lab_reference_cdf`, `pcd_mask` |

Keterangan isi:

- `Y`: label mask (0 = lantai, 1 = motor, 255 = ignore)
- `meta`: DataFrame berisi `frame, video_id, date, time, weather, who, t_sec, split, floor % of ROI, free spaces`.
  Split dibagi **per hari** (test: 4 & 12 Nov, val: 5 & 10 Nov, sisanya train). Jangan diacak ulang agar tidak terjadi kebocoran data.
- `geo[video_id]`: geometri bird's-eye untuk menghitung slot kosong lewat `parking.free_spaces(mask, geo[vid])`
- `pcd_mask`: hasil segmentasi PCD murni tiap frame, bisa dipakai sebagai baseline atau fitur tambahan
- `dataset_seg.pkl` identik dengan keluaran notebook 01, sehingga `notebooks/03_ml_classical` bisa langsung dijalankan

Contoh pemakaian:

```python
import pickle

d = pickle.load(open('data/processed/dataset_pcd.pkl', 'rb'))
X, Y, meta, geo = d['X'], d['Y'], d['meta'], d['geo']

train_idx = meta.index[meta['split'] == 'train']
val_idx   = meta.index[meta['split'] == 'val']
test_idx  = meta.index[meta['split'] == 'test']
```

Opsi tambahan:

```bash
python src/pcd/export_dataset.py --only pcd                 # hanya dataset_pcd.pkl
python src/pcd/export_dataset.py --pcd "gw + hm + median"   # varian PCD lain (lihat VARIANTS di search_best.py)
```

## Pipeline PCD terbaik

```
frame -> resize 960x540 -> ROI
  -> gray world (shades of gray, p=6) -> histogram matching Lab ke frame train
  -> HSV -> median V (5) -> closing V (5)              = preprocessing (X di dataset_pcd.pkl)
  -> skor V - S -> Otsu -> kerapatan tepi Canny
  -> opening + closing -> buang blob kecil             = segmentasi PCD (pcd_mask)
```

Konfigurasi lengkap ada di `results/pcd_eval/best_config.json`.

| Metode | mIoU (test) | MAE slot kosong (test) |
|---|---|---|
| Threshold V tanpa PCD (baseline lama) | 0.582 | 12.2 |
| **Pipeline PCD terbaik** | **0.743** | **3.7** |

CLAHE tidak dipakai karena justru menurunkan hasil.

## Struktur folder

```
frames.zip               93 frame hasil alignment (~77 MB), ekstrak ke data/raw/
data/
  raw/frames/            hasil ekstrak frames.zip (tidak di-commit)
  annotations/
    masks/               label mask per frame (0 lantai, 1 motor, 255 ignore)
    roi/                 ROI per video (json)
    references/          frame referensi untuk alignment
    videos.csv           waktu & cuaca per video
  processed/             dataset_seg.pkl, dataset_pcd.pkl (dibangun lokal, tidak di-commit)
notebooks/
  00_video_to_frames     video -> frame + alignment
  01_roi_and_masks       ROI, label mask, dataset_seg.pkl
  02_pcd_color_space     eksperimen PCD awal
  03_ml_classical        segmentasi ML klasik (random forest, xgboost)
src/
  pcd/pipeline.py        seluruh langkah PCD, segmentasi PCD, dan evaluasi
  pcd/export_dataset.py  membangun dataset_seg.pkl & dataset_pcd.pkl
  pcd/search_best.py     pencarian kombinasi PCD + segmentasi terbaik
  pcd/make_figures.py    gambar per langkah per varian (untuk presentasi)
  pcd/draw_roi.py, label_masks.py   tool anotasi
  ml/                    kode ML (parking.py, predict.py)
app/pcd_demo.py          demo web (Streamlit)
results/pcd_eval/        hasil evaluasi PCD (best_config.json, search_*.csv)
```

## Script PCD lainnya (opsional)

```bash
python src/pcd/search_best.py          # cari ulang kombinasi terbaik (~20-30 menit, cache ~800 MB)
python src/pcd/make_figures.py         # gambar per langkah -> results/pcd_eval/figures/
python -m streamlit run app/pcd_demo.py
```

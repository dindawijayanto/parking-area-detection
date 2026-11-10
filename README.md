# Parking Space Availability Detection at FILKOM, Universitas Brawijaya

Final project for **Digital Image Processing (PCD) and Computer Vision (Viskom)**, topic #5: *Deteksi Ketersediaan Tempat Parkir*.

**Contents:** [1. Team](#1-team) · [2. Job desk](#2-job-desk) · [3. Project summary](#3-project-summary) · [4. Video capture](#4-video-capture) · [5. From video to dataset](#5-from-video-to-dataset) · [6. Methods](#6-methods) · [7. Timeline](#7-timeline) · [8. Repository](#8-repository-structure) · [9. Results](#9-results) · [10. Exam questions](#10-questions-to-prepare-for) · [11. Contribution log](#11-contribution-log) · [12. Links](#12-links)

---

## 1. Team

| Name | Role |
|---|---|
| **Dinda** (team leader) | Student 1: Data and image processing (PCD) |
| **Keiza** | Student 2: Features and classic machine learning |
| **Tobi** | Student 3: Deep learning, computer vision and demo |

All three record from **the same spot** (the only accessible one), at different days and times.

---

## 2. Job desk

Each person has a **main responsibility**, but the course rules say:
- **All three members record data.** One person may not record everything.
- **Everyone must understand the whole pipeline.** The lecturer can ask anyone about any part. For example, Tobi must be able to explain CLAHE, and Dinda must be able to explain how MobileNetV2 works.
- Grades are partly individual, so **each person keeps evidence of their own work** (see [section 11](#11-contribution-log)).

### 2.1 Shared by everyone
| Task | Details |
|---|---|
| Recording | 3 clips per weekday each (morning, afternoon, evening), from the shared spot. See [section 4](#4-video-capture). |
| Boxes | Whoever recorded a video draws its boxes (`draw_slots.py`, see [section 5](#5-from-video-to-dataset)). |
| Labeling | **Whoever recorded a clip labels it.** Dinda reviews all labels. |
| Presentations | Each member presents their own part at every progress meeting. |
| Report | Each member writes the report sections for their own part. |

### 2.2 Dinda: Student 1, Data and image processing (+ team leader)
| | |
|---|---|
| **Main job** | Make sure the data is good, and design the PCD steps that everything else uses. |
| **Tasks** | Reference picture and slot boxes · extract and align frames · labels and `dataset.pkl` · review labels · split by day · EDA · compare PCD techniques (perspective correction, color spaces, CLAHE, filters, edges, thresholding, morphology) · choose the PCD pipeline · PCD ablation · error analysis |
| **Hands over to K and T** | Slot crops (`raw/` and later `pcd/`), `crops.csv`, `dataset.pkl`, split by day |
| **Leader duties** | Merge the slides and report, submit to BRONE, publish the dataset |
| **Report sections** | Dataset and data collection, PCD methods, PCD ablation, error analysis |

### 2.3 Keiza: Student 2, Features and classic machine learning
| | |
|---|---|
| **Main job** | Turn slot crops into features and classify them with classic ML. **Experiment A.** |
| **Tasks** | Features: edge density, HOG, LBP, color statistics (GLCM optional) · normalization, optional PCA/selection · **SVM, Random Forest, KNN** (at least 2 required) · confusion matrices · feature ablation · comparison table A vs B vs C · final results and conclusions · Overleaf with the course LaTeX template |
| **Report sections** | Feature extraction, ML methods and results, comparison and conclusions |

### 2.4 Tobi: Student 3, Deep learning, computer vision and demo
| | |
|---|---|
| **Main job** | Train a DL model, test whether PCD helps it, build the demo. **Experiments B and C** (optional D). |
| **Tasks** | GitHub repo · early DL run · **MobileNetV2** on raw crops (**B**) and PCD crops (**C**) with identical settings · optional YOLOv8 (**D**, mAP) · demo: frame in → empty slots green, occupied red, "X of Y free" · test the demo on clips not used in training |
| **Report sections** | DL methods and results, raw vs PCD analysis, system and demo |

---

## 3. Project summary

**Problem.** At busy hours, finding a free motorcycle space at FILKOM takes time. Riders circle the parking area while spaces sit empty in another row.

**Benefit.** A camera that already overlooks the parking area could show how many spaces are free, in real time and without extra sensors in every space. This saves riders time and reduces traffic inside the parking area.

**Object.** The motorcycle parking area seen from **Lt 2 Gedung G** (2nd floor, Building G), FILKOM UB. All videos come from one spot, handheld.

**Goals.** For every frame of the parking area, the system:
1. decides for every parking space whether it is **empty** or **occupied**,
2. reports how many spaces are free, and
3. lets us measure **how much PCD helps** classic ML and deep learning.

**Classes.** `0 = empty`, `1 = occupied`, per parking space. Spaces that can't be judged are labeled `-1 = unclear` and left out.

**Metrics.**
- Per space: Accuracy, Precision, Recall, **F1 (empty)**, **macro-F1**, confusion matrix. Macro-F1 is the main metric because most spaces are occupied, so accuracy alone looks too good.
- Per frame: **MAE of the free-space count**
- mAP@0.5, only for the optional YOLO experiment

**Pipeline:**
```
Videos from Lt 2 Gedung G (one spot, handheld)
   │  extract 1 frame / 15 s  (+ frames where a space changes)
   ▼
[PCD] image registration (align every frame to its video's reference: SIFT + RANSAC homography)
      → slot boxes (per video) → perspective-correct each slot → grayscale / HSV / LAB
      → CLAHE → Gaussian/median blur → Canny / adaptive threshold → morphology
   │
   ├──► [Features] edge density, HOG, LBP, color statistics
   │        └──► [ML] SVM, Random Forest, KNN          → Experiment A
   ├──► [DL] MobileNetV2 on RAW slot crops            → Experiment B
   └──► [DL] MobileNetV2 on PCD slot crops            → Experiment C
                │
                ▼
     Evaluation (split by day) → A vs B vs C → PCD ablation → Demo
```

| ID | Input | Method | Question it answers |
|---|---|---|---|
| **A** | PCD + handcrafted features | SVM, RF, KNN | How well does classic ML do? |
| **B** | Raw slot crops | MobileNetV2 | How well does DL do without PCD? |
| **C** | PCD slot crops | MobileNetV2 (same settings as B) | Does PCD help DL? |
| D *(optional)* | Full raw frames | YOLOv8n, detections mapped to slots | Bonus: detection + mAP |

**Privacy.** We film from the 2nd floor, so plates and faces are small. Anything readable is blurred before publishing.

---

## 4. Video capture

### 4.1 Rules
| Item | Rule |
|---|---|
| Location | **Lt 2 Gedung G**, looking down at the parking area |
| Spot | Everyone uses **the same spot** (the only accessible one). Start every video pointing at the same landmarks: the grass bed with the small tree in the middle, and the white lines of the front rows. |
| Angle | Looking down at about **45–70°**, so every space is visible and no motorcycle hides another |
| When | Every weekday, 3 time slots: **morning 07:00–09:00**, **afternoon 12:00–14:00**, **evening 16:00–18:00** |
| How many | 1 clip per member per time slot → 3 clips per member per day, **9 clips per day** for the team |
| Clip length | **Max 2 minutes.** Keep your hands still, or rest the phone on the railing. |
| Camera | Landscape, 1080p 30 fps, 1× zoom, stabilization on, focus locked, no filters |
| Proof | One selfie with the parking area behind you per day (for the report, not the dataset) |
| Privacy | No close-ups of people or plates |
| Can't make it? | Swap with a teammate **before** the slot and write it in the Notes column of the Schedule sheet |

### 4.2 Upload and file names
Upload the **original files** the same day to the shared Drive, then copy them into `data/raw/`. Preferred name:
```
raw_{DD-MM-YYYY}_{time}_{name}.mov        e.g.  raw_04-11-2026_evening_Keiza.mov
```
- `{time}`: `morning`, `afternoon` or `evening`. **Always these three words.**
- Add the weather if it's special: `afternoonRain`.

The exact name doesn't matter much: every video gets one row in **`data/raw/videos.csv`** (`file, date, time, weather, who`). The script adds new videos there automatically, with a guess from the file name. **Open the file and check the guess** for your own videos.

### 4.3 Numbers
| Unit | Amount |
|---|---|
| One 2-minute clip | 3,600 raw frames → about **8–12 kept frames** |
| One member, one day | 3 clips, 6 minutes of video, about 750 MB |
| Whole team, one day | 9 clips → about 70–100 frames |
| Each frame | × number of boxes in the ROI (about 25–40) → slot crops |

Total crops = clips × ~8 frames × boxes in the ROI. Fill in the real numbers after the boxes are drawn.

> Frames from the same clip are almost identical. More frames per clip add little; **more days and time slots** add real variety.

---

## 5. From video to dataset

All scripts are in `src/pcd/`. Run them **from the project folder** (`project-akhir/`), with the videos in `data/raw/`.

```
.mov ─► frames (1 every 15 s, resized to 1920×1080)
     ─► aligned to the video's own reference frame   (the camera was handheld)
     ─► boxes drawn once per video, on its reference  (fit every frame of that video)
     ─► each box labeled 0 / 1
     ─► each box cut out + straightened ─► .png crops + crops.csv + dataset.pkl
```

| Step | How often | Command | Output |
|---|---|---|---|
| 1. Extract + align frames | Once for all videos; re-run when new videos arrive (old ones are skipped) | `python src/pcd/extract_frames.py` | `data/raw/videos.csv` (check it!), `data/raw/frames/20261104_evening_keiza_t015.jpg`, `frames_index.csv`, `data/annotations/references/<video id>.jpg`, `data/raw/align_check/<video id>.jpg` |
| 2. Draw boxes | **Once per video** (~3 min; copy from a previous video with `--from`) | `python src/pcd/draw_slots.py 20261104_evening_keiza --from 20261103_afternoon_keiza` | `data/annotations/slots/<video id>.json` |
| 3. *(optional)* Extra frames on space change | After step 2 | `python src/pcd/extract_frames.py --redo --changes 12` | Adds frames where a space changes |
| 4. Label | Each person labels their own videos | `python src/pcd/label_slots.py --who keiza` | `data/annotations/labels.csv` |
| 5. Crop + package | Once at the end; re-run when labels change | `python src/pcd/crop_slots.py` | `data/processed/raw/crops/*.png`, `crops.csv`, `data/processed/dataset.pkl` |

`python src/pcd/draw_slots.py` with no video id lists all videos and marks the ones that already have boxes.

### 5.1 Alignment (image registration)
The phone is handheld, so the picture shifts **inside a video** (up to ~200 px in Keiza's 3 Nov clip) and **between videos** (different person, different day, slightly different standing point).

- **Inside a video (automatic, works):** the sharpest frame of the first 3 seconds becomes the video's **reference**. Every other frame is matched to it with **SIFT keypoints → ratio test → RANSAC homography** and warped onto it. Same light and same scene, so 1,400+ points match. Before/after pictures are in `data/raw/align_check/`.
- **Between videos (not automatic):** we tried SIFT + RANSAC, a similarity model with gated refinement, ECC on gradient images, and floor-only matching. None was reliable: motorcycles move between days, the light changes (sunny vs dusk vs rain), the floor tiles all look alike, and each person stood slightly differently, so the near floor and the far tent can't both line up with one homography (**parallax**). So **each video gets its own boxes**. To save time, `--from` copies the boxes of a previous video, and you shift them into place with `i/j/k/l` (move) and `[` / `]` (shrink/grow), then fix single boxes.

This is a good discussion point for the report: why registration works within one video but not across videos.

### 5.2 Boxes (slot outlines)
- Only the **ROI**: the clear front rows, about 25–40 boxes. The far rows are too small and partly behind trees. Mention this choice in the report.
- Use the **same rows and the same box numbering in every video**, so slot 7 is always the same place.
- There are no painted lines per motorcycle. Split each row into equal **zones about one motorcycle wide**.
- Click the **4 corners** of each box (any order). Go **row by row, left to right**.
- Right-click deletes a box, `u` undoes, `i/j/k/l` moves all boxes, `[` / `]` shrinks/grows all boxes, `q` saves and quits.

### 5.3 Labeling
- `python src/pcd/label_slots.py --who keiza` (or `--date 20261104`).
- Click a box to toggle **green (empty) ↔ red (occupied)**. Right-click marks it **gray (unclear)**.
- `space` saves and goes to the next frame. Inside the same video it **starts with the previous frame's labels**, so you only click what changed. A new video starts as all occupied and shows "NEW VIDEO: check every box".
- Boxes that fall in the black (unseen) area after alignment are set to unclear automatically.
- Videos without boxes yet are skipped, with a message.
- `e` / `a` sets everything empty / occupied, `b` goes back, `q` quits. The next run continues where you stopped.
- Agree on the unclear cases first in `docs/labeling_rules.md` (a motorcycle across two zones, a person walking through, a box hidden by a branch).
- **Dinda** reviews a sample of every day and fixes mistakes.

### 5.4 File formats
| What | Format | Why |
|---|---|---|
| Video list | `data/raw/videos.csv` | Date, time, weather and who for every video |
| Frames | `.jpg` (aligned, 1920×1080) | Part of the public dataset, together with the original videos |
| References | `annotations/references/<video id>.jpg` | The picture each video is aligned to |
| Boxes | `annotations/slots/<video id>.json` | Drawn once per video, fit every frame of that video |
| Labels | `labels.csv` (`image, date, time, who, slot_id, label`) | Readable, easy to fix by hand |
| Slot crops | `.png` (224×224) + `crops.csv` | Input for DL (B, C) and PCD |
| Training package | `dataset.pkl` | One file for ML: `X` (64×64 crops), `y` (labels), `meta` (date, time, weather, who…) |

The `.jpg`, `.json` and `.csv` files are the dataset. `dataset.pkl` is a convenience copy for training and can always be rebuilt with step 5.

### 5.5 Split by day, never randomly
Frames from the same clip, or the same day, look almost identical. A random split puts near-copies of test images into training and gives a fake ~100% score.
- **Test:** whole days that are never used for training or tuning. Every time slot must appear in the test days.
- **Validation:** one other day.
- **Train:** all remaining days.
- **Bonus test:** train on sunny clips, test on rain and evening clips. This shows whether PCD (e.g. CLAHE) makes the models more robust to light and weather.

---

## 6. Methods

### 6.1 PCD (Dinda)
| Step | Technique | Why |
|---|---|---|
| Registration | Align every frame to its video's reference frame: SIFT + RANSAC homography | Handheld camera; without it the boxes don't fit |
| Geometric | Perspective correction of each slot (`cv2.getPerspectiveTransform` from its 4 corners) | Slots far from the camera look small and skewed; this makes every slot the same square shape |
| ROI | Crop each slot, resize (64×64 for ML, 224×224 for DL) | Fixed-size input |
| Color | Grayscale, HSV, LAB | Shadows change brightness (V, L) more than hue |
| Enhancement | Histogram equalization vs **CLAHE**, gamma | Morning and afternoon sun gives uneven light and building shadows |
| Noise | Gaussian, median, bilateral | Remove paving texture noise |
| Edges / segmentation | Canny, Sobel, adaptive threshold, Otsu | Motorcycles produce many strong edges; empty ground doesn't |
| Morphology | Opening, closing, dilation | Remove small noise, join broken edges |

Deliverables: a comparison figure (the same slot under each PCD setting) and the chosen final PCD pipeline, with reasons.

### 6.2 Features + ML (Keiza)
- **Edge density:** fraction of edge pixels in the slot
- **HOG:** shape and gradient structure
- **LBP histogram:** texture (ground vs vehicle)
- **Color/gray statistics:** mean, std, histograms in HSV or LAB
- **GLCM** *(optional)*: contrast, homogeneity, energy

`StandardScaler` → optional PCA or feature selection → **SVM (RBF)**, **Random Forest**, **KNN**, tuned on the validation day. Use `class_weight="balanced"` or similar, since occupied spaces dominate. Also: RF feature importance and a feature-group ablation.

### 6.3 DL + demo (Tobi)
- **MobileNetV2**, ImageNet-pretrained, fine-tuned on slot crops (runs on Colab)
- **B (raw)** and **C (PCD)** use the same split, augmentation, epochs and random seed. Only the input changes.
- Light augmentation: small brightness/contrast changes, horizontal flip, small rotation
- Optional **D**: YOLOv8n on full frames. A detected vehicle that overlaps a slot outline above an IoU threshold marks that slot occupied.
- **Demo:** Streamlit app or notebook. Upload a frame from the spot (aligned to the reference automatically) → slots drawn green (empty) or red (occupied) → "X of Y spaces free".

---

## 7. Timeline

### 7.1 Key dates
| Milestone | Date | Must show |
|---|---|---|
| Recording starts | Tue 3 Nov | First day of recording at Lt 2 Gedung G |
| **Progress 1** (~20%) | **Wed 11 Nov** | Problem, pipeline, job desk, first frames, data plan |
| **Progress 2** (~55%) | **Wed 18 Nov** | Dataset labeled, PCD comparison, at least 2 ML models |
| **Progress 3** (~80–85%) | **Wed 25 Nov** | Raw+DL vs PCD+DL vs PCD+ML, error analysis, demo |
| **Final** | **Wed 2 Dec** | Live demo, ablation, report, code, dataset, submit on BRONE |

### 7.2 Tasks
| # | Task | PIC | Deadline |
|---|---|---|---|
| 1 | Script: video → frames, labels.csv and .pkl | Dinda | 6 Nov |
| 2 | Recording finished | All | 8 Nov |
| 3 | Progress 1 slides and rehearsal | All | 10 Nov |
| 4 | All frames labeled and reviewed | All | 10 Nov |
| 5 | Split by day, PCD comparison, chosen pipeline | Dinda | 17 Nov |
| 6 | Features, at least 2 ML models, confusion matrices | Keiza | 17 Nov |
| 7 | Early DL run | Tobi | 17 Nov |
| 8 | Exp B (raw) and Exp C (PCD) with MobileNetV2 | Tobi | 24 Nov |
| 9 | Comparison table A vs B vs C | Keiza | 24 Nov |
| 10 | Error analysis with examples | Dinda | 24 Nov |
| 11 | First working demo | Tobi | 24 Nov |
| 12 | PCD ablation | Dinda | 30 Nov |
| 13 | Final results and conclusions | Keiza | 30 Nov |
| 14 | Publish dataset (video, frames, labels, .pkl) | Dinda | 30 Nov |
| 15 | Report (own sections) and final slides | All | 30 Nov |
| 16 | Submit everything on BRONE | Dinda | 3 Dec |

Status is tracked in the **Tasks** sheet of `parkir parkir filkom.xlsx`, which also sends reminder emails.

### 7.3 Milestone checklists

**Progress 1: Proposal and first data (11 Nov)**
- [ ] Problem, benefit, object, goals, classes — **Dinda**
- [ ] Pipeline diagram (PCD → features → ML → DL) — **Tobi**
- [ ] First frames (sunny / evening / rain), one frame with boxes, alignment before/after, proof selfies — **Dinda**
- [ ] Data plan: clips recorded so far, planned total, variation, split by day — **Dinda**
- [ ] Evaluation plan (metrics, experiments A/B/C) — **Keiza**
- [ ] Job desk slide — **All**
- [ ] Everyone rehearses the whole deck — **All**

**Progress 2: Dataset, PCD, baseline ML (18 Nov)**
- [ ] Dataset statistics: frames and crops per day, time slot, weather, class — **Dinda**
- [ ] Examples of variation (time slot, light, weather) — **Dinda**
- [ ] PCD comparison figures and the chosen pipeline — **Dinda**
- [ ] Feature extraction results — **Keiza**
- [ ] At least 2 ML models + confusion matrices — **Keiza**
- [ ] Problems found and next steps — **All**

**Progress 3: DL, comparison, integration (25 Nov)**
- [ ] Final dataset and PCD pipeline — **Dinda**
- [ ] Experiment B and C results — **Tobi**
- [ ] Comparison table A vs B vs C — **Keiza**
- [ ] Error analysis with example images — **Dinda**
- [ ] Working demo — **Tobi**

**Final (2 Dec): submit everything**
- [ ] Live demo — **Tobi**
- [ ] PCD ablation — **Dinda**
- [ ] Final results, analysis and conclusions — **Keiza**
- [ ] Each member's contribution slide — **All**
- [ ] Report PDF (LaTeX, course template) — **All**, merged by **Dinda**
- [ ] GitHub link in the report, accessible to graders — **Tobi**
- [ ] Public dataset link + documentation — **Dinda**
- [ ] Upload to BRONE — **Dinda**

---

## 8. Repository structure

```
project-akhir/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # videos + videos.csv; frames/ (aligned) + frames_index.csv; align_check/ (not committed)
│   ├── annotations/          # references/, slots/ (per video), labels.csv (committed)
│   ├── processed/            # raw/ and pcd/ crops, crops.csv, dataset.pkl (not committed)
│   └── splits/               # train/val/test day lists (committed)
├── docs/
│   ├── acquisition/          # proof selfies, viewpoint photos, capture log
│   └── labeling_rules.md
├── notebooks/                # 01_eda, 02_pcd_exploration, 03_features_ml, 04_dl_raw_vs_pcd, 05_evaluation
├── src/
│   ├── pcd/                  # extract_frames, align, draw_slots, label_slots, crop_slots, slot_utils
│   ├── features/             # edge density, HOG, LBP, color statistics
│   ├── ml/                   # SVM, RF, KNN
│   └── dl/                   # MobileNetV2, optional YOLO
├── results/                  # metric tables, confusion matrices, figures
├── demo/                     # Streamlit app
└── report/                   # LaTeX report + slides
```

**Setup:**
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

---

## 9. Results
Test set = held-out days. Also kept in the **Results** sheet.

| Exp | Input | Model | Accuracy | F1 (empty) | Macro-F1 | MAE free |
|---|---|---|---|---|---|---|
| A1 | PCD + features | SVM | | | | |
| A2 | PCD + features | Random Forest | | | | |
| A3 | PCD + features | KNN | | | | |
| B | Raw crops | MobileNetV2 | | | | |
| C | PCD crops | MobileNetV2 | | | | |
| D | Full raw frames | YOLOv8n *(optional)* | | | | mAP@0.5: |

**Effect of PCD on DL:** macro-F1 C − B = ___

**PCD ablation:**

| PCD variant | Best ML macro-F1 | DL macro-F1 |
|---|---|---|
| Full pipeline | | |
| − perspective correction | | |
| − CLAHE | | |
| − blur | | |
| − morphology | | |

**Condition test (train on sunny, test on other conditions):**

| Test condition | Best ML macro-F1 | DL raw (B) | DL PCD (C) |
|---|---|---|---|
| Rain | | | |
| Evening / dusk | | | |

---

## 10. Questions to prepare for
Grading is about explaining the results, not just getting high scores.
- Why does CLAHE help (or not) with the shadows from Gedung G? Why CLAHE instead of global histogram equalization?
- Why is edge density a good feature here? When does it fail (painted lines, leaves, wet ground)?
- Does PCD help ML more than DL, or the other way round? Why?
- Why did we split by day and not randomly? What would a random split do to the scores?
- Why macro-F1 instead of accuracy, when most spaces are occupied?
- How does perspective correction make far and near slots comparable?
- Why did automatic alignment work inside one video but not between videos? What is parallax, and why does one homography fit only one plane?
- Which conditions (time slot, light, rain) cause the most errors, and why?
- What would need to change to use the system in another parking area?

---

## 11. Contribution log
Record what each person actually did, with evidence. This goes into the report.

| Date | Member | Contribution | Evidence (file / commit / photo) |
|---|---|---|---|
| | | | |

## 12. Links
- Course rules and timeline: `Penjelasan Project Akhir PCD dan Viskom.xlsx`
- Team tracker (overview, schedule, tasks, results): `parkir parkir filkom.xlsx`
- Raw videos (Drive): https://drive.google.com/drive/folders/1SBEY6itxUyp_nvoWgMVEIELD1itVRPb2?usp=sharing
- Course report template (LaTeX): https://drive.google.com/file/d/1uATd2mh6Bz-ajpLY0JGiNvPWdUlVTyvc/view?usp=sharing
- GitHub repository: *TBD*
- Public dataset link: *TBD*
- Overleaf project: *TBD*

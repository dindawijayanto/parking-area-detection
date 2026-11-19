# Label every frame as a mask: floor (0), motorcycle (1) or ignore (255), by clicking superpixels.
#
#   python src/pcd/label_masks.py            all frames of videos that have an ROI
#   python src/pcd/label_masks.py keiza      only frames whose name contains "keiza" (or a date, a video id, ...)
#
# Mouse:  left click / drag = paint with the current brush     right click = swap floor <-> motorcycle
# Keys:   1 = floor brush   2 = motorcycle brush   3 = ignore brush
#         space = save and next frame   b = save and go back   c = copy the previous frame's mask
#         r = reset (everything in the ROI = motorcycle)   [ / ] = smaller / bigger superpixels
#         g = show / hide superpixel borders   h = hold to hide the colours   q = save and quit
#
# Inside one video the frames are aligned, so a new frame starts with the mask of the frame before it:
# only paint what changed.

import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from skimage.segmentation import find_boundaries, slic

ROOT = Path(__file__).resolve().parents[2]
FRAMES_DIR = ROOT / 'data' / 'raw' / 'frames'
ROI_DIR = ROOT / 'data' / 'annotations' / 'roi'
MASK_DIR = ROOT / 'data' / 'annotations' / 'masks'

FRAME_W, FRAME_H = 1920, 1080
SEG_W, SEG_H = 960, 540      # masks are stored at half size
FLOOR, MOTOR, IGNORE = 0, 1, 255
COLORS = {FLOOR: (0, 200, 0), MOTOR: (0, 0, 230), IGNORE: (150, 150, 150)}   # BGR
BRUSH_NAMES = {FLOOR: 'floor', MOTOR: 'motorcycle', IGNORE: 'ignore'}
MAX_W = 1400
WIN = 'label masks'


def video_of(frame_name):
    return frame_name.rsplit('_t', 1)[0]   # 20261104_evening_keiza_t015.jpg -> 20261104_evening_keiza


def roi_mask(vid):
    data = json.loads((ROI_DIR / f'{vid}.json').read_text(encoding='utf-8'))
    mask = np.zeros((SEG_H, SEG_W), np.uint8)
    for poly in data['roi']:
        cv.fillPoly(mask, [np.int32(np.float32(poly) * SEG_W / FRAME_W)], 1)
    return mask


def superpixels(img, roi, size):
    # SLIC on the RGB picture, only inside the ROI (label 0 = outside), superpixels of about size x size pixels
    n_segments = max(1, int(np.sum(roi > 0) / size ** 2))
    return slic(img[:, :, ::-1], n_segments=n_segments, compactness=10, start_label=1, mask=roi > 0)


def put_text(img, text, org, scale=0.6, color=(255, 255, 255)):
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv.LINE_AA)


def main():
    text = sys.argv[1].lower() if len(sys.argv) > 1 else ''
    frames = sorted(FRAMES_DIR.glob('*.jpg'))
    no_roi = sorted(set(video_of(f.name) for f in frames if not (ROI_DIR / f'{video_of(f.name)}.json').exists()))
    if no_roi:
        print('no ROI yet (run src/pcd/draw_roi.py <video id>), skipped:', ', '.join(no_roi))
    frames = [f for f in frames if video_of(f.name) not in no_roi and text in f.name.lower()]
    if len(frames) == 0:
        print('no frames to label')
        return
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    scale = MAX_W / SEG_W
    state = {'i': 0, 'brush': FLOOR, 'grid': True, 'hide': False, 'sp_size': 12, 'painting': False}

    def mask_path(k):
        return MASK_DIR / (frames[k].stem + '.png')

    def load(k):
        img = cv.resize(cv.imread(str(frames[k])), (SEG_W, SEG_H), interpolation=cv.INTER_AREA)
        roi = roi_mask(video_of(frames[k].name))
        roi[img.max(axis=2) == 0] = 0              # black border left by the alignment warp
        same_video = k > 0 and video_of(frames[k - 1].name) == video_of(frames[k].name)

        if mask_path(k).exists():
            mask, note = cv.imread(str(mask_path(k)), cv.IMREAD_GRAYSCALE), ''
        elif same_video and mask_path(k - 1).exists():
            mask, note = cv.imread(str(mask_path(k - 1)), cv.IMREAD_GRAYSCALE), 'copied from the previous frame: paint only what changed'
        else:
            mask, note = np.full((SEG_H, SEG_W), MOTOR, np.uint8), 'NEW VIDEO: everything starts as motorcycle, paint the floor'
        mask[roi == 0] = IGNORE

        state.update(i=k, img=img, roi=roi, mask=mask, note=note)
        make_superpixels()

    def make_superpixels():
        state['sp'] = superpixels(state['img'], state['roi'], state['sp_size'])
        state['border'] = find_boundaries(state['sp'], mode='inner')

    def paint(x, y, cls=None):
        x, y = int(x / scale), int(y / scale)
        if not (0 <= x < SEG_W and 0 <= y < SEG_H):
            return
        label = state['sp'][y, x]
        if label == 0:
            return                                  # outside the ROI
        region = state['sp'] == label
        if cls is None:                             # right click: swap floor <-> motorcycle
            cls = MOTOR if state['mask'][y, x] == FLOOR else FLOOR
        state['mask'][region] = cls

    def redraw():
        img, mask = state['img'], state['mask']
        vis = img.copy()
        if not state['hide']:
            color = np.zeros_like(img)
            for cls in COLORS:
                color[mask == cls] = COLORS[cls]
            vis = cv.addWeighted(color, 0.4, img, 0.6, 0)
            if state['grid']:
                vis[state['border']] = (255, 255, 0)
        vis = cv.resize(vis, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)

        inside = mask != IGNORE
        floor_pct = 100 * np.sum(mask == FLOOR) / max(1, np.sum(inside))
        done = sum(mask_path(k).exists() for k in range(len(frames)))
        put_text(vis, f"{frames[state['i']].name}  [{state['i'] + 1}/{len(frames)}, labeled {done}]   floor {floor_pct:.1f}% of ROI", (10, 30), 0.75)
        put_text(vis, f"brush: {BRUSH_NAMES[state['brush']]}   superpixel size: {state['sp_size']} px", (10, 60), 0.7, COLORS[state['brush']])
        put_text(vis, '[1/2/3] floor/motor/ignore  [drag] paint  [right] swap  [space] next  [b] back  [c] copy prev  [r] reset  [ [ ] ] size  [g] grid  [h] hide  [q] quit', (10, 90), 0.5)
        if state['note']:
            put_text(vis, state['note'], (10, 120), 0.7, (0, 255, 255))
        cv.imshow(WIN, vis)

    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            state['painting'] = True
            paint(x, y, state['brush'])
            redraw()
        elif event == cv.EVENT_MOUSEMOVE and state['painting']:
            paint(x, y, state['brush'])
            redraw()
        elif event == cv.EVENT_LBUTTONUP:
            state['painting'] = False
        elif event == cv.EVENT_RBUTTONDOWN:
            paint(x, y)
            redraw()

    def save():
        cv.imwrite(str(mask_path(state['i'])), state['mask'])

    # start at the first frame without a mask
    start = 0
    while start < len(frames) - 1 and mask_path(start).exists():
        start += 1

    cv.namedWindow(WIN, cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback(WIN, on_mouse)
    load(start)
    redraw()
    while True:
        key = cv.waitKey(30) & 0xFF
        if state['hide'] and key != ord('h'):
            state['hide'] = False
            redraw()
        if key in (ord('1'), ord('2'), ord('3')):
            state['brush'] = [FLOOR, MOTOR, IGNORE][key - ord('1')]
        elif key in (ord(' '), ord('n'), 13):
            save()
            if state['i'] + 1 < len(frames):
                load(state['i'] + 1)
            else:
                state['note'] = 'last frame done, press q'
        elif key == ord('b') and state['i'] > 0:
            save()
            load(state['i'] - 1)
        elif key == ord('c') and state['i'] > 0 and mask_path(state['i'] - 1).exists():
            state['mask'] = cv.imread(str(mask_path(state['i'] - 1)), cv.IMREAD_GRAYSCALE)
            state['mask'][state['roi'] == 0] = IGNORE
            state['note'] = 'copied from the previous frame'
        elif key == ord('r'):
            state['mask'][state['roi'] > 0] = MOTOR
        elif key in (ord('['), ord(']')):
            state['sp_size'] = max(4, int(round(state['sp_size'] * (0.8 if key == ord('[') else 1.25))))
            make_superpixels()
        elif key == ord('g'):
            state['grid'] = not state['grid']
        elif key == ord('h'):
            state['hide'] = True
        elif key in (ord('q'), 27) or cv.getWindowProperty(WIN, cv.WND_PROP_VISIBLE) < 1:
            save()
            break
        else:
            continue
        redraw()
    cv.destroyAllWindows()
    print('masks saved in', MASK_DIR)


if __name__ == '__main__':
    main()

# Draw the ROI, the ground rectangle and one motorcycle space on a video's reference picture.
#
#   python src/pcd/draw_roi.py                          list the videos (* = ROI already drawn)
#   python src/pcd/draw_roi.py 20261104_evening_keiza   draw / fix the ROI of this video
#   python src/pcd/draw_roi.py 20261106_afternoon_keiza 20261104_evening_keiza
#                                                       start from another video's ROI and shift it
#
# Keys:
#   1 = ROI mode       click the corners of the parking area, ENTER closes the polygon (several are allowed)
#   2 = ground mode    click 4 corners of a big rectangle on the floor (e.g. the white lines of the front block)
#   3 = motor mode     click 4 corners of the space ONE parked motorcycle takes
#   right click = delete the ROI polygon under the mouse     u = undo last point
#   i / j / k / l = move everything     [ / ] = smaller / bigger everything
#   v = bird's-eye preview     s = save     q = save and quit

import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REF_DIR = ROOT / 'data' / 'annotations' / 'references'
ROI_DIR = ROOT / 'data' / 'annotations' / 'roi'

FRAME_W, FRAME_H = 1920, 1080
MAX_W, MAX_H = 1400, 820
BEV_SIZE = 700
WIN = 'draw roi'
HELP = '[1] roi  [2] ground  [3] motor  [enter] close polygon  [u] undo  [right] delete  [ijkl] move  [ [ ] ] size  [v] preview  [s] save  [q] quit'
MODE_NAMES = {1: 'ROI (click corners, ENTER closes)', 2: 'GROUND rectangle (4 corners)', 3: 'ONE MOTORCYCLE space (4 corners)'}


def load_roi(vid):
    path = ROI_DIR / f'{vid}.json'
    if not path.exists():
        return {'roi': [], 'ground': [], 'motor': []}
    data = json.loads(path.read_text(encoding='utf-8'))
    return {'roi': data['roi'], 'ground': data['ground'], 'motor': data['motor']}


def save_roi(vid, roi):
    ROI_DIR.mkdir(parents=True, exist_ok=True)
    data = {'video': vid, 'reference_image': f'{vid}.jpg', 'image_size': [FRAME_W, FRAME_H],
            'roi': [np.round(p, 1).tolist() for p in roi['roi']],
            'ground': np.round(roi['ground'], 1).tolist(),
            'motor': np.round(roi['motor'], 1).tolist()}
    (ROI_DIR / f'{vid}.json').write_text(json.dumps(data, indent=1), encoding='utf-8')


def order_quad(pts):
    # sort the 4 points clockwise around their centre, starting at the top-left one
    pts = np.float32(pts)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def bev_homography(roi):
    # ground rectangle -> unit square, then scale / shift so the whole ROI fits in the canvas
    H = cv.getPerspectiveTransform(np.float32(roi['ground']), np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
    pts = np.concatenate([np.float32(p) for p in roi['roi']]).reshape(-1, 1, 2)
    w = H[2, 0] * pts[:, 0, 0] + H[2, 1] * pts[:, 0, 1] + H[2, 2]
    g = np.float32(roi['ground'])
    side = np.sign(H[2, 0] * g[:, 0].mean() + H[2, 1] * g[:, 1].mean() + H[2, 2])
    if np.any(w * side <= 0):
        return None, None   # part of the ROI lies above the horizon of the ground plane
    bev = cv.perspectiveTransform(pts, H).reshape(-1, 2)
    x0, y0 = bev.min(axis=0)
    x1, y1 = bev.max(axis=0)
    scale = BEV_SIZE / max(x1 - x0, y1 - y0)
    T = np.float64([[scale, 0, -x0 * scale], [0, scale, -y0 * scale], [0, 0, 1]])
    size = (int(np.ceil((x1 - x0) * scale)) + 1, int(np.ceil((y1 - y0) * scale)) + 1)
    return T @ H, size


def put_text(img, text, org, scale=0.7, color=(255, 255, 255)):
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv.LINE_AA)


def draw(img, roi, current, mode):
    vis = img.copy()
    fill = img.copy()
    for poly in roi['roi']:
        cv.fillPoly(fill, [np.int32(poly)], (255, 200, 0))
    vis = cv.addWeighted(fill, 0.3, vis, 0.7, 0)
    for poly in roi['roi']:
        cv.polylines(vis, [np.int32(poly)], True, (255, 200, 0), 2)
    if len(roi['ground']) == 4:
        cv.polylines(vis, [np.int32(roi['ground'])], True, (0, 255, 255), 2)
        for k in range(4):
            put_text(vis, 'G' + str(k + 1), (int(roi['ground'][k][0]) + 6, int(roi['ground'][k][1]) - 6), 0.6, (0, 255, 255))
    if len(roi['motor']) == 4:
        cv.polylines(vis, [np.int32(roi['motor'])], True, (255, 0, 255), 3)
        put_text(vis, 'one motor', (int(roi['motor'][0][0]), int(roi['motor'][0][1]) - 8), 0.6, (255, 0, 255))
    for p in current:
        cv.circle(vis, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
    if len(current) > 1:
        cv.polylines(vis, [np.int32(current)], False, (0, 0, 255), 2)
    return vis


def bird_eye(img, roi):
    if len(roi['roi']) == 0 or len(roi['ground']) != 4:
        return None
    H, size = bev_homography(roi)
    if H is None:
        vis = np.zeros((200, 700, 3), np.uint8)
        put_text(vis, 'ROI goes above the horizon: move the far ROI corners down', (10, 100), 0.6, (0, 0, 255))
        return vis
    vis = cv.warpPerspective(img, H, size)
    for poly in roi['roi']:
        p = cv.perspectiveTransform(np.float32(poly).reshape(-1, 1, 2), H)
        cv.polylines(vis, [np.int32(p)], True, (255, 200, 0), 2)
    if len(roi['motor']) == 4:
        p = cv.perspectiveTransform(np.float32(roi['motor']).reshape(-1, 1, 2), H)
        cv.polylines(vis, [np.int32(p)], True, (255, 0, 255), 2)
    return vis


def list_videos():
    print("videos (* = ROI already drawn):")
    for p in sorted(REF_DIR.glob('*.jpg')):
        mark = '*' if (ROI_DIR / f'{p.stem}.json').exists() else ' '
        print(f'  {mark} {p.stem}')


def main():
    if len(sys.argv) < 2:
        list_videos()
        return
    vid = sys.argv[1]
    img = cv.imread(str(REF_DIR / f'{vid}.jpg'))
    if img is None:
        print(f'no reference picture for {vid}, run notebooks/00_video_to_frames.ipynb first')
        list_videos()
        return

    roi = load_roi(vid)
    if len(roi['roi']) == 0 and len(sys.argv) > 2:
        roi = load_roi(sys.argv[2])
        print(f'starting from the ROI of {sys.argv[2]}')
    roi['roi'] = [np.float32(p) for p in roi['roi']]
    roi['ground'] = np.float32(roi['ground'])
    roi['motor'] = np.float32(roi['motor'])

    scale = min(1.0, MAX_W / FRAME_W, MAX_H / FRAME_H)
    state = {'mode': 1, 'current': [], 'preview': False}

    def redraw():
        vis = draw(img, roi, state['current'], state['mode'])
        put_text(vis, f"{vid}   mode: {MODE_NAMES[state['mode']]}", (10, 35), 0.9, (0, 255, 255))
        put_text(vis, HELP, (10, 70), 0.6)
        missing = [name for name in ['roi', 'ground', 'motor'] if len(roi[name]) == 0]
        if missing:
            put_text(vis, 'still missing: ' + ', '.join(missing), (10, 105), 0.7, (0, 0, 255))
        cv.imshow(WIN, cv.resize(vis, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA))
        if state['preview']:
            bev = bird_eye(img, roi)
            if bev is not None:
                cv.imshow("bird's-eye preview", bev)

    def on_mouse(event, x, y, flags, param):
        x, y = x / scale, y / scale
        if event == cv.EVENT_LBUTTONDOWN:
            state['current'].append([x, y])
            if state['mode'] in (2, 3) and len(state['current']) == 4:
                name = 'ground' if state['mode'] == 2 else 'motor'
                roi[name] = order_quad(state['current'])
                state['current'] = []
            redraw()
        elif event == cv.EVENT_RBUTTONDOWN:
            for k in range(len(roi['roi'])):
                if cv.pointPolygonTest(roi['roi'][k], (x, y), False) >= 0:
                    roi['roi'].pop(k)
                    break
            redraw()

    def move_all(dx, dy, factor):
        center = np.float32([img.shape[1] / 2, img.shape[0] / 2])
        if len(roi['roi']) > 0:
            center = np.concatenate(roi['roi']).mean(axis=0)
        roi['roi'] = [(p - center) * factor + center + [dx, dy] for p in roi['roi']]
        if len(roi['ground']) == 4:
            roi['ground'] = (roi['ground'] - center) * factor + center + [dx, dy]
        if len(roi['motor']) == 4:
            roi['motor'] = (roi['motor'] - center) * factor + center + [dx, dy]

    cv.namedWindow(WIN, cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback(WIN, on_mouse)
    redraw()
    while True:
        key = cv.waitKey(50) & 0xFF
        if key in (ord('1'), ord('2'), ord('3')):
            state['mode'] = key - ord('0')
            state['current'] = []
        elif key == 13 and state['mode'] == 1 and len(state['current']) >= 3:   # ENTER
            roi['roi'].append(np.float32(state['current']))
            state['current'] = []
        elif key == ord('u'):
            if len(state['current']) > 0:
                state['current'].pop()
            elif state['mode'] == 1 and len(roi['roi']) > 0:
                roi['roi'].pop()
        elif key == ord('j'):
            move_all(-5, 0, 1.0)
        elif key == ord('l'):
            move_all(5, 0, 1.0)
        elif key == ord('i'):
            move_all(0, -5, 1.0)
        elif key == ord('k'):
            move_all(0, 5, 1.0)
        elif key == ord('['):
            move_all(0, 0, 0.98)
        elif key == ord(']'):
            move_all(0, 0, 1.02)
        elif key == ord('v'):
            state['preview'] = not state['preview']
            if not state['preview']:
                cv.destroyWindow("bird's-eye preview")
        elif key == ord('s'):
            save_roi(vid, roi)
            print('saved', ROI_DIR / f'{vid}.json')
        elif key in (ord('q'), 27) or cv.getWindowProperty(WIN, cv.WND_PROP_VISIBLE) < 1:
            save_roi(vid, roi)
            print('saved', ROI_DIR / f'{vid}.json')
            break
        else:
            continue
        redraw()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

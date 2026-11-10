import cv2
import numpy as np

from slot_utils import FRAME_H, FRAME_W

WORK_W = 960  # match on a half-size picture: faster, and enough detail


class Aligner:
    def __init__(self, min_inliers=40):
        self.sift = cv2.SIFT_create(nfeatures=5000)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # helps match sunny vs dark
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.min_inliers = min_inliers
        self.s = WORK_W / FRAME_W
        self.S = np.diag([self.s, self.s, 1.0])

    def features(self, img):
        g = cv2.cvtColor(cv2.resize(img, (WORK_W, int(FRAME_H * self.s))), cv2.COLOR_BGR2GRAY)
        return self.sift.detectAndCompute(self.clahe.apply(g), None)

    def homography(self, fa, fb):
        """Homography mapping picture A onto picture B (full-size coordinates), or (None, inliers)."""
        (ka, da), (kb, db) = fa, fb
        if da is None or db is None or len(ka) < 10 or len(kb) < 10:
            return None, 0
        good = [m for m, n in (p for p in self.matcher.knnMatch(da, db, k=2) if len(p) == 2)
                if m.distance < 0.75 * n.distance]
        if len(good) < self.min_inliers:
            return None, len(good)
        src = np.float32([ka[m.queryIdx].pt for m in good])
        dst = np.float32([kb[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        inl = int(mask.sum()) if mask is not None else 0
        if H is None or inl < self.min_inliers:
            return None, inl
        H = np.linalg.inv(self.S) @ H @ self.S
        return (H, inl) if self.sane(H) else (None, inl)

    @staticmethod
    def sane(H):
        c = np.float32([[0, 0], [FRAME_W, 0], [FRAME_W, FRAME_H], [0, FRAME_H]]).reshape(-1, 1, 2)
        w = cv2.perspectiveTransform(c, H).reshape(-1, 2)
        area = cv2.contourArea(w) / (FRAME_W * FRAME_H)
        return 0.5 < area < 2.0 and cv2.isContourConvex(w.astype(np.float32))

    @staticmethod
    def warp(img, H):
        return cv2.warpPerspective(img, H, (FRAME_W, FRAME_H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

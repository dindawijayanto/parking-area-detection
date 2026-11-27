"""Run the parking model demo from the repository root with python -m app.parking_demo."""

import argparse
import base64
import csv
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock
from urllib.error import URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen

import cv2
import joblib
import numpy as np

from train_rf_vs_xgb import apply_morphology, predict_prob_map
from train_rf_vs_xgb_v2 import (
    BEV_MAP, DATA_DIR, FEATURE_NAMES_V2, OUT_DIR, compute_crop_to_bev,
    count_free_spaces, extract_pcd_features_v2, prepare_bev_geometries, smooth_prob_roi,
)


STATIC_DIR = Path(__file__).with_name("parking_demo_ui")
ZONES_PATH = Path(__file__).with_name("parking_demo_zones.json")
MODELS = {
    "xgb": ("XGBoost", "XGB_v2", "xgb_v2_model.joblib"),
    "rf": ("Random Forest", "RF_v2", "rf_v2_model.joblib"),
}
TIME_OF_DAY = {"morning": "Pagi", "afternoon": "Siang", "evening": "Sore"}
INFERENCE_LOCK = Lock()
YOLO_LOCK = Lock()


class DemoError(Exception):
    pass


@lru_cache(maxsize=1)
def marked_zones():
    """Manual outlines based on white markings, in reference photo coordinates."""
    zones = json.loads(ZONES_PATH.read_text(encoding="utf-8"))
    return {
        reference: [np.float32(poly).reshape(-1, 1, 2) for poly in polygons]
        for reference, polygons in zones.items()
    }


@lru_cache(maxsize=1)
def frames():
    with (DATA_DIR / "manifest.csv").open(encoding="utf-8-sig", newline="") as source:
        return {
            row["frame_id"]: row for row in csv.DictReader(source)
            if row["split"] == "test" and BEV_MAP.get(row["recording_id"]) in marked_zones()
        }


@lru_cache(maxsize=1)
def calibration():
    geometries, sift, matcher = prepare_bev_geometries()
    for geometry in geometries.values():
        polygons = marked_zones().get(geometry["vid"])
        if polygons is None:
            continue
        geometry["parking_zones"] = polygons

        # Fit the BEV output to the marked zones while keeping the calibrated scale.
        projection = geometry["H_ref_to_bev"]
        top_points = cv2.perspectiveTransform(np.concatenate(polygons), projection).reshape(-1, 2)
        padding = np.asarray(geometry["kernel"].shape[::-1]) // 2 + 1
        x0, y0 = np.floor(top_points.min(axis=0)).astype(int) - padding
        x1, y1 = np.ceil(top_points.max(axis=0)).astype(int) + padding
        offset = np.float64([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
        geometry["H_ref_to_bev"] = offset @ projection
        geometry["size"] = (int(x1 - x0 + 1), int(y1 - y0 + 1))
    return geometries, sift, matcher


def frame_image(frame_id):
    return DATA_DIR / frames()[frame_id]["image_path"]


@lru_cache(maxsize=2)
def trained_model(model_id):
    _, benchmark_key, filename = MODELS[model_id]
    config = json.loads((OUT_DIR / "v2_benchmark_summary.json").read_text(encoding="utf-8"))
    bundle = joblib.load(OUT_DIR / filename)
    if bundle["features"] != FEATURE_NAMES_V2:
        raise DemoError("Urutan fitur model tidak cocok dengan pipeline demo.")
    model = bundle["model"]
    model.set_params(n_jobs=4)
    return model, config[benchmark_key]["config"]


def tinted(image, mask, color, alpha):
    image[mask] = np.rint(image[mask] * (1 - alpha) + np.asarray(color) * alpha)


def jpeg_url(image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise DemoError("Hasil gambar tidak bisa dibuat.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


@lru_cache(maxsize=12)
def analyze(frame_id, model_id):
    image = cv2.imread(str(frame_image(frame_id)))
    if image is None:
        raise DemoError("Foto uji tidak bisa dibuka.")
    geometries, sift, matcher = calibration()
    g = geometries[frames()[frame_id]["recording_id"]]
    try:
        crop_to_bev, _ = compute_crop_to_bev(image, g, sift, matcher)
        ref_to_crop = np.linalg.solve(crop_to_bev, g["H_ref_to_bev"])
    except (AssertionError, cv2.error, np.linalg.LinAlgError) as error:
        raise DemoError("Frame tidak bisa disejajarkan dengan referensi parkir.") from error

    roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    outlines = []
    for polygon in g["parking_zones"]:
        projected = cv2.perspectiveTransform(polygon, ref_to_crop).reshape(-1, 2)
        outline = np.rint(projected).astype(np.int32)
        outlines.append(outline)
        cv2.fillPoly(roi_mask, [outline], 1)
    roi = (roi_mask == 1) & image.any(axis=2)
    if not roi.any():
        raise DemoError("Area parkir tidak terlihat di frame ini.")

    model, config = trained_model(model_id)
    features = extract_pcd_features_v2(image)
    probability = predict_prob_map(model, features, roi)
    probability = smooth_prob_roi(probability, roi, config["blur_sigma"])
    occupied = apply_morphology(probability >= config["thr"], roi, config["morph_k"]).astype(bool)
    labels = np.full(roi.shape, 255, dtype=np.uint8)
    labels[roi] = 0
    labels[occupied] = 1

    free_spaces, _, usable_bev = count_free_spaces(labels, crop_to_bev, g)
    height, width = roi.shape
    usable = cv2.warpPerspective(
        usable_bev, crop_to_bev, (width, height), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST,
    ).astype(bool) & roi & ~occupied
    overlay = image.copy()
    tinted(overlay, ~roi, (17, 22, 17), 0.76)
    tinted(overlay, occupied, (98, 104, 255), 0.51)
    tinted(overlay, usable, (150, 240, 82), 0.56)
    cv2.polylines(overlay, outlines, True, (255, 219, 63), 2, cv2.LINE_AA)
    return {
        "image": jpeg_url(overlay),
        "freeSpaces": free_spaces,
        "occupiedPercent": round(100 * occupied.sum() / roi.sum(), 1),
    }


@lru_cache(maxsize=12)
def detect_motors(frame_id, masking_port):
    frame = frames()[frame_id]
    data = frame_image(frame_id).read_bytes()
    width, height = int(frame["width"]), int(frame["height"])
    query = urlencode({"width": width, "height": height, "label": "motor"})
    request = Request(
        f"http://127.0.0.1:{masking_port}/api/auto-mask?{query}",
        data=data, headers={"Content-Type": "image/jpeg"},
    )
    try:
        with urlopen(request, timeout=120) as response:
            packed = response.read()
    except (URLError, TimeoutError) as error:
        raise DemoError("YOLO belum bisa diakses. Pastikan Masking Parkir berjalan, lalu coba lagi.") from error
    if len(packed) != (width * height + 7) // 8:
        raise DemoError("Ukuran mask YOLO tidak cocok dengan foto.")
    mask = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=width * height)
    mask = mask.reshape((height, width)).astype(bool)
    overlay = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    tinted(overlay, mask, (80, 200, 255), 0.57)
    return {"image": jpeg_url(overlay), "detected": bool(mask.any())}


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def reply(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, data, status=200):
        self.reply(json.dumps(data).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self):
        route = urlsplit(self.path)
        if not route.path.startswith("/api/"):
            super().do_GET()
            return
        try:
            self.api(route.path, parse_qs(route.query))
        except DemoError as error:
            self.json({"error": str(error)}, 503)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:
            self.log_error("%r", error)
            self.json({"error": "Demo gagal memproses permintaan ini. Coba lagi."}, 500)

    def api(self, path, query):
        if path == "/api/frames":
            self.json({
                "frames": [
                    {
                        "id": item["frame_id"], "recording": item["recording_id"],
                        "date": item["date"], "timestamp": item["timestamp"],
                        "timeOfDay": TIME_OF_DAY[item["time_of_day"]],
                        "width": int(item["width"]), "height": int(item["height"]),
                    }
                    for item in frames().values()
                ],
                "models": [{"id": key, "name": value[0]} for key, value in MODELS.items()],
            })
            return
        if path not in ("/api/frame", "/api/analysis", "/api/motors"):
            self.json({"error": "Halaman tidak ditemukan."}, 404)
            return
        frame_id = query.get("frame", [""])[0]
        if frame_id not in frames():
            self.json({"error": "Frame uji tidak ditemukan."}, 404)
            return
        if path == "/api/frame":
            self.reply(frame_image(frame_id).read_bytes(), "image/jpeg")
        elif path == "/api/motors":
            with YOLO_LOCK:
                result = detect_motors(frame_id, self.server.masking_port)
            self.json(result)
        else:
            model_id = query.get("model", [""])[0]
            if model_id not in MODELS:
                self.json({"error": "Model parkir tidak dikenal."}, 400)
                return
            with INFERENCE_LOCK:
                result = analyze(frame_id, model_id)
            self.json(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--masking-port", type=int, default=8766)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler) as server:
        server.masking_port = args.masking_port
        print(f"Demo Parkir  http://127.0.0.1:{args.port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()

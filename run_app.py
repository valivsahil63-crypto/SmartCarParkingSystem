"""
Flask web dashboard — Smart Parking System
Run: python run_app.py
Open browser: http://localhost:5000
"""

import cv2
import json
import time
import threading
import numpy as np
from flask import Flask, Response, render_template, jsonify, request
from flask_cors import CORS

import config
from modules.parking_detector import ParkingDetector
from modules.vehicle_counter   import VehicleCounter
from modules.plate_recognizer  import PlateRecognizer
from modules.utils             import draw_stats_panel

app  = Flask(__name__)
CORS(app)

# ── Shared state ───────────────────────────────────────────────────────────────
_lock          = threading.Lock()
output_frame   = None
stats          = {
    "total_slots": 0, "free": 0, "occupied": 0,
    "entered": 0, "exited": 0, "fps": 0
}
plate_history  = []
running        = True

detector   = None
counter    = None
recognizer = None
cap        = None


def init_pipeline():
    global detector, counter, recognizer, cap
    detector   = ParkingDetector()
    counter    = VehicleCounter()
    recognizer = PlateRecognizer()
    cap        = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[WARN] Cannot open source '{config.VIDEO_SOURCE}'. Using blank placeholder.")


def processing_loop():
    global output_frame, stats, plate_history, cap

    frame_interval = 1.0 / config.FRAME_RATE
    prev_time = time.time()

    while running:
        if cap is None or not cap.isOpened():
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video
            continue

        # Resize for performance (max 1280px wide)
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (1280, int(h * scale)))

        # ── ML Modules ─────────────────────────────────────────────────────
        frame, slot_summary  = detector.detect(frame)
        frame, count_summary = counter.process(frame)
        frame, new_plates    = recognizer.process(frame)

        # ── FPS ────────────────────────────────────────────────────────────
        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now

        # ── HUD overlay ────────────────────────────────────────────────────
        hud = {
            "Free Slots"  : slot_summary.get("Free", 0),
            "Occupied"    : slot_summary.get("Occupied", 0),
            "Entered"     : count_summary.get("Entered", 0),
            "Exited"      : count_summary.get("Exited", 0),
            "FPS"         : f"{fps:.1f}",
        }
        draw_stats_panel(frame, hud)

        # ── Update shared state ─────────────────────────────────────────────
        with _lock:
            stats["total_slots"] = slot_summary.get("Total Slots", 0)
            stats["free"]        = slot_summary.get("Free", 0)
            stats["occupied"]    = slot_summary.get("Occupied", 0)
            stats["entered"]     = count_summary.get("Entered", 0)
            stats["exited"]      = count_summary.get("Exited", 0)
            stats["fps"]         = round(fps, 1)

            for p in new_plates:
                plate_history.insert(0, {
                    "plate"     : p["plate"],
                    "confidence": p["confidence"],
                    "time"      : time.strftime("%H:%M:%S"),
                })
            plate_history[:] = plate_history[:50]

            output_frame = frame.copy()

        elapsed = time.time() - now
        time.sleep(max(0, frame_interval - elapsed))


def gen_stream():
    """MJPEG byte stream generator."""
    while True:
        with _lock:
            if output_frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Initializing...", (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
                frame = blank
            else:
                frame = output_frame.copy()

        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
        time.sleep(1.0 / config.FRAME_RATE)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/stats")
def api_stats():
    with _lock:
        return jsonify(stats.copy())

@app.route("/api/plates")
def api_plates():
    with _lock:
        return jsonify(plate_history[:20])

@app.route("/api/reload_slots", methods=["POST"])
def api_reload_slots():
    detector.reload_slots()
    return jsonify({"status": "ok", "slots": detector.slot_count})

@app.route("/api/set_source", methods=["POST"])
def api_set_source():
    global cap
    data   = request.json or {}
    source = data.get("source", 0)
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass
    with _lock:
        if cap:
            cap.release()
        cap = cv2.VideoCapture(source)
    return jsonify({"status": "ok", "source": str(source)})


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    banner = "=" * 56
    print(banner)
    print("  🚗  Smart Parking System — ML Dashboard")
    print(f"  🌐  http://localhost:{config.FLASK_PORT}")
    print(banner)
    init_pipeline()
    t = threading.Thread(target=processing_loop, daemon=True)
    t.start()
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT,
            debug=False, threaded=True)

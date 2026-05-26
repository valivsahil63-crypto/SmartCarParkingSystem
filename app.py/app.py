"""
app.py — Flask web dashboard with live MJPEG video stream + SSE stats.
Run:  python app.py
Open: http://localhost:5000
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

app      = Flask(__name__)
CORS(app)

# ── Shared state ──────────────────────────────────────────────────────────────
lock     = threading.Lock()
output_frame   = None
stats          = {
    "total_slots": 0, "free": 0, "occupied": 0,
    "entered": 0, "exited": 0, "fps": 0
}
plate_history  = []
vehicle_events = []
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
        print(f"[WARN] Cannot open source {config.VIDEO_SOURCE}. Using blank frame.")


def processing_loop():
    global output_frame, stats, plate_history, vehicle_events, cap

    frame_interval = 1.0 / config.FRAME_RATE
    prev_time = time.time()

    while running:
        if cap is None or not cap.isOpened():
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Resize for performance
        h, w = frame.shape[:2]
        if w > 1280:
            scale  = 1280 / w
            frame  = cv2.resize(frame, (1280, int(h * scale)))

        # ── Run ML modules ─────────────────────────────────────────────────
        frame, slot_summary  = detector.detect(frame)
        frame, count_summary = counter.process(frame)
        frame, new_plates    = recognizer.process(frame)

        # ── FPS ────────────────────────────────────────────────────────────
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # ── HUD Panel ──────────────────────────────────────────────────────
        hud = {
            "Slots Free"   : slot_summary.get("Free", 0),
            "Slots Occupied": slot_summary.get("Occupied", 0),
            "Entered"      : count_summary.get("Entered", 0),
            "Exited"       : count_summary.get("Exited", 0),
            "FPS"          : f"{fps:.1f}",
        }
        draw_stats_panel(frame, hud)

        # ── Update shared state ────────────────────────────────────────────
        with lock:
            stats["total_slots"] = slot_summary.get("Total Slots", 0)
            stats["free"]        = slot_summary.get("Free", 0)
            stats["occupied"]    = slot_summary.get("Occupied", 0)
            stats["entered"]     = count_summary.get("Entered", 0)
            stats["exited"]      = count_summary.get("Exited", 0)
            stats["fps"]         = round(fps, 1)

            for p in new_plates:
                plate_history.insert(0, {
                    "plate": p["plate"], "confidence": p["confidence"],
                    "time": time.strftime("%H:%M:%S")
                })
            plate_history = plate_history[:50]   # Keep latest 50

            output_frame = frame.copy()

        # Throttle
        elapsed = time.time() - now
        sleep   = max(0, frame_interval - elapsed)
        time.sleep(sleep)


def generate_stream():
    """MJPEG generator."""
    while True:
        with lock:
            if output_frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for video...", (120, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                frame = blank
            else:
                frame = output_frame.copy()

        ret, buf = cv2.imencode(".jpg", frame,
                                [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")
        time.sleep(1.0 / config.FRAME_RATE)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def api_stats():
    with lock:
        return jsonify(stats.copy())


@app.route("/api/plates")
def api_plates():
    with lock:
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
    with lock:
        if cap:
            cap.release()
        cap = cv2.VideoCapture(source)
    return jsonify({"status": "ok", "source": str(source)})


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  🚗  Smart Parking System  |  http://localhost:5000")
    print("=" * 55)
    init_pipeline()
    t = threading.Thread(target=processing_loop, daemon=True)
    t.start()
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT,
            debug=False, threaded=True)

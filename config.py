"""
config.py — Central configuration for Smart Parking System
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
LOGS_DIR        = os.path.join(DATA_DIR, "logs")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
SLOTS_FILE      = os.path.join(DATA_DIR, "parking_slots.json")
VEHICLE_LOG     = os.path.join(LOGS_DIR, "vehicle_log.csv")
PLATE_LOG       = os.path.join(LOGS_DIR, "plate_log.csv")

# ─── YOLO ─────────────────────────────────────────────────────────────────────
YOLO_MODEL      = "yolov8n.pt"           # nano — fast; swap for yolov8s.pt for accuracy
VEHICLE_CLASSES = [2, 3, 5, 7]           # COCO: car, motorcycle, bus, truck
PLATE_CONF      = 0.40                   # confidence threshold for plate detection
VEHICLE_CONF    = 0.40                   # confidence threshold for vehicle detection

# ─── Video Source ─────────────────────────────────────────────────────────────
# Set VIDEO_SOURCE = 0  to use webcam
# Set VIDEO_SOURCE = "path/to/video.mp4" for a file
VIDEO_SOURCE    = 0

# ─── Parking Slot Detection ───────────────────────────────────────────────────
SLOT_OCCUPIED_THRESHOLD = 0.5           # IOU ratio above which a slot is occupied
EMPTY_COLOR             = (0, 255, 0)   # Green overlay for free slot
OCCUPIED_COLOR          = (0, 0, 255)   # Red overlay for occupied slot
SLOT_ALPHA              = 0.4           # Transparency of slot overlay

# ─── Vehicle Counter ──────────────────────────────────────────────────────────
# Counting line position as fraction of frame height (0.0–1.0)
COUNT_LINE_RATIO        = 0.5
COUNT_LINE_COLOR        = (255, 255, 0)
COUNT_LINE_THICKNESS    = 2
MAX_DISAPPEARED         = 40            # Frames before a tracked object is dropped
MAX_DISTANCE            = 80            # Max centroid distance for ID assignment

# ─── OCR ──────────────────────────────────────────────────────────────────────
OCR_LANGUAGES           = ["en"]
OCR_GPU                 = False          # Set True if CUDA GPU is available
OCR_MIN_CONFIDENCE      = 0.5

# ─── Flask ────────────────────────────────────────────────────────────────────
FLASK_HOST              = "0.0.0.0"
FLASK_PORT              = 5000
FRAME_RATE              = 20            # Target FPS for web stream
JPEG_QUALITY            = 80

# ─── Display ──────────────────────────────────────────────────────────────────
FONT                    = 0             # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE              = 0.6
FONT_THICKNESS          = 2
TEXT_COLOR              = (255, 255, 255)

# ─── Ensure dirs exist ────────────────────────────────────────────────────────
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

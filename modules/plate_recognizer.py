"""
modules/plate_recognizer.py
Detects license plates using YOLO (license plate model fallback: YOLO general)
and reads text with EasyOCR.
"""

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import config
from modules.utils import log_to_csv, timestamp

# Fallback: use the general YOLO model and look for cars,
# then crop the bottom portion of each car as the "plate region"
PLATE_REGION_RATIO = (0.65, 0.85)   # (top%, bot%) of car bbox height


class PlateRecognizer:
    def __init__(self):
        self.model    = YOLO(config.YOLO_MODEL)
        self.reader   = easyocr.Reader(config.OCR_LANGUAGES, gpu=config.OCR_GPU)
        self.history  = []           # list of {timestamp, plate, confidence}
        self._seen    = set()        # avoid duplicate logs in same session

    def _crop_plate_region(self, frame, box):
        """Crop the lower portion of a vehicle bounding box (probable plate area)."""
        x1, y1, x2, y2 = box
        h = y2 - y1
        pt = int(y1 + h * PLATE_REGION_RATIO[0])
        pb = int(y1 + h * PLATE_REGION_RATIO[1])
        pt = max(0, pt)
        pb = min(frame.shape[0], pb)
        return frame[pt:pb, max(0, x1):min(frame.shape[1], x2)]

    def _clean_plate_text(self, raw: str) -> str:
        import re
        cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper())
        return cleaned if len(cleaned) >= 4 else ""

    def process(self, frame: np.ndarray) -> tuple:
        """
        Detect vehicles, crop plate regions, run OCR.
        Returns annotated frame + list of detected plates this frame.
        """
        results   = self.model(frame, verbose=False, conf=config.VEHICLE_CONF,
                               classes=config.VEHICLE_CLASSES)[0]
        new_plates = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = self._crop_plate_region(frame, (x1, y1, x2, y2))

            if crop.size == 0:
                continue

            # Pre-process for better OCR
            gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray  = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray  = cv2.GaussianBlur(gray, (3, 3), 0)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            ocr_results = self.reader.readtext(bw, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

            for (_, text, conf) in ocr_results:
                if conf < config.OCR_MIN_CONFIDENCE:
                    continue
                plate = self._clean_plate_text(text)
                if not plate:
                    continue

                new_plates.append({"plate": plate, "confidence": round(conf, 2)})

                # Draw on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Plate: {plate} ({conf:.0%})"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # Log unique plates
                if plate not in self._seen:
                    self._seen.add(plate)
                    entry = {
                        "timestamp"  : timestamp(),
                        "plate"      : plate,
                        "confidence" : round(conf, 2)
                    }
                    self.history.append(entry)
                    log_to_csv(config.PLATE_LOG, entry)

        return frame, new_plates

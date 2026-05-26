"""
modules/parking_detector.py
Detects whether each defined parking slot is empty or occupied.
Uses YOLO detections and polygon overlap (IOU).
"""

import cv2
import numpy as np
from ultralytics import YOLO
import config
from modules.utils import load_slots, iou_rect_poly


class ParkingDetector:
    def __init__(self):
        self.model  = YOLO(config.YOLO_MODEL)
        self.slots  = load_slots(config.SLOTS_FILE)
        self.states = {}       # slot_id -> "empty" | "occupied"

    def reload_slots(self):
        self.slots = load_slots(config.SLOTS_FILE)

    def detect(self, frame: np.ndarray) -> tuple:
        """
        Run YOLO on the frame, check each slot for overlap.
        Returns:
            annotated_frame : frame with slot overlays drawn
            summary         : dict with total/free/occupied counts
        """
        h, w = frame.shape[:2]
        results = self.model(frame, verbose=False, conf=config.VEHICLE_CONF,
                             classes=config.VEHICLE_CLASSES)[0]

        # Gather bounding boxes
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))

        overlay = frame.copy()
        occupied = 0

        for idx, slot in enumerate(self.slots):
            pts   = np.array(slot, dtype=np.int32)
            is_oc = False

            for box in boxes:
                overlap = iou_rect_poly(box, slot)
                if overlap >= config.SLOT_OCCUPIED_THRESHOLD:
                    is_oc = True
                    break

            color = config.OCCUPIED_COLOR if is_oc else config.EMPTY_COLOR
            self.states[idx] = "occupied" if is_oc else "empty"
            if is_oc:
                occupied += 1

            # Draw filled polygon
            cv2.fillPoly(overlay, [pts], color)
            # Draw slot number
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(overlay, str(idx + 1), (cx - 8, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Blend overlay
        cv2.addWeighted(overlay, config.SLOT_ALPHA, frame, 1 - config.SLOT_ALPHA, 0, frame)

        # Draw polygon outlines
        for slot in self.slots:
            pts = np.array(slot, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (255, 255, 255), 1)

        free  = len(self.slots) - occupied
        total = len(self.slots)

        summary = {
            "Total Slots" : total,
            "Free"        : free,
            "Occupied"    : occupied,
        }
        return frame, summary

    @property
    def slot_count(self):
        return len(self.slots)

    @property
    def free_count(self):
        return sum(1 for s in self.states.values() if s == "empty")

    @property
    def occupied_count(self):
        return sum(1 for s in self.states.values() if s == "occupied")

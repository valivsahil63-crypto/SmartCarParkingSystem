"""
modules/vehicle_counter.py
Counts vehicles crossing a virtual line using YOLO + centroid tracking.
Supports entry (top→bottom) and exit (bottom→top) directions.
"""

import cv2
import numpy as np
from collections import OrderedDict
from ultralytics import YOLO
from scipy.spatial import distance as dist
import config
from modules.utils import log_to_csv, timestamp


class CentroidTracker:
    """Simple centroid-based multi-object tracker."""

    def __init__(self, max_disappeared=40, max_distance=80):
        self.next_id       = 0
        self.objects       = OrderedDict()   # id -> centroid
        self.disappeared   = OrderedDict()   # id -> frames missing
        self.max_dis       = max_disappeared
        self.max_dist      = max_distance

    def register(self, centroid):
        self.objects[self.next_id]    = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, rects):
        """
        rects: list of (x1, y1, x2, y2)
        Returns dict of id -> centroid
        """
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_dis:
                    self.deregister(obj_id)
            return self.objects

        centroids = np.zeros((len(rects), 2), dtype=int)
        for i, (x1, y1, x2, y2) in enumerate(rects):
            centroids[i] = ((x1 + x2) // 2, (y1 + y2) // 2)

        if len(self.objects) == 0:
            for c in centroids:
                self.register(c)
        else:
            obj_ids   = list(self.objects.keys())
            obj_cents = list(self.objects.values())
            D         = dist.cdist(np.array(obj_cents), centroids)
            rows      = D.min(axis=1).argsort()
            cols      = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_dist:
                    continue
                obj_id = obj_ids[row]
                self.objects[obj_id]    = centroids[col]
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                obj_id = obj_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_dis:
                    self.deregister(obj_id)

            for col in unused_cols:
                self.register(centroids[col])

        return self.objects


class VehicleCounter:
    def __init__(self):
        self.model        = YOLO(config.YOLO_MODEL)
        self.tracker      = CentroidTracker(config.MAX_DISAPPEARED, config.MAX_DISTANCE)
        self.entered      = 0
        self.exited       = 0
        self._prev_cents  = {}      # id -> prev centroid y
        self._counted     = set()   # ids already counted

    def process(self, frame: np.ndarray) -> tuple:
        """
        Run YOLO, update tracker, count crossings.
        Returns annotated frame + count dict.
        """
        h, w = frame.shape[:2]
        line_y = int(h * config.COUNT_LINE_RATIO)

        results = self.model(frame, verbose=False, conf=config.VEHICLE_CONF,
                             classes=config.VEHICLE_CLASSES)[0]

        rects = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            rects.append((x1, y1, x2, y2))
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            label = f"{self.model.names[cls]} {conf:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1, cv2.LINE_AA)

        objects = self.tracker.update(rects)

        for obj_id, centroid in objects.items():
            cx, cy = centroid
            # Draw centroid dot
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"ID {obj_id}", (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # Count crossing
            if obj_id in self._prev_cents and obj_id not in self._counted:
                prev_y = self._prev_cents[obj_id]
                if prev_y < line_y <= cy:          # top → bottom = ENTRY
                    self.entered += 1
                    self._counted.add(obj_id)
                    log_to_csv(config.VEHICLE_LOG, {
                        "timestamp": timestamp(), "event": "ENTRY", "vehicle_id": obj_id
                    })
                elif prev_y > line_y >= cy:        # bottom → top = EXIT
                    self.exited += 1
                    self._counted.add(obj_id)
                    log_to_csv(config.VEHICLE_LOG, {
                        "timestamp": timestamp(), "event": "EXIT", "vehicle_id": obj_id
                    })

            self._prev_cents[obj_id] = cy

        # Draw counting line
        cv2.line(frame, (0, line_y), (w, line_y),
                 config.COUNT_LINE_COLOR, config.COUNT_LINE_THICKNESS)
        cv2.putText(frame, "COUNTING LINE", (10, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COUNT_LINE_COLOR, 1, cv2.LINE_AA)

        counts = {"Entered": self.entered, "Exited": self.exited}
        return frame, counts

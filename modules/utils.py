"""
modules/utils.py — Shared helper functions
"""

import cv2
import csv
import json
import numpy as np
import os
from datetime import datetime
import config


def load_slots(path: str) -> list:
    """Load parking slot polygons from JSON file."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_slots(slots: list, path: str):
    """Save parking slot polygons to JSON file."""
    with open(path, "w") as f:
        json.dump(slots, f, indent=2)


def draw_text_with_bg(frame, text, pos, font_scale=0.6, thickness=2,
                       fg=(255, 255, 255), bg=(0, 0, 0)):
    """Draw text with a solid background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x - 4, y - h - 6), (x + w + 4, y + 4), bg, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, fg, thickness, cv2.LINE_AA)


def draw_stats_panel(frame, stats: dict):
    """Draw a semi-transparent HUD panel in top-left corner."""
    panel_h = 30 + len(stats) * 28
    panel_w = 240
    overlay  = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (100, 100, 100), 1)

    y = 38
    for key, val in stats.items():
        draw_text_with_bg(frame, f"{key}: {val}", (18, y),
                          fg=(200, 255, 200), bg=(20, 20, 20))
        y += 28


def iou_rect_poly(box, poly_pts):
    """
    Compute the fraction of a YOLO bounding box that overlaps
    with a polygon (parking slot).
    box      : (x1, y1, x2, y2) ints
    poly_pts : list of [x, y] points
    Returns  : float in [0, 1]
    """
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return 0.0

    pts = np.array(poly_pts, dtype=np.float32)
    # Create masks on a small canvas aligned to bounding box
    h = max(frame_h := (y2 - y1 + 1), 1)
    w = max(frame_w := (x2 - x1 + 1), 1)
    box_mask  = np.ones((h, w), dtype=np.uint8)
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    shifted   = pts - np.array([x1, y1])
    cv2.fillPoly(poly_mask, [shifted.astype(np.int32)], 1)
    intersection = np.sum(box_mask * poly_mask)
    box_area     = bw * bh
    return float(intersection) / float(box_area) if box_area > 0 else 0.0


def log_to_csv(filepath: str, row: dict):
    """Append a dict row to a CSV file, creating headers if needed."""
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clamp(value, lo, hi):
    return max(lo, min(hi, value))

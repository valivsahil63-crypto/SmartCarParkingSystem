"""
setup_slots.py
Interactive tool to define parking slot polygons on a video frame.
Run ONCE before starting the main app.

Controls:
  Left-click  : Add point to current slot polygon
  Right-click : Finish current slot polygon
  'z'         : Undo last point
  'r'         : Reset / clear all slots
  's'         : Save slots and exit
  'q'         : Quit without saving
"""

import cv2
import json
import sys
import numpy as np
import config

WINDOW = "Parking Slot Setup  |  L-Click: add point  R-Click: finish slot  S: save  Q: quit"

slots      = []     # list of completed slot polygons
current    = []     # points of the in-progress slot

def mouse_callback(event, x, y, flags, param):
    global current, slots
    if event == cv2.EVENT_LBUTTONDOWN:
        current.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current) >= 3:
            slots.append(current.copy())
            print(f"  ✔ Slot {len(slots)} saved ({len(current)} pts)")
        current = []

def draw_all(frame):
    overlay = frame.copy()
    colors  = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (200, 0, 255)]

    for idx, slot in enumerate(slots):
        pts   = np.array(slot, dtype=np.int32)
        color = colors[idx % len(colors)]
        cv2.fillPoly(overlay, [pts], color)
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(overlay, str(idx + 1), (cx - 8, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.polylines(frame, [pts], True, (255, 255, 255), 1)

    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Draw in-progress slot
    for pt in current:
        cv2.circle(frame, tuple(pt), 5, (0, 255, 255), -1)
    if len(current) > 1:
        for i in range(len(current) - 1):
            cv2.line(frame, tuple(current[i]), tuple(current[i + 1]), (0, 255, 255), 1)

    info = f"Slots: {len(slots)}  |  Current pts: {len(current)}"
    cv2.putText(frame, info, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def main():
    source = config.VIDEO_SOURCE
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Could not read first frame.")
        sys.exit(1)

    base_frame = frame.copy()
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, mouse_callback)

    print("\n=== PARKING SLOT SETUP ===")
    print("  Left-click  : Place polygon point")
    print("  Right-click : Close & save current slot")
    print("  Z           : Undo last point")
    print("  R           : Reset all slots")
    print("  S           : Save and exit")
    print("  Q           : Quit without saving\n")

    while True:
        display = base_frame.copy()
        display = draw_all(display)
        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('s'):
            if slots:
                config.save_slots(slots, config.SLOTS_FILE) if hasattr(config, 'save_slots') else None
                with open(config.SLOTS_FILE, "w") as f:
                    json.dump(slots, f, indent=2)
                print(f"\n✅ Saved {len(slots)} slots → {config.SLOTS_FILE}")
            else:
                print("[WARN] No slots to save.")
            break
        elif key == ord('q'):
            print("[INFO] Quit without saving.")
            break
        elif key == ord('z'):
            if current:
                current.pop()
            elif slots:
                slots.pop()
        elif key == ord('r'):
            slots.clear()
            current.clear()
            print("[INFO] All slots cleared.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

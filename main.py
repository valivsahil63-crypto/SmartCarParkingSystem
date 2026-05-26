"""
main.py — Command-line runner (no web browser needed).
Runs the full ML pipeline in an OpenCV window.

Usage:
  python main.py                       # webcam (default)
  python main.py --source video.mp4    # video file
  python main.py --source 0 --no-plate # skip plate recognition (faster)
"""

import cv2
import argparse
import sys
import config
from modules.parking_detector import ParkingDetector
from modules.vehicle_counter   import VehicleCounter
from modules.plate_recognizer  import PlateRecognizer
from modules.utils             import draw_stats_panel


def parse_args():
    p = argparse.ArgumentParser(description="Smart Parking System — CLI Mode")
    p.add_argument("--source",    default=str(config.VIDEO_SOURCE),
                   help="Video source: 0 for webcam, or path to video file")
    p.add_argument("--no-plate",  action="store_true",
                   help="Disable number plate recognition (faster)")
    p.add_argument("--no-count",  action="store_true",
                   help="Disable vehicle counting")
    p.add_argument("--no-slots",  action="store_true",
                   help="Disable parking slot detection")
    return p.parse_args()


def main():
    args   = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print("  Smart Parking System — CLI Mode")
    print(f"  Source  : {source}")
    print(f"  Slots   : {'OFF' if args.no_slots  else 'ON'}")
    print(f"  Counter : {'OFF' if args.no_count  else 'ON'}")
    print(f"  Plates  : {'OFF' if args.no_plate  else 'ON'}")
    print(f"  Controls: Q=quit | P=pause | S=screenshot")
    print(f"{'='*50}\n")

    detector   = ParkingDetector()  if not args.no_slots  else None
    counter    = VehicleCounter()   if not args.no_count  else None
    recognizer = PlateRecognizer()  if not args.no_plate  else None

    paused    = False
    frame_num = 0
    import time
    prev_t    = time.time()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame_num += 1

        # ── ML Pipeline ────────────────────────────────────────────────────
        combined_stats = {}

        if detector:
            frame, slot_stats  = detector.detect(frame)
            combined_stats.update(slot_stats)

        if counter:
            frame, count_stats = counter.process(frame)
            combined_stats.update(count_stats)

        if recognizer:
            frame, plates      = recognizer.process(frame)
            if plates:
                print(f"  [PLATE] " + " | ".join(p["plate"] for p in plates))

        # ── FPS ─────────────────────────────────────────────────────────────
        now    = time.time()
        fps    = 1.0 / max(now - prev_t, 1e-9)
        prev_t = now
        combined_stats["FPS"] = f"{fps:.1f}"
        if paused:
            combined_stats["PAUSED"] = "yes"

        draw_stats_panel(frame, combined_stats)

        cv2.imshow("Smart Parking System  |  Q=quit  P=pause  S=screenshot", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('s'):
            fname = f"screenshot_{frame_num}.jpg"
            cv2.imwrite(fname, frame)
            print(f"  [SAVED] {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nSession ended.")


if __name__ == "__main__":
    main()

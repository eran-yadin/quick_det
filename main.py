import argparse
import os
import sys
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO human detection (live or from file)")
    parser.add_argument("--source", "-s", default="0",
                        help="Camera index (integer) or path to video/image. Use an integer for camera (default: 0).")
    parser.add_argument("--weights", "-w", default="yolov8n.pt",
                        help="Path to model weights file (default: yolov8n.pt).")
    parser.add_argument("--conf", "-c", type=float, default=0.5,
                        help="Confidence threshold for detections (default: 0.5).")
    parser.add_argument("--classes", type=str, default="0",
                        help="Comma-separated class ids to detect (default: '0' for person in COCO).")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable visual display window (useful for headless runs).")
    parser.add_argument("--save", "-o", default=None,
                        help="Path to save output video (e.g., output.mp4). If omitted, no file is written.")
    parser.add_argument("--device", default=None,
                        help="Device to run model on, e.g., 'cpu' or 'cuda'. If omitted, Ultralytics decides.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detection coordinates and extra info to console.")
    parser.add_argument("--open-timeout", type=float, default=10.0,
                        help="Seconds to wait for camera/source to open before failing (default: 10).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse source (camera index or path)
    source = int(args.source) if args.source.isdigit() else args.source

    # Parse classes into list[int]
    classes_list = [int(x) for x in args.classes.split(",") if x.strip() != ""]

    # Load model (may take time on first run)
    if args.verbose:
        print(f"Loading model from {args.weights}...")
    model = YOLO(args.weights)
    if args.verbose:
        print("Model loaded.")

    # Try to move model to device if provided
    if args.device:
        try:
            if args.verbose:
                print(f"Moving model to device: {args.device}")
            model.to(args.device)
        except Exception:
            if args.verbose:
                print("Warning: could not set device (continuing with default).")

    # Open source (camera or file)
    if args.verbose:
        print(f"Opening source: {source}")
    cap = cv2.VideoCapture(source)

    # Wait for camera/source to become available up to timeout
    import time
    start = time.time()
    while not cap.isOpened():
        if time.time() - start > args.open_timeout:
            print(f"Error: Could not open source within {args.open_timeout} seconds.")
            return
        time.sleep(0.2)
    if args.verbose:
        print("Source opened successfully.")

    writer = None
    print("Press 'q' to quit (if display enabled).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # Run inference
        results = model(frame, classes=classes_list, conf=args.conf, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                rel_x = round(center_x / width, 3)
                rel_y = round(center_y / height, 3)

                if args.verbose:
                    print(f"Human detected at: X={rel_x}, Y={rel_y} (Relative)")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {rel_x}, {rel_y}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Initialize writer if requested
        if args.save and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.save, fourcc, 20.0, (width, height))

        if writer is not None:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("YOLO Human Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import argparse
import os
import cv2
from ultralytics import YOLO
from collections import deque
import zmq
import time
import multiprocessing

# GPIO Handling (Mock for non-Raspberry Pi systems)
try:
    # 1. Try to load the REAL library (Works on Pi)
    import RPi.GPIO as GPIO
except (ImportError, RuntimeError):
    # 2. If that fails (Works on Windows), load this MOCK class
    print("Using Mock GPIO")
    class GPIO:
        BCM = "BCM"
        Board = "BOARD"
        OUT = "OUT"
        HIGH = 1
        LOW = 0
        
        @staticmethod
        def setmode(mode): print(f"Mock Mode: {mode}")
        @staticmethod
        def setup(pin, mode): print(f"Mock Setup: {pin}")
        @staticmethod
        def output(pin, state): print(f"Mock Output: {pin} -> {state}")
        @staticmethod
        def cleanup(): print("Mock Cleanup")


def io_worker(queue, config, verbose):
    """Worker process to handle IO (ZMQ and GPIO) without blocking the main loop."""
    # Setup ZMQ in this process
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")
    if verbose:
        print("IO Worker: Server started on Port 5555")

    # Setup GPIO in this process
    if config.get("use_gpio", True):
        mode = config.get("setmode", "BCM")
        if mode == "BOARD":
            GPIO.setmode(GPIO.Board)
        else:
            GPIO.setmode(GPIO.BCM)
        GPIO.setup(18, GPIO.OUT)

    last_alert_time = 0
    while True:
        person_detected = queue.get()
        if person_detected is None: # Sentinel to stop
            break

        if config.get("use_gpio", True):
            GPIO.output(18, GPIO.HIGH if person_detected else GPIO.LOW)

        if person_detected and (time.time() - last_alert_time > 1):
            socket.send_string("alert Person Detected!")
            if verbose:
                print("Sent: Person Detected!")
            last_alert_time = time.time()

def load_config(file_path):
    """Simple parser for the custom config.cfg format."""
    config = {}
    if not os.path.exists(file_path):
        return config
    
    with open(file_path, 'r') as f:
        for line in f:
            # Remove comments and whitespace
            line = line.split('#')[0].strip()
            if not line or '=' not in line:
                continue
            
            key, val = line.split('=', 1)
            key, val = key.strip(), val.strip()
            
            # Parse values
            if val.lower() == 'true': val = True
            elif val.lower() == 'false': val = False
            elif (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            elif val.isdigit(): val = int(val)
            
            config[key] = val
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO human detection (live or from file)")
    parser.add_argument("--source", "-s", default="0",
                        help="Camera index (integer) or path to video/image. Use an integer for camera (default: 0).")
    parser.add_argument("--weights", "-w", default="yolov8n.pt",
                        help="Model weights (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt). Default: yolov8n.pt")
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
    parser.add_argument("--max-history", type=int, default=5,
                        help="Maximum history length for movement tracking (default: 5).")

    # Load defaults from config.cfg if it exists
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.cfg")
    config = load_config(config_path)
    
    defaults = {}
    if "video_source" in config: defaults["source"] = str(config["video_source"]) # Ensure string for argparse
    if "model" in config: defaults["weights"] = config["model"]
    if "confidence_threshold" in config: defaults["conf"] = config["confidence_threshold"]
    if "device" in config: defaults["device"] = config["device"]
    if "no_display" in config: defaults["no_display"] = config["no_display"]
    if config.get("video_save") and "output_path" in config:
        defaults["save"] = config["output_path"]
    if "device" in config: defaults["device"] = config["device"]
    if "verbose" in config: defaults["verbose"] = config["verbose"]
    if "open_timeout" in config: defaults["open_timeout"] = config["open_timeout"]
    if "max_history" in config: defaults["max_history"] = config["max_history"]


    parser.set_defaults(**defaults)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.cfg"))
    box_x1 = float(config.get("square_x1", 0.2))
    box_y1 = float(config.get("square_y1", 0.2))
    box_x2 = float(config.get("square_x2", 0.8))
    box_y2 = float(config.get("square_y2", 0.8))

    # Performance optimizations
    skip_frames = int(config.get("skip_frames", 0))
    resize_width = int(config.get("resize_width", 0))

    # Start IO Process
    io_queue = multiprocessing.Queue()
    io_process = multiprocessing.Process(target=io_worker, args=(io_queue, config, args.verbose))
    io_process.daemon = True
    io_process.start()
    
    # Parse source (camera index or path)
    source = int(args.source) if args.source.isdigit() else args.source
    if args.verbose:
        print(f"Using source: {source}")
    print("-" * 30)
    # Parse classes into list[int]
    classes_list = [int(x) for x in args.classes.split(",") if x.strip() != ""]
    if args.verbose:    
        print(f"Detecting classes: {classes_list}")
        print("-" * 30)

    # Load model (may take time on first run)
    if args.verbose:
        print(f"Loading model from {args.weights}...")
    src = "models\\"+args.weights
    model = YOLO(src)
    if args.verbose:
        print("✅ Model loaded.")

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
    if args.verbose: 
        print("⏰ Waiting for source to open...")
    while not cap.isOpened():
        if time.time() - start > args.open_timeout:
            print(f"Error: Could not open source within {args.open_timeout} seconds.")
            return
        time.sleep(0.2)
    if args.verbose:
        print("Source opened successfully.")


    writer = None
    print("Press 'q' to quit (if display enabled).")

    track_histories = {}
    frame_count = -1
    last_results = []

    while True:
        ret, frame = cap.read() #ret = boolean, frame = image array(numpy)

        if not ret: #check if frame is read correctly
            break
        
        # Resize frame for performance if configured
        if resize_width > 0:
            h, w = frame.shape[:2]
            new_height = int(h * (resize_width / w))
            frame = cv2.resize(frame, (resize_width, new_height))

        frame_count += 1
        height, width, _ = frame.shape #get frame dimensions
        draw_alert_box(frame, box_x1,box_y1,box_x2,box_y2, "Alert Zone")
        
        # Frame Skipping Logic
        if skip_frames > 0 and (frame_count % (skip_frames + 1) != 0):
            results = last_results
        else:
            # Run tracking
            results = model.track(frame, classes=classes_list, conf=args.conf, persist=True, verbose=False)
            last_results = results
        

        person_in_zone_this_frame = False
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                #check if the detection is in the alert box
                if (center_x > box_x1 * width and center_x < box_x2 * width and
                    center_y > box_y1 * height and center_y < box_y2 * height):
                    person_in_zone_this_frame = True



                rel_x = round(center_x / width, 3)
                rel_y = round(center_y / height, 3)

                if args.verbose:
                    print(f"Human detected at: X={rel_x}, Y={rel_y} (Relative) id={box.id}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"Person {rel_x}, {rel_y}"
                
                # Handle tracking
                if box.id is not None:
                    track_id = int(box.id.item())
                    label += f" ID: {track_id}"
                    
                    if track_id not in track_histories:
                        track_histories[track_id] = deque(maxlen=args.max_history)
                    track_histories[track_id].append((center_x, center_y))
                    
                    if len(track_histories[track_id]) >= 2:
                        points = track_histories[track_id]
                        
                        # Draw trail
                        if config["show_trail"]:
                            for i in range(1, len(points)):
                                cv2.line(frame, points[i-1], points[i], (0, 255, 255), 2)
                        
                        # Draw arrow for movement direction        
                        if config["show_arrow"]:
                             cv2.arrowedLine(frame, points[0], points[-1], (0, 0, 255), 3, tipLength=0.5)

                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Send state to IO process
        io_queue.put(person_in_zone_this_frame)


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
    io_queue.put(None) # Stop worker
    io_process.join()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def draw_alert_box(frame, r_x1, r_y1, r_x2, r_y2, alert_text):
    x1 = round(r_x1 * frame.shape[1])
    y1 = round(r_y1 * frame.shape[0])
    x2 = round(r_x2 * frame.shape[1])
    y2 = round(r_y2 * frame.shape[0])
    """Draws a red alert box with text on the frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(frame, alert_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



if __name__ == "__main__":
    print("Starting YOLO Human Detection...")
    main()

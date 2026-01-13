import argparse
import os
import cv2
import numpy as np
import zmq
import time
import multiprocessing
import ast
from collections import deque
from ultralytics import YOLO

# --- GPIO Handling (Mock for non-Raspberry Pi systems) ---
try:
    import RPi.GPIO as GPIO
except (ImportError, RuntimeError):
    print("⚠️ Using Mock GPIO (Not running on Raspberry Pi)")
    class GPIO:
        BCM = "BCM"
        BOARD = "BOARD"
        OUT = "OUT"
        HIGH = 1
        LOW = 0
        
        @staticmethod
        def setmode(mode): print("setmode = "+mode)
        @staticmethod
        def setup(pin, mode): print("setup: pin="+str(pin)+" ,mode="+mode)
        @staticmethod
        def output(pin, state): print("output: pin="+str(pin)+" ,state="+str(state))
        @staticmethod
        def cleanup(): print("Mock Cleanup")

# --- Helper Functions ---

def load_config(file_path):
    """Simple parser for the custom config.cfg format."""
    config = {}
    if not os.path.exists(file_path):
        return config
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line or '=' not in line:
                continue
            
            key, val = line.split('=', 1)
            key, val = key.strip(), val.strip()
            
            try:
                val = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                if val.lower() == 'true': val = True
                elif val.lower() == 'false': val = False
            
            config[key] = val
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO human detection")
    parser.add_argument("--source", "-s", default="0", help="Camera index or file path.")
    parser.add_argument("--weights", "-w", default="yolov8n.pt", help="Model weights.")
    parser.add_argument("--conf", "-c", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--classes", type=str, default="0", help="Class IDs (0 for person).")
    parser.add_argument("--no-display", action="store_true", help="Disable window.")
    parser.add_argument("--save", "-o", default=None, help="Save video path.")
    parser.add_argument("--device", default=None, help="Device (cpu/cuda).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output.")
    parser.add_argument("--open-timeout", type=float, default=10.0, help="Source open timeout.")
    parser.add_argument("--max-history", type=int, default=5, help="Movement history length.")

    # Load defaults from config if available
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.cfg")
    config = load_config(config_path)
    
    defaults = {}
    if "video_source" in config: defaults["source"] = str(config["video_source"])
    if "model" in config: defaults["weights"] = config["model"]
    if "confidence_threshold" in config: defaults["conf"] = config["confidence_threshold"]
    if "device" in config: defaults["device"] = config["device"]
    if "max_history" in config: defaults["max_history"] = config["max_history"]
    
    parser.set_defaults(**defaults)
    return parser.parse_args()

def io_worker(queue, config, verbose):
    """Worker process for ZMQ and GPIO."""
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")
    
    if config.get("use_gpio", True):
        mode = config.get("setmode", "BCM")
        GPIO.setmode(GPIO.BOARD if mode == "BOARD" else GPIO.BCM)
        GPIO.setup(18, GPIO.OUT)

    last_alert_time = 0
    
    while True:
        alert_state = queue.get()
        if alert_state is None: # Sentinel to stop
            break

        # Trigger GPIO
        if config.get("use_gpio", True):
            GPIO.output(18, GPIO.HIGH if alert_state else GPIO.LOW)

        # Trigger ZMQ Message (limit to once per second)
        if alert_state and (time.time() - last_alert_time > 1):
            socket.send_string("alert Person Detected!")
            if verbose: print(">> Alert Sent!")
            last_alert_time = time.time()

def is_inside_polygon(point, polygon):
    """
    Checks if point (x,y) is inside the polygon.
    Returns: True if inside or on edge.
    """
    # measureDist=False returns +1 (inside), -1 (outside), 0 (edge)
    result = cv2.pointPolygonTest(polygon, point, False)
    return result >= 0

def draw_poly_zone(frame, polygon, color=(0, 255, 0), thickness=2):
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=thickness)

def is_moving_in_direction(p1, p2, target_vector, tolerance=0.5):
    """
    Calculates if movement from p1 to p2 matches target_vector direction.
    """
    # 1. Calculate movement vector
    movement = np.array(p2) - np.array(p1)
    
    # Ignore tiny movements (jitter)
    if np.linalg.norm(movement) < 2.0: 
        return False

    # 2. Normalize vectors
    move_norm = movement / np.linalg.norm(movement)
    target_norm = target_vector / np.linalg.norm(target_vector)

    # 3. Dot Product
    dot_product = np.dot(move_norm, target_norm)
    
    if tolerance is None:
        tolerance = 0.5

    # Check if direction matches within tolerance (1.0 is exact match)
    return bool(dot_product > (1.0 - tolerance))

# --- Main Execution ---

def main():
    args = parse_args()
    config = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.cfg"))
    
    # 1. Setup Configuration
    # Target Direction Vector (e.g., [0, 1] is DOWN, [0, -1] is UP)
    vec_x = float(config.get("vector_x", 0))
    vec_y = float(config.get("vector_y", 1)) # Default to 'Down'
    VECTOR_TARGET = np.array([vec_x, vec_y])

    # Performance settings
    skip_frames = int(config.get("skip_frames", 0))
    resize_width = int(config.get("resize_width", 0))

    # Define Polygon Zone (Normalized 0.0 to 1.0)
    # Edit this array to change your shape!
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    p_1 = config.get("square_p1")
    p_2 = config.get("square_p2")
    p_3 = config.get("square_p3")
    p_4 = config.get("square_p4")

    poly_norm = np.array([
        p_1,
        p_2,
        p_3,
        p_4
    ])
    poly_pixels = None

    # 2. Start IO Background Process
    io_queue = multiprocessing.Queue()
    io_process = multiprocessing.Process(target=io_worker, args=(io_queue, config, args.verbose))
    io_process.daemon = True
    io_process.start()
    
    # 3. Load YOLO Model
    print(f"Loading model: {args.weights}...")
    # Handle path if weights are in a subfolder
    model_path = args.weights if os.path.exists(args.weights) else os.path.join("models", args.weights)
    model = YOLO(model_path)
    if args.device:
        model.to(args.device)

    # 4. Open Source
    src_val = int(args.source) if args.source.isdigit() else args.source
    if isinstance(src_val,str):
        src_val = "input\\"+src_val
    cap = cv2.VideoCapture(src_val)
    
    start_wait = time.time()
    while not cap.isOpened():
        if time.time() - start_wait > args.open_timeout:
            print(f"❌ Error: Could not open source {src_val}")
            return
        time.sleep(0.1)
    
    print("✅ System Started. Press 'q' to quit.")

    # Tracking Variables
    track_histories = {}
    frame_count = -1
    last_results = []
    classes_list = [int(x) for x in args.classes.split(",") if x.strip()]
    writer = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Resize for performance
            if resize_width > 0:
                h, w = frame.shape[:2]
                new_height = int(h * (resize_width / w))
                frame = cv2.resize(frame, (resize_width, new_height))
            
            h, w = frame.shape[:2]

            # Initialize Polygon Pixels (Do this once per resolution change)
            if poly_pixels is None:
                poly_pixels = (poly_norm * [w, h]).astype(np.int32).reshape((-1, 1, 2))

            # Draw Zone (Blue Polygon)
            draw_poly_zone(frame, poly_pixels, color=(255, 0, 0))

            frame_count += 1
            
            # Frame Skipping Logic
            if skip_frames > 0 and (frame_count % (skip_frames + 1) != 0):
                results = last_results
            else:
                results = model.track(frame, classes=classes_list, conf=args.conf, persist=True, verbose=False)
                last_results = results
            
            alert_active = False

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get Box Coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Calculate Center
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    center_point = (float(cx), float(cy))

                    # 1. Check Zone
                    in_zone = is_inside_polygon(center_point, poly_pixels)
                    color = (0, 255, 0) # Green (Safe)

                    # Tracking Logic
                    if box.id is not None:
                        tid = int(box.id.item())
                        
                        if tid not in track_histories:
                            track_histories[tid] = deque(maxlen=args.max_history)
                        track_histories[tid].append((cx, cy))
                        
                        # 2. Check Direction
                        points = track_histories[tid]
                        is_moving_correctly = False
                        
                        if len(points) >= 2:
                            is_moving_correctly = is_moving_in_direction(points[0], points[-1], VECTOR_TARGET, config.get("vector_tolerance"))
    
                            # VISUALS: Trail
                            if config.get("show_trail", True):
                                pts = np.array(list(points)).reshape((-1, 1, 2))
                                cv2.polylines(frame, [pts], False, (0, 255, 255), 2)

                            # VISUALS: Movement Arrow
                            if config.get("show_arrow", True):
                                cv2.arrowedLine(frame, points[0], points[-1], (0, 165, 255), 2, tipLength=0.5)

                        # 3. TRIGGER ALERT
                        if in_zone and is_moving_correctly:
                            alert_active = True
                            color = (0, 0, 255) # Red (Alert)
                            cv2.putText(frame, "ALERT!", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    # Draw Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID: {int(box.id) if box.id else '?'}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Send Alert to Worker
            io_queue.put(alert_active)

            # Draw Target Vector Helper (Top-Left Corner)
            # This shows you which way the system "wants" people to move
            start_pt = (50, 50)
            end_pt = (int(50 + VECTOR_TARGET[0] * 40), int(50 + VECTOR_TARGET[1] * 40))
            cv2.arrowedLine(frame, start_pt, end_pt, (255, 0, 255), 2, tipLength=0.3)
            cv2.putText(frame, "Target Dir", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # Video Writer
            if args.save:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save, fourcc, 20.0, (w, h))
                writer.write(frame)

            # Display
            if not args.no_display:
                cv2.imshow("Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        cap.release()
        io_queue.put(None)
        io_process.join()
        if writer: writer.release()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    # Windows Support for Multiprocessing
    multiprocessing.freeze_support()
    main()
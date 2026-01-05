import cv2
from ultralytics import YOLO

def main():
    # 1. Load the YOLO model (yolov8n.pt or yolov11n.pt)
    # The model will auto-download on first run
    model = YOLO('yolov8n.pt') 

    # 2. Initialize camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get frame dimensions
        height, width, _ = frame.shape

        # 3. Run inference (stream=True for better performance)
        # classes=[0] filters for 'person' in the COCO dataset
        results = model(frame, classes=[0], conf=0.5, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates for the bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # 4. Calculate relative coordinates (center of the person)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                rel_x = round(center_x / width, 3)
                rel_y = round(center_y / height, 3)

                # 5. CLI Output
                print(f"Human detected at: X={rel_x}, Y={rel_y} (Relative)")

                # 6. Draw the square/rectangle on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {rel_x}, {rel_y}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 7. Show live footage
        cv2.imshow("YOLO Human Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
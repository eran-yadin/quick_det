import cv2
from ultralytics import YOLO

def main():
    # Load the YOLOv8 model
    # 'yolov8n.pt' is the nano version, which is faster for CPU/real-time
    # It will automatically download the model weights if not present.
    model = YOLO('yolov8n.pt')

    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Get frame dimensions to calculate relative coordinates
        height, width, _ = frame.shape

        # Run YOLO detection on the frame
        # stream=True is more memory efficient for video loops
        # verbose=False keeps the console clean for your output
        results = model(frame, stream=True, verbose=False, classes=[0]) # class 0 is 'person'

        for result in results:
            for box in result.boxes:
                # Get absolute coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Calculate relative center position (0.0 - 1.0)
                center_x = ((x1 + x2) / 2) / width
                center_y = ((y1 + y2) / 2) / height

                print(f"Human detected at relative pos: ({center_x:.4f}, {center_y:.4f})")

                # Draw bounding box (square) around the human
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Human", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the output window
        cv2.imshow('YOLO Human Detection', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
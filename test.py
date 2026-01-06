import cv2

def test_camera_access():
    cap = cv2.VideoCapture(0)  # Open default camera
    assert cap.isOpened(), "Failed to open camera." # Check if camera opened successfully
    print("Camera opened successfully: backend name", cap.getBackendName)
    ret, frame = cap.read()
    assert ret, "Failed to read frame from camera." #check there is a frame to read
    assert frame is not None, "Frame is None." #check frame is not None
    if ret:
        print("Frame read successfully.")
    if frame is not None:
        print("Frame is valid.")
    height, width, _ = frame.shape #get frame dimensions
    assert height > 0 and width > 0, "Invalid frame dimensions." #get frame dimensions are valid
    print(f"Frame dimensions: {width}x{height}")
    cap.release() # Release the camera

def test_view_port():
    cap = cv2.VideoCapture(0)  # Open default camera
    print("Starting viewport. Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()  # ret = boolean, frame = image array(numpy)
        if not ret:  # check if frame is read correctly
            break

        height, width, _ = frame.shape  # get frame dimensions

        cv2.imshow('Camera Feed', frame)  # Display the frame

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_access()
    print("All camera access tests passed.")
    ans = input("do you want to test view port? (y/n): ")
    if ans.lower() == 'y':
        
        test_view_port()


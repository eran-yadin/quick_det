import cv2
import numpy as np
import os

CONFIG_FILE = 'config.cfg'

def get_video_source():
    """Reads the video source from the config file."""
    source = 0
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            for line in f:
                if line.strip().startswith('video_source'):
                    parts = line.split('=')
                    if len(parts) > 1:
                        # Clean up the value string
                        val = parts[1].strip().split('#')[0].strip().strip('"\'')
                        if val.isdigit():
                            source = int(val)
                        else:
                            source = val
                    break
    return source

def update_config(square_points, vector_coords):
    """Updates the config.cfg file with new coordinates."""
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: {CONFIG_FILE} not found.")
        return

    with open(CONFIG_FILE, 'r') as f:
        lines = f.readlines()

    # Prepare the new values
    updates = {
        'square_p1': f"[{square_points[0][0]:.3f}, {square_points[0][1]:.3f}]",
        'square_p2': f"[{square_points[1][0]:.3f}, {square_points[1][1]:.3f}]",
        'square_p3': f"[{square_points[2][0]:.3f}, {square_points[2][1]:.3f}]",
        'square_p4': f"[{square_points[3][0]:.3f}, {square_points[3][1]:.3f}]",
        'vector_x': f"{vector_coords[0]:.3f}",
        'vector_y': f"{vector_coords[1]:.3f}",
    }

    with open(CONFIG_FILE, 'w') as f:
        for line in lines:
            key = line.split('=')[0].strip()
            if key in updates:
                # Preserve existing comments
                comment = ""
                if '#' in line:
                    comment = "  #" + line.split('#', 1)[1].strip()
                f.write(f"{key} = {updates[key]}{comment}\n")
            else:
                f.write(line)
    print(f"Configuration saved to {CONFIG_FILE}.")

def configure_detector():
    """Main function to run the visual configuration tool."""
    source = get_video_source()
    print(f"Opening video source: {source}")
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    # Read one frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read frame from video source")
        return

    height, width = frame.shape[:2]
    points = []
    
    print("\n--- Configuration Instructions ---")
    print("1. Click 4 points to define the SQUARE area.")
    print("2. Click 2 points to define the VECTOR (Start -> End).")
    print("   (Total 6 clicks required)")
    print("Press 'r' to reset all points.")
    print("Press 's' to save and exit (enabled after 6 clicks).")
    print("Press 'q' to quit without saving.\n")

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 6:
                points.append((x, y))
                draw_overlay()

    def draw_overlay():
        display_img = frame.copy()
        
        # --- Draw Square (First 4 points) ---
        for i, pt in enumerate(points[:4]):
            cv2.circle(display_img, pt, 5, (0, 255, 0), -1) # Green dots
            # Draw lines between points for visualization
            if i > 0:
                 cv2.line(display_img, points[i-1], pt, (0, 255, 0), 1)
        
        if len(points) >= 4:
             # Close the loop for the 4 points
             cv2.line(display_img, points[3], points[0], (0, 255, 0), 1)
             
             # Draw the polygon
             pts = np.array(points[:4], np.int32).reshape((-1, 1, 2))
             cv2.polylines(display_img, [pts], True, (0, 255, 255), 2)

        # --- Draw Vector (Next 2 points) ---
        for i, pt in enumerate(points[4:]):
            color = (0, 0, 255) if i == 0 else (255, 0, 0) # Red start, Blue end
            cv2.circle(display_img, pt, 6, color, -1)
        
        if len(points) == 6:
            cv2.arrowedLine(display_img, points[4], points[5], (0, 255, 255), 3)
            
            # Show save instruction
            cv2.putText(display_img, "Press 's' to SAVE", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Configure Detector", display_img)

    cv2.namedWindow("Configure Detector")
    cv2.setMouseCallback("Configure Detector", mouse_callback)
    draw_overlay()

    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Configuration cancelled.")
            break
        elif key == ord('r'):
            points = []
            draw_overlay()
            print("Points reset.")
        elif key == ord('s'):
            if len(points) == 6:
                # 1. Get Square Points (Normalized)
                sq_points = []
                for pt in points[:4]:
                    sq_points.append((pt[0]/width, pt[1]/height))
                
                # 2. Calculate Vector
                v_start = points[4]
                v_end = points[5]
                # Vector relative to screen size
                vx = (v_end[0] - v_start[0]) / width
                vy = (v_end[1] - v_start[1]) / height
                
                update_config(sq_points, (vx, vy))
                break
            else:
                print(f"Incomplete. You need 6 points (4 for square, 2 for vector). Current: {len(points)}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    configure_detector()

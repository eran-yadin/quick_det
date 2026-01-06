1. Basic I/O (Images)

Read Image: `img = cv2.imread('file.jpg')`
Show Image: `cv2.imshow('Title', img)` followed by `cv2.waitKey(0)` to keep it open
Save Image: `cv2.imwrite('output.png', img)`
Destroy Windows: `cv2.destroyAllWindows()`

2. Video Capture Snippet
The standard loop for webcam `(index 0)` or video files:
```py
import cv2
cap = cv2.VideoCapture(0) 
```

- 0 for webcam,
- `'file.mp4'` for video

```py
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    cv2.imshow('Live Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to exit  
        break

cap.release() #release the camera
cv2.destroyAllWindows() #close any open windows
```


3. Essential Transformations

```py
    Resize: resized = cv2.resize(img, (width, height))
    Crop: cropped = img[y1:y2, x1:x2] (Using array slicing)
    Color Conversion: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Blur: blurred = cv2.GaussianBlur(img, (5, 5), 0)
    Thresholding: ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
```
4. Drawing Functions
Note: These modify the image in place.
```py
    Line: cv2.line(img, (x1, y1), (x2, y2), (B, G, R), thickness)
    Rectangle: cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    Circle: cv2.circle(img, (center_x, center_y), radius, (255, 0, 0), -1) (-1 fills the circle)
    Text: cv2.putText(img, 'Text', (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)
```
5. Common Metadata
```py
    Get FPS: cap.get(cv2.CAP_PROP_FPS)
    Get Width/Height: cap.get(cv2.CAP_PROP_FRAME_WIDTH) / HEIGHT
```
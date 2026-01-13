# Human Detection Project

Short project for live human detection using Ultralytics YOLO and OpenCV.

## **Setup**
- **Python:** use Python 3.10 or newer.
- **Create a virtual environment and activate it:**

```bash
python -m venv venv
venv\Scripts\activate
```

- **If you don't have `pip` installed:** the repository contains `get-pip.py`. Run it only if `pip` is missing:

```bash
python get-pip.py
```

- **Install dependencies:**

```bash
python -m pip install --upgrade pip
python -m pip install -r requireme~nts.txt
```

- **Run the app:**

```bash
python main.py
```

- **test input**
if you need to test your camera you can run the `test.py`
it will ask you if you want live view port
you can say no (n)
to run it using bash
 ```bash
python test.py
```
## set `detection square` and `exit vecotr`
for now:
`config.cfg`
```py
# square /relative
square_x1 = 0.3
square_y1 = 0.3
square_x2 = 0.7
square_y2 = 0.7 
#vector exit
vector_x = 0.3
vector_y = 0
```

## left to do
- make a GUI for choosing the active squre
- make docker continer for moving to diffrent systems
- make GUI for settings
- rusburry pi inplement (test)
- socket implement (test)

## Notes:
- `torch` and `torchvision` versions in `requirements.txt` may require a specific CUDA/cuDNN match for GPU acceleration. If you plan to use GPU, follow PyTorch install instructions at https://pytorch.org to install a compatible build.
- The file `yolov8n.pt` is included; the code will also let Ultralytics attempt to fetch the model if not present.

## **Project structure**
- `main.py` — main application script that captures camera frames, runs YOLO inference and displays bounding boxes and relative coordinates.
- `yolov8n.pt` — YOLOv8 model weights (small/neutral). Kept locally for offline runs.
- `requirements.txt` — pinned Python packages used by the project.
- `get-pip.py` — pip bootstrap script; only needed if your Python environment has no `pip`.

## **Full documentation of `main.py`**

File: [main.py](main.py)

High-level behavior:
- Loads a YOLO model via `YOLO('yolov8n.pt')`.
- Opens the default camera (`cv2.VideoCapture(0)`).
- For each frame: runs model inference filtered to class `0` (the COCO `person` class) with `conf=0.5`.
- For each detected person, computes the bounding box center and prints relative coordinates (X,Y normalized to frame width/height).
- Draws rectangles and labels on the frame and displays the live window. Press `q` to quit.

Important code notes and parameters:
- `model = YOLO('yolov8n.pt')` — loads local weights; if the file is not present Ultralytics may attempt to download the weights automatically.
- `cap = cv2.VideoCapture(0)` — opens camera index 0. Change the index or provide a file path to use a different source.
- `results = model(frame, classes=[0], conf=0.5, verbose=False)` —
  - `classes=[0]` restricts detection to the COCO `person` class. Remove or change to detect other classes.
  - `conf=0.5` sets confidence threshold. Lower to detect more candidates (more false positives), raise to be stricter.
  - The code passes the raw `frame` (NumPy BGR image) to the model, which the Ultralytics API accepts.
- Bounding box coordinates are taken from `box.xyxy[0]` and converted to integers for drawing with OpenCV.
- Relative coordinates printed as `rel_x` and `rel_y` are normalized by frame width and height and rounded to 3 decimals.

How to adapt `main.py`:
- To run on a video file, replace `cv2.VideoCapture(0)` with `cv2.VideoCapture('path/to/video.mp4')`.
- To save detections to disk or stream them over network, create a logger or socket send after computing `rel_x, rel_y`.
- To tune performance, consider using `model.predict()` or batching, and ensure you have a compatible `torch` build that uses CUDA (if available).

Troubleshooting
- If OpenCV fails to open the camera: verify camera index, close other applications using the camera, and ensure proper drivers.
- If `pip install -r requirements.txt` fails due to `torch` wheels: install `torch` manually using the correct CUDA option from https://pytorch.org and then re-run `pip install -r requirements.txt` excluding `torch`/`torchvision` or adjust versions.
- If Ultralytics throws errors loading the model, ensure `ultralytics` in `requirements.txt` is installed and compatible with the local `yolov8n.pt` file.


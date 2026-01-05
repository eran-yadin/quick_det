# Command-line arguments

This file documents every CLI argument added to `main.py`.

- `--source`, `-s` (default: `0`)
  - Camera index (integer) or path to a video/image file. Use an integer (e.g., `0`) to open the default camera.

- `--weights`, `-w` (default: `yolov8n.pt`)
  - Path to the YOLO model weights used for inference.

- `--conf`, `-c` (default: `0.5`)
  - Confidence threshold for filtering detections. Float between 0 and 1.

- `--classes` (default: `0`)
  - Comma-separated list of COCO class ids to detect. Example: `0` for person, `0,1` for person and bicycle.

- `--no-display` (flag)
  - If present, disables the OpenCV display window (useful for headless servers or when saving output only).

- `--save`, `-o`
  - Path to write output video (e.g., `output.mp4`). If omitted, output is not saved.

- `--device` (optional)
  - Preferred device for the model (e.g., `cpu` or `cuda`). If not provided, Ultralytics decides automatically.

- `--verbose`, `-v` (flag)
  - If present, prints detection coordinates and additional info to the console.

- `--open-timeout` (default: `10.0`)
  - Number of seconds to wait for the camera or source to open before exiting with an error. Increase if your camera or source is slow to initialize.

Usage examples:

Run webcam with defaults:

```bash
python main.py
```

Use a video file and save output:

```bash
python main.py --source path/to/video.mp4 --save output.mp4 --verbose
```

Detect multiple classes:

```bash
python main.py --classes 0,1,2
```

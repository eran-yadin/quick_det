YOLOv8 comes in 5 standard sizes. The trade-off is always between speed (how many frames per second) and accuracy (how well it detects objects).

## download more models
https://github.com/ultralytics/ultralytics?tab=readme-ov-file

I downloaded the `yolov8n` ,`yolov11n`, `yolov11s`, `yolov11m`

you can download the rest

please note: very model are **bigger** and **more gpu intensive**

## YOLOv8 Model Comparison

| Model Name | Size | Description | Best Use Case |
| --- | --- | --- | --- |
| **yolov8n.pt** | Nano | Fastest, smallest, least accurate. | Real-time on CPU or older laptops. (Your current default) |
| **yolov8s.pt** | Small | Slower than Nano, but more accurate. | A good upgrade if Nano misses too many people. |
| **yolov8m.pt** | Medium | Balanced speed and accuracy. | Requires a decent GPU for real-time performance. |
| **yolov8l.pt** | Large | Very accurate, but heavy/slow. | High-end GPUs only. |
| **yolov8x.pt** | X-Large | Most accurate, slowest. | Server-side processing or very powerful GPUs. |

## bash
```sh
# Use the Small model
python main.py --weights yolov8s.pt

# Use the Medium model
python main.py --weights yolov8m.pt
```



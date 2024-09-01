import cv2
from ultralytics import YOLO
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
from pathlib import Path

model = YOLO("yolov5nu.pt")

logging.basicConfig(level=logging.INFO)

load_dotenv()

source = f"rtsp://{os.getenv('CAMERA_USERNAME')}:{os.getenv('CAMERA_PASSWORD')}@{os.getenv('IP_ADDRESS')}:80/live/0/h264.sdp"
logging.info(f"Connecting to {source}")

results = model(source, stream=True, verbose=False)

(Path.cwd() / "imgs").mkdir(exist_ok=True)

cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Stream", 1280, 720)

def display_img(img, boxes, names):
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{names[box.cls.item()]} {box.conf.item():.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return img


count = 0
current_datetime = datetime.now()
for r in results:
    count += 1
    img = display_img(r.orig_img, r.boxes, r.names)
    cv2.imshow("Stream", img)
    labels = r.boxes.cls

    new_datetime = datetime.now()
    if 0 in labels and (new_datetime - current_datetime).seconds >= 2:
        current_datetime = new_datetime
        index = r.boxes.cls.argmin()
        conf = r.boxes.conf[index]
        str_time = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        logging.info(f"Person detected with confidence {conf:.2f}")
        cv2.imwrite(f"imgs/frame_{str_time}.jpg", img)
        with open(f"imgs/frame_{str_time}.txt", "w") as f:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_index = box.cls.item()
                confidence = box.conf.item()
                
                # Normalize coordinates
                img_h, img_w, _ = img.shape
                x_center = (x1 + x2) / 2.0 / img_w
                y_center = (y1 + y2) / 2.0 / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                
                f.write(f"{class_index} {x_center} {y_center} {width} {height} {confidence}\n")

    if count == 10000:
        break

cv2.destroyAllWindows()

import cv2
import torch
import os
import json
from collections import Counter
import matplotlib.pyplot as plt

# Load pretrained YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Input video
cap = cv2.VideoCapture('input_video.mp4')
frame_index = 0
results_json = []
class_counts = Counter()
max_div_frame = {'frame': 0, 'classes': set()}

os.makedirs("output/frames", exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_index % 5 == 0:
        result = model(frame)
        detections = result.pred[0]
        frame_data = {'frame_index': frame_index, 'objects': []}
        classes_in_frame = set()

        for *box, conf, cls in detections.tolist():
            label = result.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)
            conf = round(conf, 2)
            frame_data['objects'].append({
                'label': label,
                'bbox': [x1, y1, x2, y2],
                'confidence': conf
            })
            class_counts[label] += 1
            classes_in_frame.add(label)

        if len(classes_in_frame) > len(max_div_frame['classes']):
            max_div_frame = {'frame': frame_index, 'classes': classes_in_frame}

        results_json.append(frame_data)

    frame_index += 1

cap.release()

with open("output/summary.json", "w") as f:
    json.dump(results_json, f, indent=4)

# Bar chart
plt.bar(class_counts.keys(), class_counts.values())
plt.title("Object Frequency")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/object_frequency.png")

print(" Detection completed.")
print(" Frame with max class diversity:", max_div_frame['frame'], "→", max_div_frame['classes'])
print(" Object frequency chart saved as 'output/object_frequency.png'")
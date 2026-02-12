import cv2
from ultralytics import YOLO

# Load YOLOv8 model (tiny and fast)
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Define object weights for scoring
# Define object weights for scoring
weights = {
    "bottle": 4, "cup": 3, "wine glass": 4, "bowl": 3,
    "banana": 2, "apple": 2, "sandwich": 2, "orange": 2, "broccoli": 2, "carrot": 2, 
    "hot dog": 5, "pizza": 5, "donut": 5, "cake": 5,
    "chair": 2, "couch": 2, "potted plant": 2, "bed": 5, "dining table": 2, "toilet": 5,
    "tv": 5, "laptop": 2, "mouse": 1, "remote": 2, "keyboard": 1, "cell phone": 3,
    "microwave": 5, "oven": 5, "toaster": 5, "sink": 5, "refrigerator": 5,
    "book": 2, "clock": 2, "vase": 2, "scissors": 4, "teddy bear": 3, "hair drier": 5, "toothbrush": 5,
    "backpack": 4, "umbrella": 4, "handbag": 4, "tie": 3, "suitcase": 5,
    # Original items (some might overlap or not be standard COCO, keeping for safety if model is custom, otherwise ignored)
    "pen": 1, "paper": 2, "mug": 3, "trash": 5, "cable": 3, "clothes": 3, 
}

# Compute chaos score from results
def compute_chaos_score(results):
    score = 0
    for box in results[0].boxes:
        label_idx = int(box.cls)
        label = results[0].names[label_idx]

        if label == "person":
            continue

        score += weights.get(label, 1)
    return score

def get_chaos_level(score):
    if score == 0:
        return "✨ Spotless! No clutter detected."
    elif score <= 5:
        return "🧼 Very tidy desk. Keep it up!"
    elif score <= 10:
        return "🗂️ Slightly messy, but manageable."
    elif score <= 20:
        return "📚 Messy — might need some cleaning."
    else:
        return "🚨 Disaster zone! Time to declutter."

import time

# Initialize FPS variables
prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()
    for box in results[0].boxes:
        class_idx = int(box.cls)
        label = results[0].names[class_idx]

        if label == "person":
            continue

        print(f"Detected: {label} (class index: {class_idx})")
    score = compute_chaos_score(results)

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)

    # Display Chaos Score and FPS
    cv2.putText(annotated_frame, f"Chaos Score: {score}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(annotated_frame, f"FPS: {fps}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Messy Desk Scanner", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

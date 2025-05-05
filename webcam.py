import cv2
from ultralytics import YOLO

# Load YOLOv8 model (tiny and fast)
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Define object weights for scoring
weights = {
    "pen": 1, "paper": 2, "mug": 3, "trash": 5, "book": 2, "cable": 3, "bottle": 4, "clothes": 3,
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

while True:
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

    cv2.putText(annotated_frame, f"Chaos Score: {score}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Messy Desk Scanner", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import torch
import urllib.request
import cv2
import numpy as np

# Load models
yolo_model = YOLO("yolov8n.pt")  # or yolov8m.pt for better accuracy
emotion_model = load_model("mobilenet_dog_behavior_model.h5")
class_labels = ['angry', 'happy', 'relaxed', 'sad']

# Constants for distance estimation
KNOWN_DISTANCE_CM = 100  # adjust based on your calibration
KNOWN_HEIGHT_PX = 150    # height in pixels of object at known distance
FOCAL_LENGTH = (KNOWN_HEIGHT_PX * KNOWN_DISTANCE_CM) / KNOWN_HEIGHT_PX  # = 100

# Distance estimation function
def estimate_distance(bbox_height):
    if bbox_height == 0:
        return None
    distance = (FOCAL_LENGTH * KNOWN_DISTANCE_CM) / bbox_height
    return distance

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame, classes=[16])  # class 16 = dog

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(coord.item()) for coord in box.xyxy[0]]
            confidence = float(box.conf[0])

            if confidence > 0.5:
                bbox_height = y2 - y1
                if bbox_height <= 0:
                    continue

                distance_cm = estimate_distance(bbox_height)
                print(f"Bounding Box Height: {bbox_height}")
                print(f"Estimated Distance: {distance_cm:.2f} cm")

                # Crop dog region
                dog_roi = frame[y1:y2, x1:x2]
                if dog_roi.size == 0:
                    continue

                # Emotion prediction
                img = cv2.resize(dog_roi, (224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                prediction = emotion_model.predict(img_array)
                predicted_class = class_labels[np.argmax(prediction)]

                # Draw rectangle and info
                label = f"{predicted_class} ({confidence*100:.1f}%)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if distance_cm:
                    cv2.putText(frame, f"{int(distance_cm)} cm away", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Dog Emotion and Distance Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

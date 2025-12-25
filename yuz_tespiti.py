import cv2
import mediapipe as mp
import numpy as np
from transformers import pipeline
from PIL import Image

# -------- AGE MODEL --------
age_classifier = pipeline(
    "image-classification",
    model="nateraw/vit-age-classifier"
)

# -------- GENDER MODEL --------
gender_net = cv2.dnn.readNet(
    "models/gender_net.caffemodel",
    "models/gender_deploy.prototxt"
)
GENDER_LIST = ["Male", "Female"]

def predict_gender(face_img):
    blob = cv2.dnn.blobFromImage(
        face_img, 1.0, (227, 227),
        (78.42, 87.76, 114.89),
        swapRB=False
    )
    gender_net.setInput(blob)
    return GENDER_LIST[gender_net.forward()[0].argmax()]

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# -------- CAMERA --------
cap = cv2.VideoCapture(0)
mp_face = mp.solutions.face_detection

frame_count = 0
age = "?"
gender = "?"

with mp_face.FaceDetection(min_detection_confidence=0.6) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            face = frame[y:y+bh, x:x+bw]

            # 🔑 Her frame'de değil, 15 frame'de 1 tahmin
            if face.size > 0 and frame_count % 15 == 0:
                gender = predict_gender(face)
                age = age_classifier(cv2_to_pil(face))[0]['label']

            cv2.rectangle(frame, (x,y), (x+bw,y+bh), (0,255,0), 2)
            cv2.putText(
                frame,
                f"{gender}, Age: {age}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

        cv2.imshow("Age & Gender Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

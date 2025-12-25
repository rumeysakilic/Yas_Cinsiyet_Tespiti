import cv2
import mediapipe as mp

# MediaPipe face detection modülleri
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Kamerayı aç
cap = cv2.VideoCapture(0)

# Face detection modeli
with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detector:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kare alınamadı!")
            break

        # BGR → RGB dönüşümü (MediaPipe için gerekli)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)

        # Yüz bulunduysa çiz
        if results.detections:
            for detection in results.detections:
                mp_draw.draw_detection(frame, detection)

        # Görüntüyü göster
        cv2.imshow("MediaPipe Face Detection", frame)

        # Q ile çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load model
model = joblib.load("scripts/sign_model.pkl")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            # Prediction
            X = np.array(row).reshape(1, -1)
            prediction = model.predict(X)[0]
            confidence = model.predict_proba(X).max()

            # Display
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"{prediction} ({confidence*100:.1f}%)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
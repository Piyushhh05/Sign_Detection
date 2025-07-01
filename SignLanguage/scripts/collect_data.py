
import cv2
import mediapipe as mp
import pandas as pd
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Start webcam
cap = cv2.VideoCapture(0)
data = []

label = input("Enter label for this sign (e.g., A, B, Hello): ")

print("📷 Webcam started. Press ESC to stop recording...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        print("✋ Hand detected!")
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            row.append(label)
            data.append(row)

    # Show the frame with landmarks (if detected)
    cv2.imshow("Collecting", frame)

    key = cv2.waitKey(1) & 0xFF
    if key != 255:
       print(f"Key pressed: {chr(key)} ({key})")
    if key == 27:  # ESC key
       print("ESC pressed! Exiting.")
       break

cap.release()
cv2.destroyAllWindows()

# Save data if available
if data:
    columns = [f"{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']] + ["label"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("data/hand_landmarks.csv", mode='a', header=False, index=False)
    print(f"✅ Data saved to data/hand_landmarks.csv. Total records: {len(data)}")
else:
    print("⚠️ No hand landmarks were recorded. Nothing was saved.")

from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("scripts/sign_model.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
# Global variable to store prediction
latest_prediction = ""

def generate_frames():
    global latest_prediction
    

    while True:
        success, frame = cap.read()
        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                
                # Predict and update global variable
                if len(row) == 63:  # 21 landmarks * 3
                    prediction = model.predict([row])[0]
                    latest_prediction = prediction
                    print("✋ Predicted:", latest_prediction)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    return jsonify({'prediction': latest_prediction})

if __name__ == '__main__':
    print("✅ Starting Flask...")
    app.run(debug=True, port=8000)
from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
from landmarks import landmarks

# Load the model
try:
    with open('deadlift.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Initialize Flask app
app = Flask(__name__)

# Global variables
current_stage = ''
counter = 0
bodylang_prob = np.array([0, 0])
bodylang_class = ''
smooth_factor = 5
recent_predictions = []

# Normalization function
def normalize_data(row):
    row = np.array(row).reshape(-1, 4)
    max_values = np.max(row, axis=0)
    min_values = np.min(row, axis=0)
    row = (row - min_values) / (max_values - min_values)
    return row.flatten().tolist()

def gen_frames():
    global current_stage, counter, bodylang_class, bodylang_prob, recent_predictions
    cap = cv2.VideoCapture(0)  # Open the camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return  # Exit if the camera cannot be accessed

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.resize(frame, (640, 480))  # Resize the frame for display
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
                                      mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10))

            try:
                row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
                row = normalize_data(row)
                X = pd.DataFrame([row], columns=landmarks)
                bodylang_prob = model.predict_proba(X)[0]
                bodylang_class = model.predict(X)[0]

                recent_predictions.append(bodylang_class)
                if len(recent_predictions) > smooth_factor:
                    recent_predictions.pop(0)
                smoothed_class = max(set(recent_predictions), key=recent_predictions.count)

                # Adjusting threshold and stage detection logic
                threshold = 0.65  # Adjusted threshold value
                if smoothed_class == "down" and bodylang_prob.max() > threshold:
                    current_stage = "down"
                elif current_stage == "down" and smoothed_class == "up" and bodylang_prob.max() > threshold:
                    current_stage = "up"
                    counter += 1

                # Debugging outputs
                print(f"Current stage: {current_stage}, Counter: {counter}, Probability: {bodylang_prob.max():.2f}")
                print(f"Predicted class: {bodylang_class}, Probability: {bodylang_prob.max():.2f}")

            except Exception as e:
                print(f"Error during pose classification: {e}")

        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  # Release the camera when done


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_data')
def update_data():
    global current_stage, counter, bodylang_prob
    # Returning data as JSON
    return jsonify(stage=current_stage, counter=counter, prob=f"{bodylang_prob.max():.2f}")

if __name__ == "__main__":
    app.run(debug=True)

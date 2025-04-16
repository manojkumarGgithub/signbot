import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Load the sign language model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Flask app
app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label mapping (A-Y, Space, Delete)
labels_dict = {i: chr(65 + i) for i in range(25)}  # A to Y
labels_dict[26] = " "  # Space
labels_dict[27] = "Delete"  # Delete

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to process images
def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)
    detected_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_, y_, data_aux = [], [], []

            for i in range(len(hand_landmarks.landmark)):
                x_.append(hand_landmarks.landmark[i].x)
                y_.append(hand_landmarks.landmark[i].y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_index = int(prediction[0])
                detected_text += labels_dict.get(predicted_index, "")

    return detected_text

# Function to process videos
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    detected_text = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_, y_, data_aux = [], [], []

                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

                if len(data_aux) == 42:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_index = int(prediction[0])
                    detected_text += labels_dict.get(predicted_index, "")

    cap.release()
    return detected_text

# Route for uploading and processing images/videos
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if filename.split('.')[-1].lower() in {'png', 'jpg', 'jpeg'}:
            detected_text = process_image(filepath)
        else:
            detected_text = process_video(filepath)

        return render_template("test.html", detected_text=detected_text)

    return jsonify({"error": "Invalid file type"})

# Route for rendering upload page
@app.route('/')
def index():
    return render_template('samp.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

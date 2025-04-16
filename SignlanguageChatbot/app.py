import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import google.generativeai as genai
import os
from flask import Flask, render_template, Response, jsonify
from flask_mail import Mail, Message
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# Set up Google Generative AI API
genai.configure(api_key='AIzaSyAWtVNjcUz82dOMCZVPxUUL_j0EemlYkkc')

# Load the sign language model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Flask app
app = Flask(__name__)




def send_violation_email():
    """Send email alert when 'H' is detected three times."""
    
    # Gmail SMTP server details
    smtp_server = "smtp.gmail.com"
    smtp_port = 465  # SSL port

    # Sender details
    sender_email = "manojkumer844@gmail.com"
    app_password = "odal faip qjab azsk"  # Use an App Password, NOT your regular password

    # Recipient details
    recipient_email = "22341A4515@gmrit.edu.in"

    # Email Content
    subject = "Sign Language Alert: H Detected 3 Times"
    body = "hey i am in danger"

    # Create Email Message
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        # Connect to Gmail SMTP server using SSL
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender_email, app_password)  # Login with App Password
        server.sendmail(sender_email, recipient_email, msg.as_string())  # Send email
        server.quit()
        
        print("📧 Alert email sent successfully!")
    except Exception as e:
        print(f"❌ Error sending email: {e}")



# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Mapping of labels (A-Y) + Space (26) + Delete (27)
labels_dict = {i: chr(65 + i) for i in range(25)}  # A to Y
labels_dict[26] = " "  # Space
labels_dict[27] = "Delete"  # Delete

# Initialize variables
sign_string = ""  # To store the detected letters
last_sign_time = time.time()  # Timer for inactivity
inactive_threshold = 3  # Seconds to wait before adding a space
letter_confirmation_time = 3  # Seconds to confirm the same letter

current_letter = None
letter_start_time = None

# Delete gesture flag and timer
delete_triggered = False
delete_start_time = None  # Timer to track when the delete gesture starts

# Initialize video capture
cap = cv2.VideoCapture(0)

# Function to generate frames for the video feed
def gen_frames():
    global sign_string, current_letter, letter_start_time, delete_triggered, delete_start_time

    while True:
        data_aux = []  # Feature list (should contain 42 features)
        x_ = []  # To store x coordinates
        y_ = []  # To store y coordinates

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Extract landmarks (x and y coordinates)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize x and y coordinates, and add them to the feature list
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Normalize x
                    data_aux.append(y - min(y_))  # Normalize y

                # Ensure data_aux has only 42 features (21 landmarks * 2 coordinates)
                if len(data_aux) != 42:
                    continue

                # Prediction
                prediction = model.predict([np.asarray(data_aux)])

                # Get the predicted index
                predicted_index = int(prediction[0])

                if predicted_index == 26:
                    predicted_character = " "  # Handle the special case of 26 (Space)
                elif predicted_index == 27:
                    predicted_character = "Delete"  # Handle the special case of 27 (Delete)

                    # Handle delete action: Check if delete is held for 2 seconds
                    if delete_start_time is None:
                        # Start the delete timer when the delete gesture is detected
                        delete_start_time = time.time()
                    elif time.time() - delete_start_time >= 2:
                        # After 2 seconds, delete the last character
                        if sign_string:
                            sign_string = sign_string[:-1]  # Remove the last letter
                        delete_triggered = True  # Mark delete as completed
                        delete_start_time = None  # Reset delete timer

                else:
                    predicted_character = labels_dict.get(predicted_index, "Invalid")

                # Don't add "Delete" to the sign string, it's only for triggering the delete action
                if predicted_character != "Delete":
                    # Check if the predicted letter is the same as the current letter
                    if predicted_character == current_letter:
                        if time.time() - letter_start_time > letter_confirmation_time:
                            sign_string += current_letter
                            current_letter = None  # Reset the current letter after confirming
                    else:
                        current_letter = predicted_character
                        letter_start_time = time.time()

                # Draw the prediction on the frame
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Convert frame to byte format and yield
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_chatbot_response', methods=['POST'])
def get_chatbot_response():
    global sign_string
    # Process the accumulated sign_string when "Stop" button is pressed
    if sign_string:
        # Call the chatbot to get a response
        response = genai.GenerativeModel(model_name="gemini-1.5-flash").generate_content(sign_string.strip())
        sign_string = ""  # Clear the sign_string after processing
        return jsonify({"response": response.text, "sign_string": sign_string})
    else:
        return jsonify({"response": "No sign detected yet.", "sign_string": sign_string})


email_sent = False

@app.route('/get_current_sign', methods=['GET'])
def get_current_sign():
    global sign_string, email_sent

    if "HHH" in sign_string and not email_sent:
        send_violation_email()
        email_sent = True  # ✅ Set flag to prevent multiple emails
   
    return jsonify({"sign_string": sign_string})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


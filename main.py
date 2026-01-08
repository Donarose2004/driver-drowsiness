# main.py

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import eventlet
import threading

# Monkey patch for eventlet (needed for SocketIO async)
eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app)

# --- Global State ---
state_lock = threading.Lock()
alert_active = False
drowsiness_counter = 0
YAWN_COUNT = 0   # Now global, so it can be reset by socket events

# --- Configuration ---
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load Eye State Model
try:
    eye_interpreter = tf.lite.Interpreter(model_path='eye_state_model.tflite')
    eye_interpreter.allocate_tensors()
    eye_input_details = eye_interpreter.get_input_details()
    eye_output_details = eye_interpreter.get_output_details()
except Exception as e:
    print(f"Error loading eye_state_model.tflite: {e}")
    exit()

# Load Yawn Model
try:
    yawn_interpreter = tf.lite.Interpreter(model_path='yawn_model.tflite')
    yawn_interpreter.allocate_tensors()
    yawn_input_details = yawn_interpreter.get_input_details()
    yawn_output_details = yawn_interpreter.get_output_details()
except Exception as e:
    print(f"Error loading yawn_model.tflite: {e}")
    exit()

_, IMG_HEIGHT, IMG_WIDTH, _ = eye_input_details[0]['shape']


def detect_drowsiness():
    """
    Core detection loop: processes frames, detects drowsiness,
    and streams to Flask while updating SocketIO clients.
    """
    global alert_active, drowsiness_counter, YAWN_COUNT

    # Thresholds
    DROWSINESS_THRESHOLD = 15   # frames before alert triggers
    YAWN_CONFIDENCE_THRESHOLD = 0.95
    EYE_CONFIDENCE_THRESHOLD = 0.80

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        frame_status = 'AWAKE'
        current_eye_status = 'N/A'

        if len(faces) == 0:
            frame_status = 'UNCERTAIN'
            current_eye_status = 'Face Not Detected'

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]

            # --- Yawn Detection ---
            face_for_yawn = cv2.resize(roi_gray, (IMG_WIDTH, IMG_HEIGHT))
            face_for_yawn = np.expand_dims(np.expand_dims(face_for_yawn, axis=-1), axis=0) / 255.0
            yawn_interpreter.set_tensor(yawn_input_details[0]['index'], face_for_yawn.astype(np.float32))
            yawn_interpreter.invoke()
            yawn_pred = yawn_interpreter.get_tensor(yawn_output_details[0]['index'])[0][0]

            if yawn_pred > YAWN_CONFIDENCE_THRESHOLD:
                frame_status = 'DROWSY'
                YAWN_COUNT += 1  # Increment only when strong yawn detected

            # --- Eye Detection ---
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=6)
            if len(eyes) == 0:
                current_eye_status = 'Not Detected'
            else:
                current_eye_status = 'Open'
                for (ex, ey, ew, eh) in eyes:
                    if ey > h / 2:  # Ignore non-eye detections
                        continue
                    eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                    final_image = cv2.resize(eye_roi, (IMG_WIDTH, IMG_HEIGHT))
                    final_image = np.expand_dims(np.expand_dims(final_image, axis=-1), axis=0) / 255.0

                    eye_interpreter.set_tensor(eye_input_details[0]['index'], final_image.astype(np.float32))
                    eye_interpreter.invoke()
                    eye_pred = eye_interpreter.get_tensor(eye_output_details[0]['index'])[0][0]

                    if eye_pred < (1 - EYE_CONFIDENCE_THRESHOLD):  # closed
                        current_eye_status = 'Closed'
                        frame_status = 'DROWSY'
                        break

        # --- State Update ---
        with state_lock:
            if frame_status == 'DROWSY':
                drowsiness_counter += 1
            elif frame_status == 'AWAKE':
                drowsiness_counter = max(0, drowsiness_counter - 2)  # Faster recovery
            # UNCERTAIN does not reset counter, prevents flickering

            if drowsiness_counter > DROWSINESS_THRESHOLD and not alert_active:
                alert_active = True
                socketio.emit('drowsiness_alert', {'alert': True})

            current_drowsiness_level = min(100, (drowsiness_counter / max(1, DROWSINESS_THRESHOLD)) * 100)

        # --- Send Updates to Client ---
        socketio.emit('update_status', {
            'yawn_count': YAWN_COUNT,
            'eye_status': current_eye_status,
            'drowsy_level': current_drowsiness_level
        })

        # --- Video Streaming ---
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if flag:
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

        socketio.sleep(0.01)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(detect_drowsiness(), mimetype='multipart/x-mixed-replace; boundary=frame')


# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('stop_alert_from_client')
def handle_stop_alert():
    """
    Resets the alert state when the user clicks "I'm Awake!" button.
    """
    global alert_active, drowsiness_counter, YAWN_COUNT
    with state_lock:
        if alert_active:
            print("Alert reset by client.")
            alert_active = False
            drowsiness_counter = 0
            YAWN_COUNT = 0  # Reset yawn counter as requested


if __name__ == '__main__':
    socketio.run(app, debug=True, host='127.0.0.1', port=5000)

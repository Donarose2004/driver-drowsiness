import cv2
import numpy as np
import tensorflow as tf
import threading
import time

# --- Threaded Webcam Class ---
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped: return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# --- Configuration & Models ---
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

eye_interpreter = tf.lite.Interpreter(model_path='eye_state_model.tflite')
eye_interpreter.allocate_tensors()
eye_input_details = eye_interpreter.get_input_details()
eye_output_details = eye_interpreter.get_output_details()

yawn_interpreter = tf.lite.Interpreter(model_path='yawn_model.tflite')
yawn_interpreter.allocate_tensors()
yawn_input_details = yawn_interpreter.get_input_details()
yawn_output_details = yawn_interpreter.get_output_details()

_, IMG_HEIGHT, IMG_WIDTH, _ = eye_input_details[0]['shape']

# --- Parameters ---
CLOSED_EYE_CLASS = 0 
YAWN_CLASS = 1 
DROWSINESS_COUNTER = 0
DROWSINESS_THRESHOLD = 20 
YAWN_CONFIDENCE_THRESHOLD = 0.95
EYE_CONFIDENCE_THRESHOLD = 0.80 
ALERT_DISPLAY_DURATION = 90 
alert_display_counter = 0

# --- NEW: Frame Skip Parameters ---
frame_count = 0
SKIP_FRAMES = 3  # Process AI every 3rd frame

# --- Start Stream ---
vs = WebcamVideoStream(src=0).start()
time.sleep(1.0) 

while True:
    frame = vs.read()
    if frame is None: break
    
    frame_count += 1
    
    # We always reset the flag, but we only set it to True during AI frames
    is_drowsy_frame = False

    # --- ONLY RUN AI ON SPECIFIC FRAMES ---
    if frame_count % SKIP_FRAMES == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Yawn Detection
            face_for_yawn = cv2.resize(roi_gray, (IMG_WIDTH, IMG_HEIGHT))
            face_for_yawn = np.expand_dims(face_for_yawn, axis=-1)
            face_for_yawn = np.expand_dims(face_for_yawn, axis=0)
            face_for_yawn = (face_for_yawn / 255.0).astype(np.float32)

            yawn_interpreter.set_tensor(yawn_input_details[0]['index'], face_for_yawn)
            yawn_interpreter.invoke()
            yawn_pred = yawn_interpreter.get_tensor(yawn_output_details[0]['index'])[0][0]

            if (YAWN_CLASS == 1 and yawn_pred > YAWN_CONFIDENCE_THRESHOLD) or \
               (YAWN_CLASS == 0 and yawn_pred < (1 - YAWN_CONFIDENCE_THRESHOLD)):
                is_drowsy_frame = True

            # Eye Detection
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
            if len(eyes) == 0:
                is_drowsy_frame = True
            
            for (ex, ey, ew, eh) in eyes:
                if ey > h / 2: continue
                eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                final_image = cv2.resize(eye_roi, (IMG_WIDTH, IMG_HEIGHT))
                final_image = np.expand_dims(final_image, axis=-1)
                final_image = np.expand_dims(final_image, axis=0)
                final_image = (final_image / 255.0).astype(np.float32)

                eye_interpreter.set_tensor(eye_input_details[0]['index'], final_image)
                eye_interpreter.invoke()
                eye_pred = eye_interpreter.get_tensor(eye_output_details[0]['index'])[0][0]

                if (CLOSED_EYE_CLASS == 0 and eye_pred < (1 - EYE_CONFIDENCE_THRESHOLD)) or \
                   (CLOSED_EYE_CLASS == 1 and eye_pred > EYE_CONFIDENCE_THRESHOLD):
                    is_drowsy_frame = True

        # Update Counter Logic only on AI frames
        if is_drowsy_frame:
            DROWSINESS_COUNTER += 1
        else:
            DROWSINESS_COUNTER = max(0, DROWSINESS_COUNTER - 1)
            
        if DROWSINESS_COUNTER > DROWSINESS_THRESHOLD:
            alert_display_counter = ALERT_DISPLAY_DURATION
            DROWSINESS_COUNTER = 0 

    # --- ALERT DISPLAY (Always visible) ---
    if alert_display_counter > 0:
        cv2.putText(frame, "DROWSINESS ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        alert_display_counter -= 1

    cv2.imshow('Drowsiness Detection (Optimized)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

vs.stop()
cv2.destroyAllWindows()
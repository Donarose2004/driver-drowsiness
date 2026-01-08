import cv2
import numpy as np
import tensorflow as tf

# --- Configuration ---
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# --- Load Eye State Model ---
try:
    eye_interpreter = tf.lite.Interpreter(model_path='eye_state_model.tflite')
    eye_interpreter.allocate_tensors()
    eye_input_details = eye_interpreter.get_input_details()
    eye_output_details = eye_interpreter.get_output_details()
except Exception as e:
    print(f"Error loading eye_state_model.tflite: {e}")
    exit()

# --- Load Yawn Model ---
try:
    yawn_interpreter = tf.lite.Interpreter(model_path='yawn_model.tflite')
    yawn_interpreter.allocate_tensors()
    yawn_input_details = yawn_interpreter.get_input_details()
    yawn_output_details = yawn_interpreter.get_output_details()
except Exception as e:
    print(f"Error loading yawn_model.tflite: {e}")
    exit()


# Get image size from one of the models (they should be the same)
_, IMG_HEIGHT, IMG_WIDTH, _ = eye_input_details[0]['shape']

# --- IMPORTANT: CONFIGURE YOUR CLASS INDICES HERE ---
# Look at the notes you took during training.
# Example: If your training output for the eye model was {'closed': 0, 'open': 1},
# then you must set CLOSED_EYE_CLASS = 0.
CLOSED_EYE_CLASS = 0  # <-- CHANGE THIS VALUE BASED ON YOUR TRAINING OUTPUT

# Example: If your training output for the yawn model was {'no_yawn': 0, 'yawn': 1},
# then you must set YAWN_CLASS = 1.
YAWN_CLASS = 1  # <-- CHANGE THIS VALUE BASED ON YOUR TRAINING OUTPUT
# ----------------------------------------------------

# --- Alert & Confidence Configuration ---
DROWSINESS_COUNTER = 0
DROWSINESS_THRESHOLD = 20      # Frames to wait before triggering alert
YAWN_CONFIDENCE_THRESHOLD = 0.95 # Stricter threshold for yawns
EYE_CONFIDENCE_THRESHOLD = 0.80  # A more lenient threshold for eyes

# Alert Timer Configuration
ALERT_DISPLAY_DURATION = 90  # Show alert for 90 frames (approx. 3 seconds at 30fps)
alert_display_counter = 0
# --- --- --- --- --- ---

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # This flag is reset for every frame
    is_drowsy_frame = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # --- YAWN DETECTION ---
        face_for_yawn = cv2.resize(roi_gray, (IMG_WIDTH, IMG_HEIGHT))
        face_for_yawn = np.expand_dims(face_for_yawn, axis=-1)
        face_for_yawn = np.expand_dims(face_for_yawn, axis=0)
        face_for_yawn = face_for_yawn / 255.0

        yawn_interpreter.set_tensor(yawn_input_details[0]['index'], face_for_yawn.astype(np.float32))
        yawn_interpreter.invoke()
        yawn_pred = yawn_interpreter.get_tensor(yawn_output_details[0]['index'])[0][0]

        yawn_status = 'No Yawn'
        # Correctly check prediction for the YAWN model
        if (YAWN_CLASS == 1 and yawn_pred > YAWN_CONFIDENCE_THRESHOLD) or \
           (YAWN_CLASS == 0 and yawn_pred < (1 - YAWN_CONFIDENCE_THRESHOLD)):
            yawn_status = 'Yawn'
            is_drowsy_frame = True
            
        cv2.putText(frame, f"Yawn: {yawn_status}", (x, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # --- EYE DETECTION ---
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        
        if len(eyes) == 0:
            is_drowsy_frame = True
            cv2.putText(frame, "Eyes Not Detected", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        for (ex, ey, ew, eh) in eyes:
            # All of the following logic is now correctly indented inside this loop
            if ey > h / 2:
                continue

            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            
            final_image = cv2.resize(eye_roi, (IMG_WIDTH, IMG_HEIGHT))
            final_image = np.expand_dims(final_image, axis=-1)
            final_image = np.expand_dims(final_image, axis=0)
            final_image = final_image / 255.0

            eye_interpreter.set_tensor(eye_input_details[0]['index'], final_image.astype(np.float32))
            eye_interpreter.invoke()
            eye_pred = eye_interpreter.get_tensor(eye_output_details[0]['index'])[0][0]

            eye_status = 'Open'
            # Correctly check prediction for the EYE model
            if (CLOSED_EYE_CLASS == 0 and eye_pred < (1 - EYE_CONFIDENCE_THRESHOLD)) or \
               (CLOSED_EYE_CLASS == 1 and eye_pred > EYE_CONFIDENCE_THRESHOLD):
                eye_status = 'Closed'
                is_drowsy_frame = True
            
            cv2.putText(roi_color, eye_status, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # --- Drowsiness Counter & Alert Logic ---
    if is_drowsy_frame:
        DROWSINESS_COUNTER += 1
    else:
        DROWSINESS_COUNTER = max(0, DROWSINESS_COUNTER - 1)
        
    # If the drowsiness threshold is crossed, trigger the alert timer
    if DROWSINESS_COUNTER > DROWSINESS_THRESHOLD:
        alert_display_counter = ALERT_DISPLAY_DURATION
        DROWSINESS_COUNTER = 0 # Reset the detection counter

    # If the alert timer is active, display the alert and count down
    if alert_display_counter > 0:
        cv2.putText(frame, "DROWSINESS ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        alert_display_counter -= 1

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
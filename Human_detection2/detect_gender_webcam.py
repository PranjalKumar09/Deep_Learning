import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# Load model
model = None
try:
    model = load_model('Human_detection2/model.h5', compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Open webcam
webcam = cv2.VideoCapture(0)

classes = ['man', 'woman']

# Loop through frames
while webcam.isOpened():
    # Read frame from webcam
    status, frame = webcam.read()

    if not status:
        break

    # Apply face detection
    face, confidence = cv.detect_face(frame)

    # Loop through detected faces
    for idx, f in enumerate(face):
        # Get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # Preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float32") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Ensure model is defined before predicting
        if model is not None:
            # Apply gender detection on face
            conf = model.predict(face_crop)[0]  # model.predict returns a 1D array

            # Get label with max accuracy
            idx = np.argmax(conf)
            label = classes[idx]

            label = "{}: {:.2f}%".format(label, conf[idx] * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # Write label and confidence above face rectangle
            cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

    # Display output
    cv2.imshow("Gender Detection", frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()

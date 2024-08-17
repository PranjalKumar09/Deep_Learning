import cv2
import csv
import numpy as np
from datetime import datetime
import pandas as pd
from geopy.geocoders import Nominatim


# wget https://pjreddie.com/media/files/yolov3.weights
# 

# Placeholder function for person detection and gender classification


def capture_frame(camera_id):
    # Open a connection to the camera
    cap = cv2.VideoCapture(camera_id)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera with ID {camera_id}")
        return None

    # Read a frame from the camera
    ret, frame = cap.read()

    # Release the camera
    cap.release()

    # Check if the frame was captured successfully
    if not ret:
        print(f"Error: Could not read frame from camera with ID {camera_id}")
        return None

    return frame
def load_model():
    # Replace with actual model loading code
    model = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')  # Placeholder
    return model
model = load_model()

def preprocess_frame(frame):
    input_size = (300, 300)
    resized_frame = cv2.resize(frame, input_size)
    preprocessed_frame = resized_frame.astype(np.float32) / 255.0
    return preprocessed_frame
def load_gender_model():
    gender_net = cv2.dnn.readNetFromCaffe(
        'deploy_gender.prototxt', 
        'gender_net.caffemodel'
    )
    return gender_net

gender_model = load_gender_model()

def classify_gender(face_img, gender_model):
    blob = cv2.dnn.blobFromImage(face_img, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False, crop=False)
    gender_model.setInput(blob)
    gender_preds = gender_model.forward()
    gender = "Male" if gender_preds[0].argmax() == 0 else "Female"
    return gender

def model_1(input_frame, model, gender_model):
    height, width = input_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(input_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    model.setInput(blob)

    layer_names = model.getLayerNames()
    unconnected_out_layers = model.getUnconnectedOutLayers()

    if isinstance(unconnected_out_layers, np.ndarray):
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
    else:
        output_layers = [layer_names[unconnected_out_layers - 1]]

    detections = model.forward(output_layers)

    detected_persons = []

    for detection in detections:
        for object in detection:
            scores = object[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(object[0] * width)
                center_y = int(object[1] * height)
                w = int(object[2] * width)
                h = int(object[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2

                # Extract face/person region for gender classification
                face_img = input_frame[y:y + h, x:x + w]

                # Classify gender
                gender = classify_gender(face_img, gender_model)

                detected_persons.append({
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "confidence": confidence,
                    "class_id": class_id,
                    "Gender": gender
                })

                cv2.rectangle(input_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(input_frame, f"ID: {class_id} Gender: {gender}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return detected_persons

def assess_proximity_and_risk(detected_persons):
    proximity_women = "Low"
    closeness_others = "Medium"
    safety_risk = "Low"

    women_count = sum(1 for person in detected_persons if person["Gender"] == "Female")
    men_count = sum(1 for person in detected_persons if person["Gender"] == "Male")

    if women_count == 1 and men_count > 3:
        safety_risk = "High"
        proximity_women = "High"
        closeness_others = "High"

    return safety_risk, proximity_women, closeness_others

def update_data_csv(data, filename='safety_data.csv'):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    
def update_csv(camera_id, latitude, longitude, detected_persons):
    # Your existing code to update the CSV file

    # Example CSV update logic:
    for person in detected_persons:
        # Extract information from detected persons
        # Add any additional processing needed before writing to CSV
        row = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Camera_ID': camera_id,
            'Latitude': latitude,
            'Longitude': longitude,
            'Person_ID': person.get('Person_ID', 'Unknown'),
            'Gender': person.get('Gender', 'Unknown'),
            'Age': person.get('Age', 'Unknown'),
            'Confidence': person.get('Confidence', 0),
            # Additional fields can be added here
        }

        # Append the row to your CSV file
        with open('output.csv', 'a', newline='') as csvfile:
            fieldnames = list(row.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)


def process_live_camera(camera_id, latitude, longitude):
    cap = cv2.VideoCapture(0)  # Open the default camera

    frame_count = 0
    process_every_n_frames = 5  # Adjust this to process every nth frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every nth frame
        if frame_count % process_every_n_frames == 0:
            detected_persons = model_1(frame, model)
            update_csv(camera_id, latitude, longitude, detected_persons)

        # Display the frame in a window
        cv2.imshow('Camera Feed', frame)

        frame_count += 1

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Example usage
camera_id = 0  # Default camera
latitude = 28.6139
longitude = 77.2090
process_live_camera(camera_id, latitude, longitude)

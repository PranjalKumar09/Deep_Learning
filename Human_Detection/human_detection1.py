import csv
import datetime
import cv2
import tensorflow as tf
import tensorflow_hub as hub


model_url = 'https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1'
detection_model = hub.load(model_url)



# Initialize CSV file and writer
csv_file_path = 'detection_data.csv'
fieldnames = ['Timestamp', 'Camera_ID', 'Box', 'Gender', 'Latitude', 'Longitude']

def initialize_csv():
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

def append_to_csv(timestamp, camera_id, box, gender, latitude, longitude):
    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({
            'Timestamp': timestamp,
            'Camera_ID': camera_id,
            'Box': box,
            'Gender': gender,
            'Latitude': latitude,
            'Longitude': longitude
        })

initialize_csv()  # Call this once to set up the file and header

def process_frame(frame, camera_id, latitude, longitude):
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(input_frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Perform detection
    detections = detection_model(input_tensor)
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy()

    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            box = boxes[i]
            ymin, xmin, ymax, xmax = box
            ymin, xmin, ymax, xmax = int(ymin * frame.shape[0]), int(xmin * frame.shape[1]), int(ymax * frame.shape[0]), int(xmax * frame.shape[1])
            person_img = frame[ymin:ymax, xmin:xmax]
            person_img = cv2.resize(person_img, (224, 224))

            # Gender Classification (assuming you have a gender model)
            gender_model = tf.keras.models.load_model('path_to_gender_model')
            person_img = person_img / 255.0
            person_img = person_img[tf.newaxis, ...]
            gender_prediction = gender_model.predict(person_img)
            gender = 'Male' if gender_prediction > 0.5 else 'Female'

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, gender, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            append_to_csv(timestamp, camera_id, str(box), gender, latitude, longitude)

    return frame

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for default camera
camera_id = 'Camera_1'  # Example camera ID

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    latitude = 0.0  # Replace with actual latitude
    longitude = 0.0  # Replace with actual longitude

    processed_frame = process_frame(frame, camera_id, latitude, longitude)
    cv2.imshow('Real-Time Human Detection and Gender Classification', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

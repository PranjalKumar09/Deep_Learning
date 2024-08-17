import cv2
import numpy as np

def load_model(config_path, weights_path):
    """
    Load the YOLO model from the configuration and weights files.

    Parameters:
    config_path (str): Path to the YOLO configuration file.
    weights_path (str): Path to the YOLO weights file.

    Returns:
    Net: A Net object representing the loaded YOLO model.
    List[str]: The names of the output layers for the YOLO model.
    """
    model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    layer_names = model.getLayerNames()

    # Some versions of OpenCV return layer indices as a single integer, so we need to adjust the access accordingly
    if isinstance(model.getUnconnectedOutLayers()[0], list) or isinstance(model.getUnconnectedOutLayers()[0], np.ndarray):
        output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    else:
        output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]

    return model, output_layers


def detect_persons(frame, model, output_layers, confidence_threshold=0.5, nms_threshold=0.4):
    """
    Detect persons in a given video frame using the YOLO model.

    Parameters:
    frame (ndarray): The video frame in which to detect persons.
    model (Net): The loaded YOLO model.
    output_layers (list): The names of the output layers for the YOLO model.
    confidence_threshold (float): The threshold for filtering weak detections.
    nms_threshold (float): The threshold for non-maxima suppression.

    Returns:
    List[dict]: A list of dictionaries containing detection information (e.g., bounding box, confidence, class).
    """
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    model.setInput(blob)
    outputs = model.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Only proceed if the detected object is a person (class_id = 0 for YOLOv3 COCO dataset)
            if confidence > confidence_threshold and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to suppress weaker, overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    detections = []
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        detections.append({
            'box': (x, y, w, h),
            'confidence': confidences[i],
            'class_id': class_ids[i]
        })

    return detections

if __name__ == "__main__":
    config_path = 'yolov3.cfg'
    weights_path = 'yolov3.weights'
    model, output_layers = load_model(config_path, weights_path)

    cap = cv2.VideoCapture(0)  # Use your camera ID or video file path here

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_persons(frame, model, output_layers)

        for detection in detections:
            x, y, w, h = detection['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Person: {detection['confidence']:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Person Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

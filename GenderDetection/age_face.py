import cv2
from deepface import DeepFace

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Analyze the frame for age and gender
    try:
        result = DeepFace.analyze(frame, actions=['age', 'gender'])
        age = result[0]['age']
        gender = result[0]['gender']
        
        # Display the result on the frame
        cv2.putText(frame, f"Gender: {gender}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {age}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print(f"Error: {e}")

    # Show the frame with annotations
    cv2.imshow('Gender and Age Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()

import cv2

def get_camera_feed(camera_id=0):
    """
    Initialize the camera feed.

    Parameters:
    camera_id (int): The ID of the camera to use. Default is 0 for the primary camera.

    Returns:
    VideoCapture: A VideoCapture object to read frames from the camera.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    return cap

def release_camera(cap):
    """
    Release the camera and close any open windows.

    Parameters:
    cap (VideoCapture): The VideoCapture object to release.
    """
    cap.release()
    cv2.destroyAllWindows()

def show_camera_feed(cap):
    """
    Display the camera feed in a window.

    Parameters:
    cap (VideoCapture): The VideoCapture object to read frames from the camera.
    """
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        cv2.imshow('Camera Feed', frame)

        # Press 'q' to quit the camera feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    # Initialize the camera feed
    cap = get_camera_feed()

    # If the camera feed is successfully initialized, show the feed
    if cap:
        show_camera_feed(cap)

    # Release the camera and close windows
    release_camera(cap)

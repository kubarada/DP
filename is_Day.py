import cv2
import numpy as np


def is_daytime(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the mean brightness of the grayscale image
    mean_brightness = np.mean(gray)

    # You can adjust this threshold as needed
    threshold = 100  # You may need to tune this value based on your specific use case

    # Determine if it's daytime based on the mean brightness
    if mean_brightness > threshold:
        return True
    else:
        return False


# Initialize the video capture
cap = cv2.VideoCapture(0)  # You can specify a different video source or file path if needed

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Check if it's daytime
    if is_daytime(frame):
        cv2.putText(frame, "Daytime", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Nighttime", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Day-Night Recognizer', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

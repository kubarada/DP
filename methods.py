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

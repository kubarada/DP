import cv2
import numpy as np


def is_daytime(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mean_brightness = np.mean(gray)

    threshold = 100

    if mean_brightness > threshold:
        return True
    else:
        return False

def calculate_bbox_center(bbox):
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return center_x, center_y

def draw_trajectory(image, coordinates):
    for i in range(1, len(coordinates)):
        cv2.line(image, coordinates[i - 1], coordinates[i], (0, 0, 255), thickness=2)

    return image
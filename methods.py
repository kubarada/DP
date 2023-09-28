import cv2
import numpy as np
import json


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

def load_points_from_file(file_path):
    points = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip().strip('()')
                x, y = map(int, line.split(','))
                points.append((x, y))

        return points

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None

def extract_bbox(json_file, class_name):
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    category_id = None
    for category in coco_data['categories']:
        if category['name'] == class_name:
            category_id = category['id']
            break

    if category_id is None:
        raise ValueError(f"Class name '{class_name}' not found in the COCO dataset.")

    bbox_positions = []

    for annotation in coco_data['annotations']:
        if annotation['category_id'] == category_id:
            bbox = annotation['bbox']
            x, y, width, height = bbox
            x2, y2 = x + width, y + height
            bbox_positions.append((x, y, x2, y2))

    return bbox_positions

def list_to_file(my_list, file_path):
    try:
        with open(file_path, "w") as file:
            for item in my_list:
                file.write(str(item) + "\n")
        print(f"List has been written to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


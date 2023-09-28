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

def calculate_iou(box1, box2):
    # Extract coordinates from the bounding box tuples
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Calculate the intersection coordinates
    x_intersection = max(0, min(x2, x4) - max(x1, x3))
    y_intersection = max(0, min(y2, y4) - max(y1, y3))

    # Calculate the area of intersection and union
    intersection_area = x_intersection * y_intersection
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - intersection_area

    # Calculate the Intersection over Union
    iou = intersection_area / union_area

    return iou

def calculate_final_iou(iou_list):
    return sum(iou_list)/len(iou_list)

def load_bounding_boxes_from_file(file_path):
    bounding_boxes = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Remove parentheses and split the line into individual values
                values = line.strip('()\n').split(', ')

                # Convert the values to floats and create a tuple
                bbox_tuple = tuple(map(float, values))

                # Append the tuple to the list of bounding_boxes
                bounding_boxes.append(bbox_tuple)
    except FileNotFoundError:
        print(f"File not found: {file_path}")

    return bounding_boxes
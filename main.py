import methods
import cv2

FRAME_PATH = 'data/input/frame_027673.PNG'
POINTS_PATH = 'data/input/trajectory.txt'
OUTPUT = 'data/output/out1.png'
JSON_PATH = 'data/input/instances_default.json'
INPUT_BBOX1 = 'data/input/bbox_ground_truth.txt'
INPUT_BBOX2 = 'data/input/bbox_siamese_rpn.txt'
INPUT_BBOX3 = 'data/input/bbox_faster_rcnn.txt'




iou_list = []
bbox_siamese_rpn = methods.load_bounding_boxes_from_file(INPUT_BBOX2)
bbox_ground_truth = methods.load_bounding_boxes_from_file(INPUT_BBOX1)

for i in range(len(bbox_ground_truth)):
    iou_list.append(methods.calculate_iou(bbox_ground_truth[i], bbox_siamese_rpn[i]))

print('Average IoU over all bboxes = ', 100 * methods.calculate_final_iou(iou_list), '%')
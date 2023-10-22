import methods
import cv2

FRAME_PATH = 'data/input/frame_027673.PNG'
POINTS_PATH = 'data/input/trajectory.txt'
OUTPUT = 'data/output/out1.png'
JSON_PATH = 'data/input/instances_default.json'
INPUT_BBOX1 = 'data/input/bbox_ground_truth.txt'
INPUT_BBOX2 = 'data/input/bbox_siamese_rpn.txt'
INPUT_BBOX3 = 'data/input/bbox_faster_rcnn.txt'
INPUT_BBOX4 = 'data/input/bbox_deepsort.txt'
INPUT_BBOX5 = 'data/input/bbox_kcf.txt'
INPUT_BBOX6 = 'data/input/bbox_mosse.txt'
INPUT_BBOX7 = 'data/input/bbox_ada_boost.txt'
INPUT_BBOX8 = 'data/input/bbox_tld.txt'
INPUT_BBOX9 = 'data/input/bbox_mean_shift.txt'



iou_list = []
bbox_siamese_rpn = methods.load_bounding_boxes_from_file(INPUT_BBOX1)
bbox_ground_truth = methods.load_bounding_boxes_from_file(INPUT_BBOX5)

for i in range(len(bbox_ground_truth)):
    iou_list.append(methods.calculate_iou(bbox_ground_truth[i], bbox_siamese_rpn[i]))

iou_list.append(0)
iou_list.append(0)
iou_list.append(0)
iou_list.append(0)

print('Average IoU over all bboxes = ', 100 * methods.calculate_final_iou(iou_list), '%')

methods.combine_images('data/input/TLDTracking_screenshot_22.10.2023.png',
                       'data/input/TLDTracking_screenshot_22.10.2023a.png', 'data/output/combined_image.jpg')
methods.combine_images('data/input/Tracking_screenshot_22.10.2023.png',
                       'data/input/Tracking_screenshot_22.10.2023a.png', 'data/output/combined_image1.jpg')
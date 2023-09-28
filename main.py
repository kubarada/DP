import methods
import cv2

FRAME_PATH = 'data/input/frame_027673.PNG'
POINTS_PATH = 'data/input/trajectory.txt'
OUTPUT = 'data/output/out1.png'
JSON_PATH = 'data/input/instances_default.json'
OUTPUT_BBOX = 'data/output/bbox.txt'

frame = cv2.imread(FRAME_PATH)

coordinates = methods.load_points_from_file(POINTS_PATH)

frame = methods.draw_trajectory(frame, coordinates)

if methods.is_daytime(frame):
     cv2.putText(frame, "is_day=1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
else:
    cv2.putText(frame, "is_day=0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

width, length = frame.shape[:2]
cv2.putText(frame, "img_size = (" + str(width) + ', ' + str(length) + ')', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imwrite(OUTPUT, frame)

bbox_list = methods.extract_bbox(JSON_PATH, 'pig_shape')[1::2]
methods.list_to_file(bbox_list, OUTPUT_BBOX)
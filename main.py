import methods
import cv2

FRAME_PATH = 'data/frame_027673.png'
POINTS_PATH = 'data/trajectory.txt'
OUTPUT = 'data/output/out1.png'

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

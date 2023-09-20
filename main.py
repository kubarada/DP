import methods
import cv2

INPUT = 'data/frame_027673.png'
OUTPUT = 'data/output/out1.png'

frame = cv2.imread(INPUT)

if methods.is_daytime(frame):
     cv2.putText(frame, "is_day=1", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
else:
    cv2.putText(frame, "is_day=0", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imwrite(OUTPUT, frame)

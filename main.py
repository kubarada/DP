import methods
import cv2

INPUT = 'data/frame_027673.png'
OUTPUT = 'data/output/out1.png'

coordinates = [(843, 477), (841, 480), (842, 481), (843, 483), (845, 485), (846, 488), (845, 490), (844, 492), (845, 495), (845, 498), (844, 500), (843, 503), (843, 506), (842, 508), (842, 511), (841, 514), (840, 517), (840, 520), (839, 523), (838, 526)]

frame = cv2.imread(INPUT)

frame = methods.draw_trajectory(frame, coordinates)

if methods.is_daytime(frame):
     cv2.putText(frame, "is_day=1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
else:
    cv2.putText(frame, "is_day=0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

width, length = frame.shape[:2]
cv2.putText(frame, "img_size = (" + str(width) + ', ' + str(length) + ')', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imwrite(OUTPUT, frame)

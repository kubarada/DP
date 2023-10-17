import cv2
import methods
OUTPUT_BBOX = 'data/output/bbox_mean_shift.txt'


# Open a video capture object or load a video file
cap = cv2.VideoCapture('data/input/1.mp4')

# Read the first frame
ret, frame = cap.read()

# Define the initial region of interest (ROI)
x, y, w, h = 750, 407, (1041-750), (605-407)  # Adjust these coordinates as needed
roi = frame[y:y+h, x:x+w]

# Setup the termination criteria for Mean-Shift
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
bboxes = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to the HSV color space (usually better for tracking)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calculate the histogram of the ROI in the first frame
    roi_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Calculate the back projection
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Apply the Mean-Shift algorithm to track the object
    ret, track_window = cv2.meanShift(dst, (x, y, w, h), term_crit)

    # Update the ROI coordinates
    x, y, w, h = track_window
    bboxes.append([track_window[0], track_window[1], track_window[0]+track_window[2], track_window[1]+track_window[3]])

    # Draw the tracked object on the frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Mean-Shift Tracking', frame)

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

methods.list_to_file(bboxes, OUTPUT_BBOX)

cap.release()
cv2.destroyAllWindows()

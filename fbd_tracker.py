import cv2

# Create a video capture object
cap = cv2.VideoCapture('video_path.mp4')  # Replace 'video_path.mp4' with your video file

# Read the first frame
ret, frame = cap.read()

if not ret:
    raise ValueError("Cannot read the first frame. Check the video file.")

# Define a region of interest (ROI) containing the object you want to track
x, y, width, height = 100, 100, 50, 50  # Modify these coordinates to define your ROI
roi = frame[y:y+height, x:x+width]

# Convert the ROI to grayscale
roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Create a list of feature points in the ROI
p0 = cv2.goodFeaturesToTrack(roi_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

# Create a mask image for drawing purposes
mask = np.zeros_like(roi)

while True:
    # Read a new frame from the video
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if we reach the end of the video

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade tracker
    p1, st, err = cv2.calcOpticalFlowPyrLK(roi_gray, frame_gray, p0, None)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw tracking lines on the frame
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    # Add the mask to the frame
    result = cv2.add(frame, mask)

    # Update the previous frame and points
    roi_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Display the result
    cv2.imshow('Feature-Based Tracking', result)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

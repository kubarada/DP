import cv2
import os

# Create a video capture object
cap = cv2.VideoCapture('data/input/1.mp4')  # Replace 'video_path.mp4' with your video file

# Define the output directory to save frames
output_dir = 'data/output/boost'
os.makedirs(output_dir, exist_ok=True)

# Read the first frame
ret, frame = cap.read()

if not ret:
    raise ValueError("Cannot read the first frame. Check the video file.")

x1, y1, x2, y2 = 750, 407, 1041, 605

# Calculate the width and height of the bounding box
width = x2 - x1
height = y2 - y1

# Create a Boosting tracker
tracker = cv2.TrackerBoosting_create()  # Note the 'legacy' prefix

# Initialize the tracker with the first frame and bounding box
bbox = (x1, y1, width, height)

# Create a Boosting tracker
tracker = cv2.TrackerBoosting_create()  # Note the 'legacy' prefix

# Initialize the tracker with the first frame and bounding box
success = tracker.init(frame, bbox)

frame_number = 0  # Initialize frame number for file naming

while True:
    # Read a new frame from the video
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if we reach the end of the video

    # Update the tracker to get the new bounding box
    success, bbox = tracker.update(frame)

    if success:
        # Draw the bounding box on the frame
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

    # Save the frame as an image in the output directory
    frame_number += 1
    output_path = os.path.join(output_dir, f"frame_{frame_number:04d}.png")
    cv2.imwrite(output_path, frame)

# Release the video capture object
cap.release()

import subprocess
import os

frame_folder = 'data/output/boost'
output_video = 'data/output/boost.mp4'

frames = [f for f in os.listdir(frame_folder) if f.endswith('.png')]
frames.sort()

if not frames:
    print("No frames found in the specified folder.")
else:
    frame_pattern = os.path.join(frame_folder, 'frame_%04d.png')

    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', str(20),
        '-i', frame_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video
    ]

    subprocess.run(ffmpeg_cmd)

print(f"Video '{output_video}' created successfully.")
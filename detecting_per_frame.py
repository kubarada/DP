import cv2
import mmcv
from mmdet.apis import init_detector, inference_detector
import methods
import time

# Specify the path to model config and checkpoint file
config_file = 'C:/Users/Jakub/mmdetection/mmdetection/configs/faster_rcnn/test.py'
checkpoint_file = 'D:/Škola/bc_prace/epoch_12_faster_rcnn.pth'
OUTPUT_BBOX = 'data/output/bbox_faster_rcnn.txt'
bbox_detect = []
# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

video_path = 'data/input/1.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

i = 0
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if we have reached the end of the video
    result = inference_detector(model, frame)
    i += 1
    print('Frame: ', i, '/1385')
    bbox_detect.append(tuple(result[0][1][:4]))

end_time = time.time()

elapsed_time = end_time - start_time
cap.release()
print(f"Elapsed time: {elapsed_time:.4f} seconds")
methods.list_to_file(bbox_detect, OUTPUT_BBOX)


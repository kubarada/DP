import cv2
import mmcv
from mmdet.apis import init_detector, inference_detector
import mmcv
import tempfile
from mmtrack.apis import init_model
from mmtrack.apis import inference_sot
import methods
import time


# Specify the path to model config and checkpoint file
config_file = 'C:/Users/Jakub/mmdetection/mmdetection/configs/faster_rcnn/test.py'
checkpoint_file = 'D:/Škola/bc_prace/epoch_12_faster_rcnn.pth'
OUTPUT_BBOX = 'data/output/bbox_siamese_rpn.txt'


model = init_detector(config_file, checkpoint_file, device='cuda:0')
img = 'data/input/1.jpg'
result = inference_detector(model, img)


input_video = 'data/input/1.mp4' # input video
sot_config = 'C:/Users/Jakub/mmtracking/configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py'
sot_checkpoint = 'C:/Users/Jakub/mmtracking/checkpoints/siamese_rpn_r50_fp16_20e_lasot_20220422_181501-ce30fdfd.pth'

sot_model = init_model(sot_config, sot_checkpoint, device='cuda:0')

init_bbox = result[0][1][:4]
imgs = mmcv.VideoReader(input_video)

prog_bar = mmcv.ProgressBar(len(imgs))
out_dir = tempfile.TemporaryDirectory()
out_path = out_dir.name

bbox_track = []
start_time = time.time()

for i, img in enumerate(imgs):
    result = inference_sot(sot_model, img, init_bbox, frame_id=i)
    bbox = result['track_results'][1:5]
    bbox_track.append(tuple(bbox))
    sot_model.show_result(img, result, wait_time=int(1000. / imgs.fps), out_file=f'{out_path}/{i:06d}.jpg')
    prog_bar.update()
end_time = time.time()
elapsed_time = end_time - start_time

output = 'data/output/siamese_rpn_output.mp4'
print(f'\n making the output video at {output} with a FPS of {imgs.fps}')
print(f"Elapsed time: {elapsed_time:.4f} seconds")
mmcv.frames2video(out_path, output, fps=imgs.fps, fourcc='mp4v')
out_dir.cleanup()
methods.list_to_file(bbox_track, OUTPUT_BBOX)
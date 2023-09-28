import cv2
import mmcv
from mmdet.apis import init_detector, inference_detector
import mmcv
import tempfile
from mmtrack.apis import inference_mot, init_model


# Specify the path to model config and checkpoint file
config_file = 'data/configs/faster_rcnn/test.py'
checkpoint_file = 'data/checkpoints/epoch_12_faster_rcnn.pth'

# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Test a single image and show the results
img = '1.jpg'
result = inference_detector(model, img)
print(result[0][1][:4])

from mmtrack.apis import inference_sot
input_video = '1.mp4' # input video
sot_config = 'data/configs/siamese_rpn/siamese_rpn_r50_20e_lasot.py'
sot_checkpoint = 'data/checkpoints/siamese_rpn_r50_fp16_20e_lasot_20220422_181501-ce30fdfd.pth'
# build the model from a config file and a checkpoint file
sot_model = init_model(sot_config, sot_checkpoint, device='cuda:0')
init_bbox = result[0][1][:4]
imgs = mmcv.VideoReader(input_video)
prog_bar = mmcv.ProgressBar(len(imgs))
out_dir = tempfile.TemporaryDirectory()
out_path = out_dir.name
bbox_track = []
bbox_track.append(result[0][1][:4])
for i, img in enumerate(imgs):
    result = inference_sot(sot_model, img, init_bbox, frame_id=i)
    bbox_track.append(result[0][1][:4])
    sot_model.show_result(
            img,
            result,
            wait_time=int(1000. / imgs.fps),
            out_file=f'{out_path}/{i:06d}.jpg')
    prog_bar.update()
output = 'data/output/sot1.mp4'
print(f'\n making the output video at {output} with a FPS of {imgs.fps}')
mmcv.frames2video(out_path, output, fps=imgs.fps, fourcc='mp4v')
out_dir.cleanup()
print(bbox_track)

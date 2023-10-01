import mmcv
import tempfile
from mmtrack.apis import inference_mot, init_model
import methods
import time
mot_config = 'C:/Users/Jakub/mmtracking/configs/mot/deepsort/test.py'
input_video = 'data/input/1.mp4'
OUTPUT_BBOX = 'data/output/bbox_deepsort.txt'
imgs = mmcv.VideoReader(input_video)
# build the model from a config file
mot_model = init_model(mot_config, device='cuda:0')
prog_bar = mmcv.ProgressBar(len(imgs))
out_dir = tempfile.TemporaryDirectory()
out_path = out_dir.name
bbox_track = []
# test and show/save the images
start_time = time.time()

for i, img in enumerate(imgs):
    result = inference_mot(mot_model, img, frame_id=i)
    bbox = result['track_bboxes']
    bbox2 = [tuple(row) for row in bbox]
    bbox = tuple(bbox2[0][1][0:5])
    bbox1 = tuple(bbox2[0][0][0:5])
    if bbox[0] == 1:
        bbox_track.append(tuple(bbox[1:5]))
    else:
        bbox_track.append(tuple(bbox1[1:5]))

    mot_model.show_result(
            img,
            result,
            show=False,
            wait_time=int(1000. / imgs.fps),
            out_file=f'{out_path}/{i:06d}.jpg')
    prog_bar.update()

end_time = time.time()
elapsed_time = end_time - start_time
output = 'data/output/deepsort_output.mp4'
print(f'\n making the output video at {output} with a FPS of {imgs.fps}')
print(f"Elapsed time: {elapsed_time:.4f} seconds")
out_dir.cleanup()
methods.list_to_file(bbox_track, OUTPUT_BBOX)
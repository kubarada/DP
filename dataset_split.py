import os
import json
import random
import shutil

# Specify the path to your COCO format annotation file
coco_annotation_file = 'dataset/dp_dataset/frames_with_annotations/instances_default.json'

# Specify the ratios for train, validation, and test splits (e.g., 70-15-15)
train_ratio = 0.9
valid_ratio = 0.05
test_ratio = 0.05

# Create directories for train, validation, and test sets
os.makedirs('dataset/train', exist_ok=True)
os.makedirs('dataset/valid', exist_ok=True)
os.makedirs('dataset/test', exist_ok=True)

# Load the COCO format annotation file
with open(coco_annotation_file, 'r') as file:
    coco_data = json.load(file)

# Get the list of image file names and their corresponding annotations
images = coco_data['images']
annotations = coco_data['annotations']

# Shuffle the frames_with_annotations and annotations randomly
random.shuffle(images)

# Calculate the number of frames_with_annotations for each split
num_images = len(images)
num_train = int(num_images * train_ratio)
num_valid = int(num_images * valid_ratio)

# Split the frames_with_annotations and annotations and update file paths
train_images = images[:num_train]
valid_images = images[num_train:num_train + num_valid]
test_images = images[num_train + num_valid:]

train_annotations = [anno for anno in annotations if anno['image_id'] in [img['id'] for img in train_images]]
valid_annotations = [anno for anno in annotations if anno['image_id'] in [img['id'] for img in valid_images]]
test_annotations = [anno for anno in annotations if anno['image_id'] in [img['id'] for img in test_images]]

# Update image file paths in the annotations
train_image_dir = 'train'
valid_image_dir = 'valid'
test_image_dir = 'test'

for image in train_images:
    image['file_name'] = os.path.join(train_image_dir, image['file_name'])

for image in valid_images:
    image['file_name'] = os.path.join(valid_image_dir, image['file_name'])

for image in test_images:
    image['file_name'] = os.path.join(test_image_dir, image['file_name'])

# Update image IDs in the annotations to match the new IDs
id_mapping = {image['id']: i + 1 for i, image in enumerate(train_images + valid_images + test_images)}
for annotation in annotations:
    annotation['image_id'] = id_mapping[annotation['image_id']]

# Create the new COCO format annotation files for each split
train_coco_data = {'frames_with_annotations': train_images, 'annotations': train_annotations, 'categories': coco_data['categories']}
valid_coco_data = {'frames_with_annotations': valid_images, 'annotations': valid_annotations, 'categories': coco_data['categories']}
test_coco_data = {'frames_with_annotations': test_images, 'annotations': test_annotations, 'categories': coco_data['categories']}

# Save the new annotation files
with open('dataset/train_annotations.json', 'w') as file:
    json.dump(train_coco_data, file)

with open('dataset/valid_annotations.json', 'w') as file:
    json.dump(valid_coco_data, file)

with open('dataset/test_annotations.json', 'w') as file:
    json.dump(test_coco_data, file)

# Move the frames_with_annotations to their respective directories
for image in train_images:
    src_image_path = os.path.join('dataset/dp_dataset/frames_with_annotations', os.path.basename(image['file_name']))
    dst_image_path = os.path.join('dataset/train', os.path.basename(image['file_name']))
    shutil.move(src_image_path, dst_image_path)

for image in valid_images:
    src_image_path = os.path.join('dataset/dp_dataset/frames_with_annotations', os.path.basename(image['file_name']))
    dst_image_path = os.path.join('dataset/valid', os.path.basename(image['file_name']))
    shutil.move(src_image_path, dst_image_path)

for image in test_images:
    src_image_path = os.path.join('dataset/dp_dataset/frames_with_annotations', os.path.basename(image['file_name']))
    dst_image_path = os.path.join('dataset/test', os.path.basename(image['file_name']))
    shutil.move(src_image_path, dst_image_path)

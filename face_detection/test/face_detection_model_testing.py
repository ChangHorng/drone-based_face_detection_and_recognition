import cv2
import json
import ntpath
import os
import pandas as pd
import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode


def create_annotations_data_frame(json_file_path, class_name):

    dataset = []

    # Read JSON file to extract image annotations
    with open(json_file_path) as json_data:
        data = json.load(json_data)
        images_data = pd.DataFrame(data['images'])
        faces_data = pd.DataFrame(data['annotations'])

    # Iterate through annotations of face instances in every image
    for _, row in tqdm.tqdm(faces_data.iterrows(), total=faces_data.shape[0]):
        annotation = row['bbox']
        image_id = row['image_id']
        face_data = {}

        x, y = annotation[0], annotation[1]
        width, height = annotation[2], annotation[3]

        # Compute and assign the respective information from the image annotations to a dictionary
        face_data['file_name'] = images_data.loc[image_id, 'file_name']
        face_data['class_name'] = class_name
        face_data['width'] = images_data.loc[image_id, 'width']
        face_data['height'] = images_data.loc[image_id, 'height']
        face_data['x_min'] = int(x)
        face_data['y_min'] = int(y)
        face_data['x_max'] = int(x + width)
        face_data['y_max'] = int(y + height)

        dataset.append(face_data)

    # Convert the list of dictionaries into a DataFrame object
    data_frame = pd.DataFrame(dataset)

    unique_files = data_frame.file_name.unique()

    return data_frame[data_frame.file_name.isin(unique_files)]


# Initialize config for testing
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.OUTPUT_DIR = '../train/output'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
# A threshold value chosen to filter out detections that are below 70% precision
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
# cfg.MODEL.DEVICE = 'cpu'                      # Uncomment this line if your device does not have a CUDA compatible GPU

# Set up default predictor
predictor = DefaultPredictor(cfg)


# 1 - Test on DroneFace dataset's test folder
os.makedirs('droneFace_test_result', exist_ok=True)

droneFace_testing_data_frame = create_annotations_data_frame('../droneFace_dataset/test/_annotations.coco.json', 'face')
IMAGES_PATH = '../droneFace_dataset/test'
test_image_path = droneFace_testing_data_frame.file_name.unique()

for clothing_image in test_image_path:
    file_path = f'{IMAGES_PATH}/{clothing_image}'
    im = cv2.imread(file_path)
    outputs = predictor(im)

    v = Visualizer(
        im[:, :, ::-1],
        metadata={'thing_classes': ['face']},
        scale=1.,
        instance_mode=ColorMode.IMAGE
    )

    instances = outputs["instances"].to("cpu")
    # Remove the masks from instance segmentation and retain only the bounding box
    instances.remove('pred_masks')
    v = v.draw_instance_predictions(instances)
    result = v.get_image()[:, :, ::-1]
    file_name = ntpath.basename(clothing_image)
    write_res = cv2.imwrite(f'droneFace_test_result/{file_name}', result)

# 2 - Test on DroneFace dataset's valid folder
os.makedirs('droneFace_valid_result', exist_ok=True)

droneFace_validation_data_frame = create_annotations_data_frame('../droneFace_dataset/valid/_annotations.coco.json',
                                                                'face')
IMAGES_PATH = '../droneFace_dataset/valid'
valid_image_path = droneFace_validation_data_frame.file_name.unique()

for clothing_image in valid_image_path:
    file_path = f'{IMAGES_PATH}/{clothing_image}'
    im = cv2.imread(file_path)
    outputs = predictor(im)

    v = Visualizer(
        im[:, :, ::-1],
        metadata={'thing_classes': ['face']},
        scale=1.,
        instance_mode=ColorMode.IMAGE
    )

    instances = outputs["instances"].to("cpu")
    instances.remove('pred_masks')
    v = v.draw_instance_predictions(instances)
    result = v.get_image()[:, :, ::-1]
    file_name = ntpath.basename(clothing_image)
    write_res = cv2.imwrite(f'droneFace_valid_result/{file_name}', result)

# 3 - Test on person's faces dataset's test folder
os.makedirs('person_faces_test_result', exist_ok=True)

person_faces_testing_data_frame = create_annotations_data_frame('../person_faces_dataset/test/_annotations.coco.json',
                                                                'face')
IMAGES_PATH = '../person_faces_dataset/test'
test_image_path = person_faces_testing_data_frame.file_name.unique()

for clothing_image in test_image_path:
    file_path = f'{IMAGES_PATH}/{clothing_image}'
    im = cv2.imread(file_path)
    outputs = predictor(im)

    v = Visualizer(
        im[:, :, ::-1],
        metadata={'thing_classes': ['face']},
        scale=1.,
        instance_mode=ColorMode.IMAGE
    )

    instances = outputs["instances"].to("cpu")
    instances.remove('pred_masks')
    v = v.draw_instance_predictions(instances)
    result = v.get_image()[:, :, ::-1]
    file_name = ntpath.basename(clothing_image)
    write_res = cv2.imwrite(f'person_faces_test_result/{file_name}', result)

# 4 - Test on person's faces dataset's valid folder
os.makedirs('person_faces_valid_result', exist_ok=True)

person_faces_validation_data_frame = create_annotations_data_frame('../person_faces_dataset/valid/_annotations.coco.json',
                                                                   'face')
IMAGES_PATH = '../person_faces_dataset/valid'
valid_image_path = person_faces_validation_data_frame.file_name.unique()

for clothing_image in valid_image_path:
    file_path = f'{IMAGES_PATH}/{clothing_image}'
    im = cv2.imread(file_path)
    outputs = predictor(im)

    v = Visualizer(
        im[:, :, ::-1],
        metadata={'thing_classes': ['face']},
        scale=1.,
        instance_mode=ColorMode.IMAGE
    )

    instances = outputs["instances"].to("cpu")
    instances.remove('pred_masks')
    v = v.draw_instance_predictions(instances)
    result = v.get_image()[:, :, ::-1]
    file_name = ntpath.basename(clothing_image)
    write_res = cv2.imwrite(f'person_faces_valid_result/{file_name}', result)

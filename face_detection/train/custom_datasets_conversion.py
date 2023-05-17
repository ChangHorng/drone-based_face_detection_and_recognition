import itertools
import json
import pandas as pd
import tqdm

from detectron2.structures import BoxMode


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


def convert_dataset_dicts(data_frame, classes, image_path):

    dataset_dicts = []

    #  Iterate through data frame object for image annotations
    for image_id, image_name in enumerate(data_frame.file_name.unique()):
        # Assign respective image details to record dictionary
        image_record = {}
        image_data_frame = data_frame[data_frame.file_name == image_name]

        file_path = f'{image_path}/{image_name}'
        image_record['file_name'] = file_path
        image_record['image_id'] = image_id
        image_record['height'] = int(image_data_frame.iloc[0].height)
        image_record['width'] = int(image_data_frame.iloc[0].width)

        # Store each instance annotated in the image
        faces = []
        for _, row in image_data_frame.iterrows():
            x_min = int(row.x_min)
            y_min = int(row.y_min)
            x_max = int(row.x_max)
            y_max = int(row.y_max)

            box = [
                (x_min, y_min), (x_max, y_min),
                (x_max, y_max), (x_min, y_max)
            ]
            box = list(itertools.chain.from_iterable(box))

            face = {
                'bbox': [x_min, y_min, x_max, y_max],
                'bbox_mode': BoxMode.XYXY_ABS,
                'segmentation': [box],
                'category_id': classes.index(row.class_name),
                'iscrowd': 0
            }
            faces.append(face)

        # Assign instances annotated to the record dictionary
        image_record['annotations'] = faces
        dataset_dicts.append(image_record)

    # Return a list of dictionaries that contain all annotations in every image
    return dataset_dicts

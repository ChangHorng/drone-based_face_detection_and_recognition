"""
We trained our model on Google Colab with a subscription to Google Colab Pro+ and gained access to a high-end GPU which
shortened the training time significantly. The link to the python notebook on Google Colab is attached below.
https://colab.research.google.com/drive/1jBf1FECDkiGMPl0EorSVa7OcFVUfA2m7?usp=sharing

Note that you can also run the training code on your local device if you have a high-end GPU which will produce the
same output.
"""


import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger

from custom_datasets_conversion import create_annotations_data_frame, convert_dataset_dicts


class CocoTrainer(DefaultTrainer):
    """
    This is a custom trainer class that inherits the default trainer class of Detectron2 for model training purpose.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs('coco_evaluation', exist_ok=True)
            output_folder = 'coco_evaluation'

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


# Enable Detectron2 logger to generate output in terminal for debugging purpose
setup_logger()

# Initialize the 'face' class for object detection
classes = ['face']

# Create data frame for training and validation steps during model training for DroneFace and Person Faces datasets
droneFace_training_data_frame = create_annotations_data_frame('../droneFace_dataset/train/_annotations.coco.json', 'face')
droneFace_validation_data_frame = create_annotations_data_frame('../droneFace_dataset/valid/_annotations.coco.json', 'face')

person_faces_training_data_frame = create_annotations_data_frame('../person_faces_dataset/train/_annotations.coco.json', 'face')
person_faces_validation_data_frame = create_annotations_data_frame('../person_faces_dataset/valid/_annotations.coco.json', 'face')

# Register training and validation datasets and its respective metadata
for i in ['train', 'valid']:
    DatasetCatalog.register('droneFace_' + i + '_dataset', lambda i=i: convert_dataset_dicts(
        droneFace_training_data_frame if i == 'train' else droneFace_validation_data_frame, classes,
        '../droneFace_dataset/' + i))
    MetadataCatalog.get('droneFace_' + i + '_dataset').set(thing_classes=classes)

    DatasetCatalog.register('person_faces_' + i + '_dataset', lambda i=i: convert_dataset_dicts(
        person_faces_training_data_frame if i == 'train' else person_faces_validation_data_frame, classes,
        '../person_faces_dataset/' + i))
    MetadataCatalog.get('person_faces_' + i + '_dataset').set(thing_classes=classes)

# Initialize settings for config setting
cfg = get_cfg()

# Here we are using COCO Instance Segmentation Baselines with Mask R-CNN (R50-FPN) pre-trained model from Detectron2
# Model Zoo
# https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')

# Assign registered datasets in config setting
cfg.DATASETS.TRAIN = ('droneFace_train_dataset', 'person_faces_train_dataset')
cfg.DATASETS.TEST = ('droneFace_valid_dataset', 'person_faces_valid_dataset')

# The config attributes below are fine-tuned for our face detection model
cfg.DATALOADER.NUM_WORKERS = 1
cfg.SOLVER.BASE_LR = 0.0005
cfg.SOLVER.WARMUP_ITERS = 2075
# Total training images - 4150
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.MAX_ITER = 20750
cfg.SOLVER.STEPS = [8300, 9337, 10374, 11411, 12448, 13485, 14522, 15559, 16596, 17633]
cfg.SOLVER.GAMMA = 0.6
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
cfg.TEST.EVAL_PERIOD = 4150
# cfg.MODEL.DEVICE = 'cpu'                      # Uncomment this line if your device does not have a CUDA compatible GPU

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Initialise COCO trainer and start training with the registered training dataset
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

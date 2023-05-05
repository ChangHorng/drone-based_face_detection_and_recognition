import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger

from custom_datasets_conversion import create_annotations_data_frame, convert_dataset_dicts


class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs('coco_evaluation', exist_ok=True)
            output_folder = 'coco_evaluation'

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


setup_logger()

classes = ['face']

droneFace_training_data_frame = create_annotations_data_frame('../droneFace_dataset/train/_annotations.coco.json', 'face')
droneFace_validation_data_frame = create_annotations_data_frame('../droneFace_dataset/valid/_annotations.coco.json', 'face')

person_faces_training_data_frame = create_annotations_data_frame('../person_faces_dataset/train/_annotations.coco.json', 'face')
person_faces_validation_data_frame = create_annotations_data_frame('../person_faces_dataset/valid/_annotations.coco.json', 'face')

for i in ['train', 'valid']:
    DatasetCatalog.register('droneFace_' + i + '_dataset', lambda i=i: convert_dataset_dicts(
        droneFace_training_data_frame if i == 'train' else droneFace_validation_data_frame, classes,
        '../droneFace_dataset/' + i))
    MetadataCatalog.get('droneFace_' + i + '_dataset').set(thing_classes=classes)

    DatasetCatalog.register('person_faces_' + i + '_dataset', lambda i=i: convert_dataset_dicts(
        person_faces_training_data_frame if i == 'train' else person_faces_validation_data_frame, classes,
        '../person_faces_dataset/' + i))
    MetadataCatalog.get('person_faces_' + i + '_dataset').set(thing_classes=classes)

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
cfg.DATASETS.TRAIN = ('droneFace_train_dataset', 'person_faces_train_dataset')
cfg.DATASETS.TEST = ('droneFace_valid_dataset', 'person_faces_valid_dataset')

cfg.DATALOADER.NUM_WORKERS = 1
cfg.SOLVER.BASE_LR = 0.0005
cfg.SOLVER.WARMUP_ITERS = 2075
# Images = 4150
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.MAX_ITER = 20750
cfg.SOLVER.STEPS = [8300, 9337, 10374, 11411, 12448, 13485, 14522, 15559, 16596, 17633]
cfg.SOLVER.GAMMA = 0.6
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
cfg.TEST.EVAL_PERIOD = 4150
# cfg.MODEL.DEVICE = 'cpu'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

import cv2
import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from djitellopy import Tello
from drone_keyboard_controller.keyboard_control import start

def get_frame(drone: Tello, width, height):
    img = drone.get_frame_read().frame
    img_resize = cv2.resize(img, (width, height))
    return img_resize


def face_detection_setup():
    MetadataCatalog.get("droneFace_train_dataset").set(thing_classes="face")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    cfg.OUTPUT_DIR = '../train/output'
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
    cfg.MODEL.DEVICE = "cpu"

    predictor = DefaultPredictor(cfg)

    return predictor


def face_detection(predictor, frame):
    outputs = predictor(frame)

    v = Visualizer(
        frame[:, :, ::-1],
        metadata={"thing_classes": ['face']},
        scale=1.,
        instance_mode=ColorMode.IMAGE
    )

    instances = outputs["instances"].to("cpu")
    instances.remove('pred_masks')
    v = v.draw_instance_predictions(instances)
    result = v.get_image()[:, :, ::-1]

    return result


if __name__ == "__main__":
    predictor = face_detection_setup()

    tello = Tello()
    tello.connect()

    start()

    # tello.takeoff()
    tello.streamon()

    fps = 30
    skipping_frame = 5
    count_frame = 0
    exit = 0
    width, height = 640, 480

    while not exit:
        frame = get_frame(tello, width, height)
        if count_frame % skipping_frame == 0:
            frame = face_detection(predictor, frame)

        cv2.imshow("Real-time Face Detection", frame)

        count_frame += 1

        # exit = int(input("Type 0 if you want to stop; else 1: "))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # tello.land()
            break

    tello.streamoff()

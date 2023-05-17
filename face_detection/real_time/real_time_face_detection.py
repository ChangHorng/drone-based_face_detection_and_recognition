"""
In order to execute face detection in real-time smoothly with a higher FPS, we used a PC setup with the following
specifications.

CPU - AMD Ryzen 5 3600
GPU - MSI RTX2060 AERO ITX
RAM - Corsair Vengeance 3200 MHz C16 16GB (2 X 8GB)
"""


import cv2
import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from djitellopy import Tello


def detectron2_face_detection_config_setup():

    # Retrieve metadata of registered dataset
    MetadataCatalog.get("droneFace_train_dataset").set(thing_classes="face")

    # Initialize config for real-time
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.OUTPUT_DIR = '../train/output'
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    # cfg.MODEL.DEVICE = 'cpu'                  # Uncomment this line if your device does not have a CUDA compatible GPU

    # Set up default predictor
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
    # Remove the masks from instance segmentation and retain only the bounding box
    instances.remove('pred_masks')
    v = v.draw_instance_predictions(instances)
    result = v.get_image()[:, :, ::-1]

    return result


def get_frame(drone, width, height):

    # Retrieve and resize frame from Tello EDU drone's camera
    img = drone.get_frame_read().frame
    img_resize = cv2.resize(img, (width, height))

    return img_resize


if __name__ == "__main__":

    predictor = detectron2_face_detection_config_setup()

    # Set the output window to a specific width and height
    width, height = 640, 480

    # Create Tello object and connect device with Tello EDU drone
    tello = Tello()
    tello.connect()

    # Switch on the camera of Tello EDU drone
    tello.streamon()

    exit = 0
    while not exit:
        # Keep getting frame from Tello EDU drone's camera
        frame = get_frame(tello, width, height)
        # Process the frame with our trained model
        frame = face_detection(predictor, frame)

        # Show the output frame using cv2
        cv2.imshow("Real-time Face Detection", frame)

        # User can terminate the program by pressing on key 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Switch off the camera of Tello EDU drone and disconnect with device
    tello.streamoff()
    tello.end()

"""
In order to execute face detection in real-time smoothly with a higher FPS, we used a PC setup with the following
specifications.

CPU - AMD Ryzen 5 3600
GPU - MSI RTX2060 AERO ITX
RAM - Corsair Vengeance 3200 MHz C16 16GB (2 X 8GB)
"""


import cv2
import keyboard_control as kp
import os
import pygame.display

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

    cfg.OUTPUT_DIR = '../face_detection/train/output'
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
    takeoff = False
    landed = True
    open_cam = False

    # Create Tello object and connect device with Tello EDU drone
    tello = Tello()
    tello.connect()

    # Start keyboard controller
    window = kp.start_control()
    vals = kp.get_keyboard_input()

    # Keep iterating while drone hasn't landed after takeoff
    while vals[4] is not False:
        # Retrieve keyboard input from user
        vals = kp.get_keyboard_input()

        # Check if user has given input for drone to takeoff or land
        if vals[4] is True:
            if takeoff is False:
                tello.takeoff()
                takeoff = True
                landed = False
            else:
                print("Already take off!")
        elif vals[4] is False:
            if landed is False:
                tello.land()
                landed = True
            else:
                # reset the value to keep system running
                vals[4] = True
                print('Drone has not taken off!')

        # Check if user has given input for drone's camera to start or end streaming
        if vals[5] is True:
            tello.streamon()
            open_cam = True
        elif vals[5] is False:
            tello.streamoff()
            open_cam = False

        # Send the displacement values for the drone to carry out the respective movement
        tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])

        # If the drone camera is streaming, then apply face detection to every frame; else switch off cv2 window
        if open_cam is True:
            frame = get_frame(tello, width, height)
            frame = face_detection(predictor, frame)
            cv2.imshow("Active Surveillance System", frame)
        elif open_cam is False:
            cv2.destroyAllWindows()

        # Update the battery life of Tello EDU drone on pygame window to ensure the drone has enough battery to deploy
        battery = tello.get_battery()
        font = pygame.font.SysFont("Arial", 26)
        txt_surf = font.render("Battery Left: " + str(battery), True, (255, 255, 255))
        # Write text on the pygame window
        window.fill((0, 0, 0))
        window.blit(txt_surf, (200 - txt_surf.get_width() // 2, 200 - txt_surf.get_height() // 2))
        pygame.display.flip()

    # Quit the pygame window to switch off keyboard controller and disconnect Tello EDU drone with device
    pygame.display.quit()
    tello.end()

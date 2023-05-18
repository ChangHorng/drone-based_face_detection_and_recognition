"""
The images stored in the Face Recognition Dataset folder should be in file formats that are supported by cv2, where
images with formats such as .HEIC should not be used, so that the program can execute successfully.
"""


import cv2
import face_recognition as fr
import numpy as np
import os

from djitellopy import Tello


def face_recognition_setup(path):

    # Path must be the file path to the image dataset
    path = path
    images = []
    person_names = []

    image_names = os.listdir(path)
    # Retrieve the names of each person in the dataset
    for image_name in image_names:
        current_image = cv2.imread(f'{path}/{image_name}')
        if current_image is not None:
            images.append(current_image)
            person_names.append(os.path.splitext(image_name)[0])

    encoding_list = find_encodings(images)

    return person_names, encoding_list


def find_encodings(images):
    encoding_list = []

    # Iterate through image dataset to get face encodings
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encoding_list.append(encode)

    return encoding_list


def face_recognition(person_names, encoding_list, frame):

    # Scales down input frame
    resized_img = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # Detect face in input frame and process the face encoding
    faces_in_current_frame = fr.face_locations(resized_img, model='vgg')
    encodings_in_current_frame = fr.face_encodings(resized_img, faces_in_current_frame)

    # Iterate through the faces within our dataset to compare with the face detected in the input frame
    for face_encoding, face_location in zip(encodings_in_current_frame, faces_in_current_frame):
        face_distances = fr.face_distance(encoding_list, face_encoding)
        matched_index = np.argmin(face_distances)

        # Assign name to recognised face, else assign 'Unknown'
        if face_distances[matched_index] < 0.50:
            name = person_names[matched_index].upper()
        else:
            name = 'Unknown'

        # Annotate the rectangle box on the faces recognised with their respective names
        y1, x2, y2, x1 = face_location
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4                         # Rescale image back to original size
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


def get_frame(drone, width, height):

    # Retrieve and resize frame from Tello EDU drone's camera
    img = drone.get_frame_read().frame
    img_resize = cv2.resize(img, (width, height))

    return img_resize


if __name__ == "__main__":

    # Initialize data required for face recognition
    person_names, encoding_list = face_recognition_setup('Face Recognition Dataset')

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
        # Process the frame with in-built face recognition library
        face_recognition(person_names, encoding_list, frame)

        # Show the output frame using cv2
        cv2.imshow("Real-time Face Recognition", frame)

        # User can terminate the program by pressing on key 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Switch off the camera of Tello EDU drone and disconnect with device
    tello.streamoff()
    tello.end()

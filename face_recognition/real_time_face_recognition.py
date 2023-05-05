import cv2
import face_recognition
import numpy as np
import os

from datetime import datetime

# from PIL import ImageGrab

path = 'Attendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    # img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS, model="vgg")
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        # if matches[matchIndex]:
        #     name = classNames[matchIndex].upper()
        # # print(name)
        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            markAttendance(name)
        else:
            name = 'Unknown'
        # print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

    # a face from multiple sides
#  Can we store image encodings somewhere so that we don't need to calculate it every time. Let say I have 20 images of me (used to achieve accuracy) and total around 100+ images in that folder. To find encoding of 100 images again and again will increase the time complexity of the program.
#  just split the code into 2 py file at 1st file get the encode data and save it in a CSV file as array. in 2nd file just load the encoded data from that CSV file and do the recognition.
# https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
# https://gist.github.com/ageitgey/1c1cb1c60ace321868f7410d48c228e1
# https://github.com/ageitgey/face_recognition/blob/master/README.rst#

'''
Sure! Here are some pros and cons of each of the face recognition models mentioned earlier:

1. CNN:
Pros:
- Provides good accuracy for face recognition
- Faster and less computationally intensive compared to some other deep learning models like VGG-Face
- Works well with lower resolution images

Cons:
- May not be as accurate as some other deep learning models, particularly on larger and more complex datasets
- May struggle with recognizing faces in varying lighting conditions or with occlusions

2. VGG-Face:
Pros:
- Has achieved high accuracy in several face recognition benchmarks
- Can recognize faces with a high degree of variability, such as pose and expression changes
- Robust to occlusions

Cons:
- Large and computationally intensive model that requires significant computational resources
- May not be suitable for real-time face recognition applications due to its high computational requirements

3. ResNet:
Pros:
- Can achieve high accuracy in face recognition tasks
- Can learn to recognize complex features from images
- Relatively faster and less computationally intensive compared to VGG-Face

Cons:
- May require a larger amount of training data to achieve high accuracy compared to other models
- May be sensitive to overfitting if not trained properly

4. Inception:
Pros:
- Can achieve high accuracy in face recognition tasks
- Efficient architecture that uses multiple filters of different sizes in the same layer
- Can learn to recognize complex features from images

Cons:
- May require a larger amount of training data to achieve high accuracy compared to other models
- May be sensitive to overfitting if not trained properly

Overall, the choice of a face recognition model depends on several factors such as accuracy requirements, computational resources, and real-time processing requirements.
'''

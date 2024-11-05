import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'imageattendance'
images = []
classNames = []
myList = os.listdir(path)

# Load images and their encodings
for cls in myList:
    curImg = cv2.imread(os.path.join(path, cls))
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name, place):
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    with open('Attendance.csv', 'r+') as f:
        mydatalist = f.readlines()
        namelist = [line.split(',')[0] for line in mydatalist]
        if name not in namelist:
            f.writelines(f'\n{name},{dtString},{place}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Start video capture
video_cap = cv2.VideoCapture(0)
if not video_cap.isOpened():
    print("Error: Could not open video.")
    exit()

place = "Ai and ML LAB"
print("Press 'q' to exit the video stream.")

while True:
    success, img = video_cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 5, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name, place)

    cv2.imshow('Webcam', img)

    # Check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_cap.release()
cv2.destroyAllWindows()

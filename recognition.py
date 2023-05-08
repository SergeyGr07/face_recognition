import face_recognition
import os
from datetime import datetime
from cv2 import cv2
import numpy as np


def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


def mark_attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []

        for line in myDataList:
            entry = line.split(';')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name};{dtString}')


def main():
    path = 'ImageAttendance'
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)
    for cls in myList:
        curImg = cv2.imread(f'{path}/{cls}')
        images.append(curImg)
        classNames.append(os.path.splitext(cls)[0])

    print(classNames)

    encodeListKnown = find_encodings(images)
    print('Encoding Complete')

    cap = cv2.VideoCapture(1)
    while True:
        success, img = cap.read()
        imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faceCurFrame = face_recognition.face_locations(imgs)
        encodeCurFrame = face_recognition.face_encodings(imgs, faceCurFrame)

        for encodeFace, facelock in zip(encodeCurFrame, faceCurFrame):
            mathes = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print('faceDis', faceDis)
            matchIndex = np.argmin(faceDis)

            if mathes[matchIndex]:
                name = classNames[matchIndex]
                print(name)
                y1, x2, y2, x1 = facelock
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 255), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                mark_attendance(name)

        # Для создания ограничивающей рамки и имени в ней
        cv2.imshow('Webcam', img)
        key = cv2.waitKey(1) & 0xFF  # Ждем нажатия клавиши в течение 1 миллисекунды
        if key == ord('q'):  # Если нажата клавиша 'q'
            break

    cap.release
    

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
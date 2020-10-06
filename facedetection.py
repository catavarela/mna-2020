# https://pypi.org/project/face-recognition/
import math

import cv2 as cv
import numpy as np


def get_face(img, width=150, height=150):
    face_classifier = cv.CascadeClassifier(cv.data.haarcascades + './haarcascade_frontalface_default.xml')
    eyes_classifier = cv.CascadeClassifier(cv.data.haarcascades + './haarcascade_eye.xml')
    faces_detected = face_classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    if len(faces_detected) > 0:
        (x, y, w, h) = faces_detected[0]

        eyes = eyes_classifier.detectMultiScale(img[y:y + h, x:x + w])

        if len(eyes) == 2:
            rows, cols = img.shape[:2]
            center_eyes = [
                [(eyes[0][0] + eyes[0][2]) / 2, (eyes[0][1] + eyes[0][3]) / 2],
                [(eyes[1][0] + eyes[1][2]) / 2, (eyes[1][1] + eyes[1][3]) / 2],
            ]
            angle = math.atan2(center_eyes[1][1] - center_eyes[0][1], center_eyes[1][0] - center_eyes[0][0])
            M = cv.getRotationMatrix2D((cols / 2, rows / 2), math.degrees(angle), 1)
            img = cv.warpAffine(img, M, (cols, rows))

        img = img[y + 1: y + h, x + 1: x + w]

    return cv.resize(img, (width, height), interpolation=cv.INTER_LINEAR)


def get_face_as_row(path, side_length=150):
    face = cv.imread(path)
    face = get_face(face, side_length, side_length)
    face = cv.cvtColor(face, cv.COLOR_RGB2GRAY)
    face = np.reshape(face, side_length * side_length)
    return face / 255.0


def get_faces_as_rows(paths, side_length=150):
    faces = np.zeros((len(paths), side_length * side_length))
    for i in range(0, len(paths)):
        faces[i] = get_face_as_row(paths[i], side_length)
    return faces


def get_face_as_column(path, side_length=150):
    return get_face_as_row(path, side_length)[:, np.newaxis]


def get_faces_as_columns(paths, side_length=150):
    faces = get_faces_as_rows(paths, side_length)
    return faces.transpose()

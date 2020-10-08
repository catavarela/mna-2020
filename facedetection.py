# https://pypi.org/project/face-recognition/
import math

import cv2 as cv
import numpy as np


class FaceRecognition:
    def __init__(self):
        self.face_classifier = cv.CascadeClassifier(cv.data.haarcascades + './haarcascade_frontalface_default.xml')
        self.eyes_classifier = cv.CascadeClassifier(cv.data.haarcascades + './haarcascade_eye.xml')
        self.faces_as_row = {} # key: path, value: face as row

    def get_face(self, img, width=150, height=150):
        faces_detected = self.face_classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

        if len(faces_detected) > 0:
            (x, y, w, h) = faces_detected[0]

            eyes = self.eyes_classifier.detectMultiScale(img[y:y + h, x:x + w])

            if len(eyes) == 2:
                rows, cols = img.shape[:2]
                center_eyes = [
                    [(eyes[0][0] + eyes[0][2]) / 2, (eyes[0][1] + eyes[0][3]) / 2],
                    [(eyes[1][0] + eyes[1][2]) / 2, (eyes[1][1] + eyes[1][3]) / 2],
                ]
                angle = math.atan2(center_eyes[1][1] - center_eyes[0][1], center_eyes[1][0] - center_eyes[0][0])

                if math.fabs(angle) > 1.5:
                    angle = math.atan2(center_eyes[0][1] - center_eyes[1][1], center_eyes[0][0] - center_eyes[1][0])
                if math.fabs(angle) < 1:
                    # Sin esto se podria dar una rotacion muy grande
                    M = cv.getRotationMatrix2D((cols / 2, rows / 2), math.degrees(angle), 1)
                    img = cv.warpAffine(img, M, (cols, rows))

            img = img[y + 1: y + h, x + 1: x + w]

        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        return cv.resize(img, (width, height), interpolation=cv.INTER_LINEAR)

    def get_face_as_row(self, path, side_length=150):
        if path not in self.faces_as_row:
            face = cv.imread(path)
            face = self.get_face(face, side_length, side_length)
            face = np.reshape(face, side_length * side_length)
            self.faces_as_row[path] = face / 255.0

        return self.faces_as_row[path]

    def get_faces_as_rows(self, paths, side_length=150):
        faces = np.zeros((len(paths), side_length * side_length))
        for i in range(0, len(paths)):
            faces[i] = self.get_face_as_row(paths[i], side_length)
        return faces

    def get_face_as_column(self, path, side_length=150):
        return self.get_face_as_row(path, side_length)[:, np.newaxis]

    def get_faces_as_columns(self, paths, side_length=150):
        faces = self.get_faces_as_rows(paths, side_length)
        return faces.transpose()

# https://pypi.org/project/face-recognition/
import math

import cv2 as cv


class FaceRecognition:
    def __init__(self):
        self.face_classifier = cv.CascadeClassifier(cv.data.haarcascades + './haarcascade_frontalface_default.xml')
        self.eyes_classifier = cv.CascadeClassifier(cv.data.haarcascades + './haarcascade_eye.xml')

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
                M = cv.getRotationMatrix2D((cols / 2, rows / 2), math.degrees(angle), 1)
                img = cv.warpAffine(img, M, (cols, rows))

            img = img[y + 1: y + h, x + 1: x + w]

        img = cv.resize(img, (width, height), interpolation=cv.INTER_LINEAR)
        return img

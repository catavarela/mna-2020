# https://pypi.org/project/face-recognition/
import face_recognition as fr
import cv2 as cv


def get_face(img, width=150, height=150):
    face_loc = fr.face_locations(img)
    if len(face_loc) > 0:
        img = img[face_loc[0][0]:face_loc[0][2], face_loc[0][3]:face_loc[0][1]]
    img = cv.resize(img, (width, height))
    return img

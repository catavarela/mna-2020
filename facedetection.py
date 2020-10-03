# https://pypi.org/project/face-recognition/
import face_recognition as fr
import cv2 as cv

def get_face(img):
    face_loc = fr.face_locations(img)
    print(face_loc)
    loc = face_loc[0]
    img = img[face_loc[0][0]:face_loc[0][2],face_loc[0][3]:face_loc[0][1]]
    return img
# Fuentes:
#   https://eprints.lancs.ac.uk/id/eprint/69825/1/face.pdf
#   https://medium.com/@ODSC/implementing-a-kernel-principal-component-analysis-in-python-495f04a7f85f


import cv2 as cv
import glob
import numpy as np
from itertools import chain
import random
import time
import facedetection as fd

from numpy import exp
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from math import ceil
images_to_use = 100
image_side_len = 150

def generate_eigenfaces_kpca(rootdir, gamma = 15,  keep_percentage = 0.5,):
    # Obtener los datos y pasarlos a una matriz como en PCA
    images = glob.glob(rootdir + '/**/*0[1-3].jpg', recursive=True)
    images.sort()
    faces = []
    for i in range(0, images_to_use):  # TODO: cambiar el images_to_use por len(faces)
        face = cv.imread(images[i])
        face = fd.get_face(face)
        face = cv.cvtColor(face, cv.COLOR_RGB2GRAY)

        faces.append(face)


    # Calcular y restar media
    avg = np.mean(faces, 1)[:, np.newaxis]
    faces_min_avg = faces - avg

    faces_reshaped = []
    # Convert to M x (250^2) matrix
    for face in faces_min_avg:
        faces_reshaped.append(np.reshape(face, image_side_len * image_side_len))

    # Calcular la distancia entre puntos y convertir la matriz en MxM
    # TODO: Gran duda, no me estoy cargando un monton de valores con el squareform???
    sq_dist = squareform(pdist(faces_reshaped, 'euclidean'))

    # Calcular la matriz del kernel segun el metodo Gaussiano
    kernel_matrix = exp(-gamma * sq_dist)

    # Centrar la matriz
    N = kernel_matrix.shape[0]
    one_n = np.ones((N,N)) / N
    kernel_matrix = kernel_matrix - one_n.dot(kernel_matrix) - kernel_matrix.dot(one_n) + one_n.dot(kernel_matrix).dot(one_n)
    
    # Calculo los autovectores a la matriz
    # TODO: cambiar por implementacion propia
    eigvalues, eigvectors = eigh(kernel_matrix)
    # Retornamos los mas significativos
    return np.column_stack((eigvectors[:,-i] for i in range(1,ceil(len(eigvectors[0]) * keep_percentage))))



generate_eigenfaces_kpca('data')




        

    
       
                           

       

    
        

generate_eigenfaces_kpca('data', 15, 0.5)


    



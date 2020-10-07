# Fuentes:
#   https://eprints.lancs.ac.uk/id/eprint/69825/1/face.pdf
#   https://medium.com/@ODSC/implementing-a-kernel-principal-component-analysis-in-python-495f04a7f85f


import cv2 as cv
import glob
import numpy as np
import Matrix_Handling_Functions as mhf
from itertools import chain
import random
import time
import facedetection as fd
from os import listdir
from os.path import join, isdir
import matplotlib.pyplot as plt
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



def kpca(rootdir, people, train, test, kernel_denom, kernel_ctx, kernel_degree):

    versize = 160
    horsize = 120

    #Here we store the directories of the path
    person_dir = [k for k in listdir(rootdir) if isdir(join(rootdir))]

    #array size parameters
    size = versize * horsize
    train_amount = people * train
    test_amount = people * test


    #image training
    images = np.zeros([train_amount, size])
    person = np.zeros([train_amount, 1])
    image_num = 0
    per = 0
    trainingname = {}

    for dire in person_dir:
        for m in range(1, train + 1):
            image = plt.imread(rootdir + dire + '/{}'.format(m) + '.pgm')
            images[image_num, :] = (np.reshape(image, [1, size]) - 127.5)/127.5
            person[image_num, 0] = per
            image_num += 1
        trainingname[per] = dire
        per += 1

    #image test

    image_test = np.zeros([test_amount, size])
    person_test = np.zeros([test_amount, 1])
    image_num = 0
    per = 0

    for dire in person_dir:
        for m in range(train_amount, train, train + test):
            image = plt.imread(rootdir + dire + '/{}'.format(m) + '.pgm')
            image_test[image_num, :] = (np.reshape(image, [1, size]) - 127.5)/ 127.5
            person_test[image_num,0] = per
            image_num += 1
        per+=1


    #KERNEL
    # TODO: chequear esto
    K = ((np.dot(images,images.T)/kernel_denom) + kernel_ctx) ** kernel_degree

    #esta transformacion es equivalente a centrar las imagenes originales
    unoM = np.ones([train_amount, train_amount]) / train_amount
    K = K - np.dot(unoM, K) - np.dot(K, unoM) + np.dot(unoM, np.dot(K, unoM))

    #Autovalores y autovectores
    w, alpha = mhf.get_eigen_from_qr(K, 1000)
    lambdas = w/ train_amount
    lambdas = w

    #ordenar ascendentemente los autovalores
    lambdas = np.flipud(lambdas)
    alpha = np.fliplr(alpha)

    for col in range(alpha.shape[1]):
        alpha[:, col] = alpha[:, col]/np.sqrt(lambdas[col])


    #pre-proyeccion
    improypre = np.dot(K.T, alpha)
    unoML = np.ones([test_amount, train_amount]) / train_amount
    Ktest = (np.dot(image_test, images.T) / kernel_denom + kernel_ctx) ** kernel_degree
    Ktest = Ktest - np.dot(unoML, K) - np.dot(Ktest, unoM) + np.dot(unoML, np.dot(K, unoM))
    imtstproypre = np.dot(Ktest, alpha)







rootdir = 'data/Fotos/'
kernel_degree = 2
kernel_ctx = 1
kernel_denom = 10
people_number = 5
train_number = 4 
test_number = 6
versize = 160
horsize = 120

kpca(rootdir, people_number, train_number, test_number, kernel_denom, kernel_ctx, kernel_degree)


    



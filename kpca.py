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
from sklearn import svm


images_to_use = 100
image_side_len = 150

def kpca_experimental(rootdir, gamma = 15,  keep_percentage = 0.5,):
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

    # Ya predifinido las longitudes de pixeles de la imagen
    VERTICAL_SIZE = 160
    HORIZONTAL_SIZE = 120

    # Directorios donde estan las imagenes de cada persona
    person_dir = [k for k in listdir(rootdir) if isdir(join(rootdir))]

    # Tamanios de los arrays que vamos a usar
    size = VERTICAL_SIZE * HORIZONTAL_SIZE
    train_amount = people * train
    test_amount = people * test


    ############################# Imagenes que usamos para training ########################
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

    ############################# Imagenes que usamos para testing #######################

    image_test = np.zeros([test_amount, size])
    person_test = np.zeros([test_amount, 1])
    image_num = 0
    per = 0

    for dire in person_dir:
        for m in range(train, train + test):
            image = plt.imread(rootdir + dire + '/{}'.format(m) + '.pgm')
            image_test[image_num, :] = (np.reshape(image, [1, size]) - 127.5)/ 127.5
            person_test[image_num,0] = per
            image_num += 1
        per+=1

    print(images.shape)

    # Aplicamos la transformacion kernel para la conjunto de imagenes de training
    # Es una transformacion polinomial simple
    # TODO: chequear esto
    kernel_matrix= (np.dot(images,images.T)/kernel_denom + kernel_ctx ) ** kernel_degree

    # Centrar la matriz del kernel
    unoM = np.ones([train_amount, train_amount]) / train_amount
    kernel_matrix= kernel_matrix- np.dot(unoM, kernel_matrix) - np.dot(kernel_matrix, unoM) + np.dot(unoM, np.dot(kernel_matrix, unoM))

    # Calculamos los autovalores y autovectores con nuestro propio metodo.
    w, alpha = mhf.get_eigen_from_qr(kernel_matrix, 1000)
    lambdas = w

    print(lambdas)
    print(alpha)

    for col in range(alpha.shape[1]):
        alpha[:, col] = alpha[:, col]/np.sqrt(lambdas[col])


    # Realizamos la proyeccion de las eigenfaces sobre el conjunto de prueba
    training_proyection = np.dot(kernel_matrix.T, alpha)



    # Realizamos la transformacion de kernel sobre el conjunto de prueba
    kernel_matrix_test = (np.dot(image_test, images.T) / kernel_denom + kernel_ctx) ** kernel_degree
    unoML = np.ones([test_amount, train_amount]) / train_amount
    
    #Centramos la matriz 
    kernel_matrix_test = kernel_matrix_test - np.dot(unoML, kernel_matrix) - np.dot(kernel_matrix_test, unoM) + np.dot(unoML, np.dot(kernel_matrix, unoM))
    
    #Proyectamos sobre el conjunto de testing de las eigenfaces
    test_proyection = np.dot(kernel_matrix_test, alpha)

    #Realizamos el calculo de svc y calculamos la precision segun la cantidad de eigenfaces
    max_eigenfaces = 30
    accs = np.zeros([max_eigenfaces,1])
    clf = svm.LinearSVC()
    for eigen_n in range(1 ,max_eigenfaces):

        improy      = training_proyection[:,0:eigen_n]
        imtstproy   = test_proyection[:,0:eigen_n]

        clf.fit(improy,person.ravel())
        accs[eigen_n] = clf.score(imtstproy,person_test.ravel())
        print('Autovalores: {1}   Precision: {0} %\n'.format(eigen_n,accs[eigen_n]*100))







rootdir = 'data/Fotos/'
kernel_degree = 2
kernel_ctx = 1
kernel_denom = 30
people_number = 5
train_number = 4 
test_number = 6
VERTICAL_SIZE = 160
HORIZONTAL_SIZE = 120

kpca(rootdir, people_number, train_number, test_number, kernel_denom, kernel_ctx, kernel_degree)


    



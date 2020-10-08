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

def classify_face_by_pca(rootdir, people, train, face_name, face_number):

    # Ya predifinido las longitudes de pixeles de la imagen
    VERTICAL_SIZE = 150
    HORIZONTAL_SIZE = 113

    # Cantidad de eigen values a quedarnos
    eigen_n = 25

    # Directorios donde estan las imagenes de cada persona
    person_dir = [k for k in listdir(rootdir) if isdir(join(rootdir))]

    # Tamanios de los arrays que vamos a usar
    size = VERTICAL_SIZE * HORIZONTAL_SIZE
    train_amount = people * train

    ########################## single image ###########################

    image_path = rootdir + '{0}'.format(face_name) + '/{0}.pgm'.format(face_number)
    image = plt.imread(image_path)/255
    result_image = (np.reshape(image, [1, size]))

    ############################# Imagenes que usamos para training ########################
    images = np.zeros([train_amount, size])
    person = np.zeros([train_amount, 1])

    get_images(person_dir, 1, train+1, images, person, 127.5, size)

    # Calculamos la media
    mean_image = np.mean(images, 0)

    # Estandarizar 

    images = [images[k, :] - mean_image for k in
                            range(images.shape[0])]
    result_image = [result_image[k, :] - mean_image for k in
                            range(result_image.shape[0])]

    # Transformaciones segun el paper
    images = np.asarray(images)
    matrix = np.dot(images, images.T)

    # Calculamos los autovalores y autovectores con nuestro propio metodo.
    w, alpha = mhf.get_eigen_from_qr(matrix, 2000)
    lambdas = w

    eigenfaces = np.dot(alpha.T, images)


    training_proyection = np.dot(images, eigenfaces.T)
    sing_proyection = np.dot(result_image, eigenfaces.T)

    # Realizamos el calculo de svc y calculamos la precision segun la cantidad de eigenfaces
    clf = svm.LinearSVC(max_iter=10**8)

    # Nos quedamos solo con los autovectores significativos
    improy      = training_proyection[:, 0:eigen_n]  
    imsingproy  = sing_proyection[:, 0:eigen_n]

    # Entrenando
    clf.fit(np.nan_to_num(improy), person.ravel())

    # Obtenemos Imagen particular
    name_predicted = person_dir[int(clf.predict(np.nan_to_num(imsingproy))[0])]
    print('La cara es de {0}'.format(name_predicted))

    input_image = cv.imread(image_path)
    cv.putText(input_image, name_predicted, (10,30), cv.FONT_HERSHEY_SIMPLEX, 1,(209, 80, 0, 255),2)
    cv.imshow('input_image', input_image)
    cv.waitKey(0)

def pca(rootdir, people, train, test):

    # Ya predifinido las longitudes de pixeles de la imagen
    VERTICAL_SIZE = 150
    HORIZONTAL_SIZE = 113

    # Directorios donde estan las imagenes de cada persona
    person_dir = [k for k in listdir(rootdir) if isdir(join(rootdir))]

    # Tamanios de los arrays que vamos a usar
    size = VERTICAL_SIZE * HORIZONTAL_SIZE
    train_amount = people * train
    test_amount = people * test

   
    ############################# Imagenes que usamos para training ########################
    images = np.zeros([train_amount, size])
    person = np.zeros([train_amount, 1])

    get_images(person_dir, 1, train+1, images, person, 127.5, size)


    ############################# Imagenes que usamos para testing #############################
    image_test = np.zeros([test_amount, size])
    person_test = np.zeros([test_amount, 1])

    get_images(person_dir, train, train+test, image_test, person_test, 127.5, size)

    # Calculamos la media
    mean_image = np.mean(images, 0)

    # Estandarizar 
    images = [images[k, :] - mean_image for k in
                            range(images.shape[0])]
    image_test = [image_test[k, :] - mean_image for k in
                            range(image_test.shape[0])]


    # Conversiones segun el paper
    images = np.asarray(images)
    matrix = np.dot(images, images.T)

    # Calculamos los autovalores y autovectores con nuestro propio metodo.
    w, alpha = mhf.get_eigen_from_qr(matrix, 2000)
    lambdas = w

    eigenfaces = np.dot(alpha.T, images)

    print("Generated eigenfaces")

    training_proyection = np.dot(images, eigenfaces.T)
    test_proyection = np.dot(image_test, eigenfaces.T)

    print("Generated projections")

    # Realizamos el calculo de svc y calculamos la precision segun la cantidad de eigenfaces
    max_eigenfaces = 30
    accs = np.zeros([max_eigenfaces,1])
    accs_sing = np.zeros([max_eigenfaces,1])
    # Linear iteration, set max_iter to avoid convergency errors
    clf = svm.LinearSVC(max_iter=10**7)

    for eigen_n in range(1 ,max_eigenfaces):

        improy      = training_proyection[:, 0:eigen_n]
        imtstproy   = test_proyection[:, 0:eigen_n]

        # Entrenando

        clf.fit(np.nan_to_num(improy), person.ravel())
        # Testeando
        accs[eigen_n] = clf.score(np.nan_to_num(imtstproy), person_test.ravel())
        # Imprimir resultados
        print('#Autovalores: {0}   Precision: {1} %\n'.format(eigen_n, '{0:.2f}'.format(accs[eigen_n][0]*100)))

    x=range(1, max_eigenfaces+1)
    y=(1-accs)*100

    plt.plot(x, y, 'go--', linewidth=2, markersize=12)
    plt.xlabel('Autocaras')
    plt.ylabel('Error')
    plt.title('PCA')
    plt.xticks(np.arange(0, max_eigenfaces+0.001, step=max_eigenfaces/10))
    plt.yticks(np.arange(0, 100+0.001, step=10))
    plt.grid(color='black', linestyle='-', linewidth=0.2)
    plt.show()

def get_images(person_dir, low_limit, high_limit, result_image, result_person, average, size):

    image_num = 0
    per = 0

    for dire in person_dir:
        for m in range(low_limit, high_limit):
            image = (plt.imread(rootdir + dire + '/{}'.format(m) + '.pgm'))/255
            result_image[image_num, :] = (np.reshape(image, [1, size]))
            result_person[image_num,0] = per
            image_num += 1
        per+=1



rootdir = 'data/fotos/'
kernel_degree = 2
people_number = 5
train_number = 4 
test_number = 6


classify_face_by_pca(rootdir, people_number, 6, 'catalina_varela', 10)
#pca(rootdir, people_number, train_number, test_number)
    



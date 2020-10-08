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

def classify_face_by_kpca(rootdir, people, train, face_name, face_number):

    # Ya predifinido las longitudes de pixeles de la imagen
    VERTICAL_SIZE = 150
    HORIZONTAL_SIZE = 113

    # Variables predefinidas por nosotros
    kernel_degree = 2
    kernel_ctx = 1
    kernel_denom = 30
    # Cantidad de eigen values a quedarnos
    eigen_n = 25

    # Directorios donde estan las imagenes de cada persona
    person_dir = [k for k in listdir(rootdir) if isdir(join(rootdir))]

    # Tamanios de los arrays que vamos a usar
    size = VERTICAL_SIZE * HORIZONTAL_SIZE
    train_amount = people * train

    ########################## single image ###########################

    image_path = rootdir + '{0}'.format(face_name) + '/{0}.pgm'.format(face_number)
    image = plt.imread(image_path)
    result_image = (np.reshape(image, [1, size]) - 127.5)/ 127.5

    ############################# Imagenes que usamos para training ########################
    images = np.zeros([train_amount, size])
    person = np.zeros([train_amount, 1])

    get_images(rootdir, person_dir, 1, train+1, images, person, 127.5, size)

    # Aplicamos la transformacion kernel para la conjunto de imagenes de training
    # Es una transformacion polinomial simple
    # TODO: chequear esto
    kernel_matrix= kernel_polynomial_transformation(images, images,kernel_denom, kernel_ctx, kernel_degree)

    # Centrar la matriz del kernel
    unoM = np.ones([train_amount, train_amount]) / train_amount
    kernel_matrix= kernel_matrix- np.dot(unoM, kernel_matrix) - np.dot(kernel_matrix, unoM) + np.dot(unoM, np.dot(kernel_matrix, unoM))

    # Calculamos los autovalores y autovectores con nuestro propio metodo.
    w, alpha = mhf.get_eigen_from_qr(kernel_matrix)
    lambdas = w

    for col in range(alpha.shape[1]):
        alpha[:, col] = alpha[:, col]/np.sqrt(abs(lambdas[col]))

    # Realizamos la proyeccion de las eigenfaces sobre el conjunto de testing
    training_proyection = np.dot(kernel_matrix.T, alpha)
    
    # Transformacion para la imagen singular
    kernel_sing = kernel_polynomial_transformation(result_image, images, kernel_denom, kernel_ctx, kernel_degree)
    unoS = np.ones([1, train_amount]) / train_amount 
    kernel_sing = kernel_sing - np.dot(unoS, kernel_matrix) - np.dot(kernel_sing, unoM) + np.dot(unoS, np.dot(kernel_matrix, unoM))
    # Proyeccion de la imagen singular
    sing_proyection = np.dot(kernel_sing, alpha)

    # Realizamos el calculo de svc y calculamos la precision segun la cantidad de eigenfaces
    clf = svm.LinearSVC(max_iter=10**7)

    # Nos quedamos solo con los autovectores significativos
    improy      = training_proyection[:, 0:eigen_n]  
    imsingproy  = sing_proyection[:, 0:eigen_n]

    # Entrenando
    clf.fit(np.nan_to_num(improy), person.ravel())

    # Obtenemos Imagen particular
    name_predicted = person_dir[int(clf.predict(np.nan_to_num(imsingproy))[0])]
    print('La cara es de {0}'.format(name_predicted))

    input_image = cv.imread(image_path)
    cv.putText(input_image, name_predicted, (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.6,(209, 80, 0, 255), 2)
    cv.imshow('input_image', input_image)
    cv.waitKey(0)

def kpca(rootdir, people, train, test, kernel_denom, kernel_ctx, kernel_degree):

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

    get_images(rootdir, person_dir, 1, train+1, images, person, 127.5, size)


    ############################# Imagenes que usamos para testing #######################

    image_test = np.zeros([test_amount, size])
    person_test = np.zeros([test_amount, 1])

    get_images(rootdir, person_dir, train, train+test, image_test, person_test, 127.5, size)


    # Aplicamos la transformacion kernel para la conjunto de imagenes de training
    # Es una transformacion polinomial simple
    # TODO: chequear esto
    kernel_matrix= kernel_polynomial_transformation(images, images,kernel_denom, kernel_ctx, kernel_degree)

    # Centrar la matriz del kernel
    unoM = np.ones([train_amount, train_amount]) / train_amount
    kernel_matrix= kernel_matrix- np.dot(unoM, kernel_matrix) - np.dot(kernel_matrix, unoM) + np.dot(unoM, np.dot(kernel_matrix, unoM))

    # Calculamos los autovalores y autovectores con nuestro propio metodo.
    w, alpha = mhf.get_eigen_from_qr(kernel_matrix)
    lambdas = w

    for col in range(alpha.shape[1]):
        alpha[:, col] = alpha[:, col]/np.sqrt(abs(lambdas[col]))


    # Realizamos la proyeccion de las eigenfaces sobre el conjunto de training
    training_proyection = np.dot(kernel_matrix.T, alpha)


    # Realizamos la transformacion de kernel sobre el conjunto de prueb
    kernel_matrix_test = kernel_polynomial_transformation(image_test, images, kernel_denom, kernel_ctx, kernel_degree)
    
    # Centramos la matriz
    unoML = np.ones([test_amount, train_amount]) / train_amount 
    kernel_matrix_test = kernel_matrix_test - np.dot(unoML, kernel_matrix) - np.dot(kernel_matrix_test, unoM) + np.dot(unoML, np.dot(kernel_matrix, unoM))

    # Proyectamos sobre el conjunto de testing de las eigenfaces
    test_proyection = np.dot(kernel_matrix_test, alpha)

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
    plt.plot(x, y, 'go--', linewidth=1, markersize=10, color="red")
    plt.xticks(np.arange(0, max_eigenfaces+0.001, step=max_eigenfaces/10))
    plt.yticks(np.arange(0, 100+0.001, step=10))
    plt.xlabel('Cantidad de Autocaras')
    plt.ylabel('% Error')
    plt.grid(color='black', linestyle='-', linewidth=0.2)
    
    plt.show()


def get_images(rootdir, person_dir, low_limit, high_limit, result_image, result_person, average, size):

    image_num = 0
    per = 0

    for dire in person_dir:
        for m in range(low_limit, high_limit):
            image = plt.imread(rootdir + dire + '/{}'.format(m) + '.pgm')
            result_image[image_num, :] = (np.reshape(image, [1, size]) - average)/ average
            result_person[image_num,0] = per
            image_num += 1
        per+=1

def kernel_polynomial_transformation(k1, k2, denom, ctx, degree):
    return (np.dot(k1, k2.T)/denom + ctx ) ** degree

def random_path(rootdir, person_dir, people_number):
    person = random.randint(0, people_number - 1)
    randNum = random.randint(1, 10)
    path = rootdir + person_dir[person] + '/{}'.format(randNum) + '.pgm'

    return path

############################### Testing ##############################3

#rootdir = 'data/fotos/'
#kernel_degree = 2
#kernel_ctx = 1
#kernel_denom = 30
#people_number = 4
#train_number = 4 
#test_number = 6

#kpca(rootdir, people_number, train_number, test_number, kernel_denom, kernel_ctx, kernel_degree)

#classify_face_by_kpca(rootdir, people_number, 4, 'catalina_varela', 8)
    



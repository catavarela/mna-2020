import cv2 as cv
import numpy as np
import Matrix_Handling_Functions as mhf
from os import listdir
from os.path import join, isdir
import matplotlib.pyplot as plt
from sklearn import svm


def classify_face_by_pca(rootdir, people, train, face_name, face_number):
    # Ya predefinido las longitudes de pixeles de la imagen
    VERTICAL_SIZE = 150
    HORIZONTAL_SIZE = 113

    # Cantidad de eigen values a quedarnos
    eigen_n = 25

    # Directorios donde estan las imagenes de cada persona
    person_dir = [k for k in listdir(rootdir) if isdir(join(rootdir, k))]

    # Tamaños de los arrays que vamos a usar
    size = VERTICAL_SIZE * HORIZONTAL_SIZE
    train_amount = people * train

    ############################# Imagenes que usamos para training ########################
    images, person = get_images(rootdir, person_dir, 1, train+1, size, train_amount)

    # Calculamos la media
    mean_image = np.mean(images, 0)

    # Estandarizar
    images = [images[k, :] - mean_image for k in range(images.shape[0])]

    # Transformaciones segun el paper
    images = np.asarray(images)
    matrix = np.dot(images, images.T)

    # Calculamos los autovalores y autovectores con nuestro propio metodo.
    w, alpha = mhf.get_eigen_from_qr(matrix)

    eigenfaces = np.dot(alpha.T, images)

    training_proyection = np.dot(images, eigenfaces.T)

    # Realizamos el calculo de svc y calculamos la precision segun la cantidad de eigenfaces
    clf = svm.LinearSVC(max_iter=10**8)

    # Nos quedamos solo con los autovectores significativos
    improy = training_proyection[:, 0:eigen_n]

    # Entrenando
    clf.fit(np.nan_to_num(improy), person.ravel())

    # Obtenemos Imagen particular
    image_path = '{0}/{1}/{2}.pgm'.format(rootdir, face_name, face_number)
    testing_image = plt.imread(image_path) / 255
    testing_image = (np.reshape(testing_image, [1, size]))
    testing_image = [testing_image[k, :] - mean_image for k in range(testing_image.shape[0])]
    test_proy = np.dot(testing_image, eigenfaces.T)
    test_proy = test_proy[:, 0:eigen_n]

    name_predicted = person_dir[int(clf.predict(np.nan_to_num(test_proy))[0])]
    print('La cara es de {0}'.format(name_predicted))

    input_image = cv.imread(image_path)
    cv.putText(input_image, name_predicted, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (209, 80, 0, 255), 2)
    cv.imshow('input_image', input_image)
    cv.waitKey(0)


def pca(rootdir, people, train, test):

    # Ya predifinido las longitudes de pixeles de la imagen
    VERTICAL_SIZE = 150
    HORIZONTAL_SIZE = 113

    # Directorios donde estan las imagenes de cada persona
    person_dir = [k for k in listdir(rootdir) if isdir(join(rootdir, k))]

    # Tamaños de los arrays que vamos a usar
    size = VERTICAL_SIZE * HORIZONTAL_SIZE
    train_amount = people * train
    test_amount = people * test
   
    ############################# Imagenes que usamos para training ########################
    images, person = get_images(rootdir, person_dir, 1, train+1, size, train_amount)

    # Calculamos la media
    mean_image = np.mean(images, 0)

    # Estandarizar 
    images = [images[k, :] - mean_image for k in range(images.shape[0])]

    # Conversiones segun el paper
    images = np.asarray(images)
    matrix = np.dot(images, images.T)

    # Calculamos los autovalores y autovectores con nuestro propio metodo.
    w, alpha = mhf.get_eigen_from_qr(matrix)
    eigenfaces = np.dot(alpha.T, images)
    training_proyection = np.dot(images, eigenfaces.T)

    ############################# Imagenes que usamos para testing #############################
    image_test, person_test = get_images(rootdir, person_dir, train, train + test, size, test_amount)
    image_test = [image_test[k, :] - mean_image for k in range(image_test.shape[0])]
    test_proyection = np.dot(image_test, eigenfaces.T)

    # Realizamos el calculo de svc y calculamos la precision segun la cantidad de eigenfaces
    max_eigenfaces = 30
    accs = np.zeros([max_eigenfaces, 1])
    # Linear iteration, set max_iter to avoid convergency errors
    clf = svm.LinearSVC(max_iter=10**7)

    for eigen_n in range(1, max_eigenfaces):

        im_train_proy = training_proyection[:, 0:eigen_n]
        im_test_proy = test_proyection[:, 0:eigen_n]

        # Entrenando
        clf.fit(np.nan_to_num(im_train_proy), person.ravel())
        # Testeando
        accs[eigen_n] = clf.score(np.nan_to_num(im_test_proy), person_test.ravel())
        # Imprimir resultados
        print('#Autovalores: {0}   Precision: {1:.2f} %\n'.format(eigen_n, accs[eigen_n][0]*100))

    x = range(1, max_eigenfaces+1)
    y = (1-accs)*100
    plt.plot(x, y, 'go--', linewidth=1, markersize=10, color="red")
    plt.xticks(np.arange(0, max_eigenfaces+0.001, step=max_eigenfaces/10))
    plt.yticks(np.arange(0, 100+0.001, step=10))
    plt.xlabel('Cantidad de Autocaras')
    plt.ylabel('% Error')
    plt.grid(color='black', linestyle='-', linewidth=0.2)
    
    plt.show()


def get_images(root_dir, person_dir, low_limit, high_limit, size, ammount):
    result_image = np.zeros([ammount, size])
    result_person = np.zeros([ammount, 1])

    image_num = 0
    per = 0

    for dire in person_dir:
        for m in range(low_limit, high_limit):
            image = (plt.imread('{0}/{1}/{2}.pgm'.format(root_dir, dire, m)))/255
            result_image[image_num, :] = (np.reshape(image, [1, size]))
            result_person[image_num, 0] = per
            image_num += 1
        per += 1
    return result_image, result_person


##################### Testing ########################

#rootdir = 'data/fotos'
#kernel_degree = 2
#people_number = 4
#train_number = 4 
#test_number = 6


#classify_face_by_pca(rootdir, people_number, 6, 'catalina', 2)
#pca(rootdir, people_number, train_number, test_number)

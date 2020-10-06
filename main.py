import cv2 as cv
import glob
import numpy as np
import random
import time
from sklearn import svm
from Matrix_Handling_Functions import get_covariance_eigenvectors
import facedetection as fd

# Source: https://www.mitpressjournals.org/doi/pdf/10.1162/jocn.1991.3.1.71


def generate_eigenfaces(paths, keep_percentage=0.5):
    faces = fd.get_faces_as_rows(paths)

    # Calculo la media
    avg = np.mean(faces, 0)
    # Resto la media
    faces_min_avg = faces - avg

    # Calculo los autovectores de la covarianza
    eigenfaces = get_covariance_eigenvectors(faces_min_avg)

    # Me quedo con los primeros
    return eigenfaces[0:int(len(eigenfaces) * keep_percentage)], avg


def get_projected_images(images, eigenfaces):
    return images.dot(eigenfaces.transpose())


def main():
    images_to_use = 200

    # ENTRENAMIENTO
    # Obtengo eigenfaces y promedio
    training_images = glob.glob('data/**/*0[1-3].jpg', recursive=True)
    training_images.sort()
    training_images = training_images[0:images_to_use]
    partial_start = start = time.time()
    eigenfaces, mean = generate_eigenfaces(training_images, 1)
    partial_end = time.time()
    print('Eigenfaces generated in:', partial_end - partial_start, 's')

    # Show mean face
    # cv.imshow('Mean face', (mean.reshape((150, 150))*255).astype('uint8'))
    # cv.waitKey(0)

    # Show all eigenfaces
    # for i, eigenface in enumerate(eigenfaces):
    #     cv.imshow('Eigenface ' + str(i), (eigenface.reshape((150, 150))*255))
    #     cv.waitKey(0)

    # Entreno Support Vector Machine (SVM)
    partial_start = time.time()
    svc = svm.LinearSVC()
    projected_training = get_projected_images(fd.get_faces_as_rows(training_images), eigenfaces)
    svc.fit(projected_training, training_images)
    partial_end = time.time()
    print('SVC trained in:', partial_end - partial_start)
    print('Training completed in:', partial_end - start)

    # PRUEBA
    images = glob.glob('data/**/*04.jpg', recursive=True)
    images.sort()
    random.seed()
    rand = int(random.random() * (images_to_use // 3))
    test_image_path = images[rand]
    print('Imagen de prueba:', test_image_path)
    partial_start = time.time()
    test_face = fd.get_face_as_row(test_image_path) - mean
    projected_test = get_projected_images(test_face, eigenfaces)
    path = svc.predict([projected_test])[0]
    partial_end = end = time.time()

    print('La imagen es:', path)
    print('Tardé', partial_end - partial_start, 's en encontrarla')
    print('Tardé en total', end - start, 's')

    # Muestro la imagen de prueba y su matcheo
    cv.imshow('test', cv.imread(test_image_path))
    cv.imshow('result', cv.imread(path))
    cv.waitKey(0)


if __name__ == '__main__':
    main()

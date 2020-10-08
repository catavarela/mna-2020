import os

import glob
import random
import time
from sklearn import svm
from facedetection import FaceRecognition


# Source: https://www.mitpressjournals.org/doi/pdf/10.1162/jocn.1991.3.1.71
from pca import PCA


def get_projected_images(centered_images, eigenfaces):
    return eigenfaces.dot(centered_images)


def get_name_from_path(path):
    return os.path.basename(os.path.dirname(path))


def main():
    face_recognition = FaceRecognition()
    images_to_use = 200

    # ENTRENAMIENTO
    # Obtengo eigenfaces y promedio
    training_images = glob.glob('data/**/*0[1-3].jpg', recursive=True)
    training_images.sort()
    if len(training_images) == 0:
        print('No images to load')
        exit(1)
    if len(training_images) < images_to_use:
        images_to_use = len(training_images)

    images_to_use = len(training_images)
    training_images = training_images[0:images_to_use]

    training_images_name = {
        path_name: get_name_from_path(path_name) for path_name in training_images
    }

    partial_start = start = time.time()
    pca = PCA(face_recognition.get_faces_as_columns(training_images))
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
    projected_training = get_projected_images(pca.centered, pca.eigenfaces)
    svc.fit(projected_training, [training_images_name[path_name] for path_name in training_images])
    partial_end = time.time()
    print('SVC trained in:', partial_end - partial_start)
    print('Training completed in:', partial_end - start)

    # PRUEBA
    images = glob.glob('data/**/*04.jpg', recursive=True)
    images.sort()
    random.seed()
    # rand = int(random.random() * (images_to_use // 3))
    # test_image_path = images[rand]
    # print('Imagen de prueba:', test_image_path)
    partial_start = time.time()

    success = 0
    for i in range(0, len(images)):
        test_face = face_recognition.get_face_as_column(images[i]) - pca.mean
        projected_test = get_projected_images(test_face, pca.eigenfaces)
        name = svc.predict(projected_test.T)[0]
        if name == get_name_from_path(images[i]):
            success += 1
    partial_end = end = time.time()

    # path = glob.glob('data/**/' + name + '/**/*0[1-3].jpg', recursive=True)[0]
    # print('El nombre es: ', name, '. Cargo path: ', path)
    print('Success: ', success, ' de: ', len(images), ' a: ', (success / len(images)) * 100, '%')
    print('Tardé', partial_end - partial_start, 's en encontrarlo')
    print('Tardé en total', end - start, 's')
    #
    # # Muestro la imagen de prueba y su matcheo
    # cv.imshow('test', cv.imread(test_image_path))
    # cv.imshow('result', cv.imread(path))
    # cv.waitKey(0)


if __name__ == '__main__':
    main()

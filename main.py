import cv2 as cv
import glob
import numpy as np
import random
import time
import facedetection as fd

# Source: https://www.mitpressjournals.org/doi/pdf/10.1162/jocn.1991.3.1.71


def get_face_as_row(path, side_length=150):
    face = cv.imread(path)
    face = fd.get_face(face, side_length, side_length)
    face = cv.cvtColor(face, cv.COLOR_RGB2GRAY)
    face = np.reshape(face, side_length * side_length)
    return face.astype('uint8')


def get_face_as_column(path, side_length=150):
    return get_face_as_row(path, side_length)[:, np.newaxis]


def get_faces_as_columns(paths, side_length=150, images_to_use=100):
    faces = np.zeros((images_to_use, side_length * side_length), dtype=np.uint8)
    for i in range(0, images_to_use):  # TODO: cambiar el images_to_use por len(faces)
        faces[i] = get_face_as_row(paths[i], side_length)
    return np.transpose(faces)


def generate_eigenfaces(paths, keep_percentage=0.5, images_to_use=100):
    faces = get_faces_as_columns(paths, images_to_use=images_to_use)

    # Calculo la media
    avg = np.mean(faces, 1)[:, np.newaxis]
    avg = avg.astype('uint8')
    # Resto la media
    faces_min_avg = faces - avg

    # Calculo los mayores autovectores de la covarianza usando el truco del paper
    L = faces_min_avg.transpose() @ faces_min_avg
    eigval, L_eigvec = np.linalg.eig(L)  # TODO: usar nuestra funcion de autovals y autovecs
    # Quiero los autovectores ordenados por mayor autovalor
    v = [vec for val, vec in sorted(zip(eigval, L_eigvec), reverse=True)]
    u = np.zeros((len(v), len(faces_min_avg)))  # Autovectores de la covarianza
    for l in range(len(v)):
        for k in range(len(v[l])):
            u[l] += v[l][k] * faces_min_avg[:, k]  # Formula 6

    # u = autovectores de la covarianza
    # Me quedo solo con un porcentaje de autovectores (los de mayor autovalor)
    u = u[0:int(len(u) * keep_percentage), :]
    return u.transpose().astype('uint8'), avg


def get_weights(face, eigenfaces, avg):
    weights = np.zeros(eigenfaces.shape[1])
    for k in range(eigenfaces.shape[1]):
        weights[k] = float(eigenfaces[:, k] @ (face - avg))
    return weights


def get_projected_image(eigenfaces, weights):
    image = np.zeros((len(eigenfaces), 1), dtype=np.uint8)
    for i in range(eigenfaces.shape[1]):
        image += eigenfaces[:, i][:, np.newaxis] * weights[i]
    return image


images_to_use = 100
start = time.time()

# ENTRENAMIENTO
training_images = glob.glob('data/**/*0[1-3].jpg', recursive=True)
training_images.sort()
eigenfaces, avg = generate_eigenfaces(training_images, images_to_use=images_to_use)
print('Eigenfaces generated in:', time.time() - start, 's')

weights = []

partial_time = time.time()
for i in range(0, images_to_use):  # TODO: cambiar el images_to_use por len(images)
    face = get_face_as_column(training_images[i])
    weights.append(get_weights(face, eigenfaces, avg))
print('Weights calculated in:', time.time() - partial_time)
print('Training completed in:', time.time() - start)

# PRUEBA
images = glob.glob('data/**/*4.jpg', recursive=True)
images.sort()
random.seed()
rand = int(random.random() * (images_to_use//3))
test_image_path = images[rand]
print('Imagen de prueba:', test_image_path)
partial_time = time.time()
test_face = get_face_as_column(test_image_path)
test_weight = get_weights(test_face, eigenfaces, avg)
# Calculo distancias
min_distance = -1
min_i = 0
accum = 0
for index, weight in enumerate(weights):
    difference = []
    for i in range(0, len(weight)):
        difference.append(weight[i] - test_weight[i])
    distance = np.linalg.norm(difference)
    if distance < min_distance or min_distance == -1:
        min_distance = distance
        min_i = index
end = time.time()
print('La imagen es:', training_images[min_i])
print('Tardé', time.time() - partial_time, 's en encontrarla')
print('Tardé en total', end - start, 's')
cv.imshow('test', cv.imread(test_image_path))
cv.imshow('result', cv.imread(training_images[min_i]))
cv.waitKey(0)

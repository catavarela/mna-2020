import cv2 as cv
import glob
import numpy as np
from itertools import chain
import random
import time
import facedetection as fd

images_to_use = 100
# Source: http://ijarcet.org/wp-content/uploads/IJARCET-VOL-1-ISSUE-9-135-139.pdf


def generate_eigenfaces(root_dir, keep_percentage=0.5):
    images = glob.glob(root_dir + '/**/*0[1-3].jpg', recursive=True)
    images.sort()
    faces = []
    for i in range(0, images_to_use):  # TODO: cambiar el images_to_use por len(faces)
        # Step 1
        face = cv.imread(images[i])
        face = fd.get_face(face)
        face = cv.cvtColor(face, cv.COLOR_RGB2GRAY)
        # Step 2
        faces.append(list(chain.from_iterable(face)))
    # Step 3
    avg = np.mean(faces, 0)
    # Step 4
    faces_not_average = faces
    for i, face in enumerate(faces_not_average):
        faces_not_average[i] -= avg
    faces_not_average = np.array(faces_not_average).transpose()
    # Step 5
    # Al final no se usa la covarianza asi que no la calculo
    # Step 6
    # TODO: usar nuestra funcion de autovals y autovecs
    eigval, l_eigvec = np.linalg.eig(faces_not_average.transpose() @ faces_not_average)
    # Quiero los eigvec ordenados por mayor eigval
    ordered_l_eigvec = [vec for val, vec in sorted(zip(eigval, l_eigvec), reverse=True)]
    ordered_eigvec = []
    v = np.array(ordered_l_eigvec)  # solo por paralelismo con el paper
    for l in range(len(v)):
        u_l = np.zeros((len(faces_not_average)))
        for k in range(len(v[l])):
            u_l += v[l][k] * faces_not_average[:, k]
        ordered_eigvec.append(np.array(u_l).transpose())
    # Step 7
    ordered_eigvec = np.array(ordered_eigvec)
    return ordered_eigvec[0:int(len(ordered_eigvec) * keep_percentage), :], avg


def get_weights(face, eigenfaces, avg):
    face = np.array(face).reshape(1, -1)[0]
    weights = []
    for i in range(0, len(face)):
        face[i] -= avg[i]
    for i in range(0, len(eigenfaces)):
        weights.append(float(eigenfaces[i] @ face.reshape(-1, 1)))
    return np.array(weights)


start = time.time()
eigenfaces, avg = generate_eigenfaces('data')
print('Eigenfaces generated')
print('Generated in:', time.time() - start, 's')
weights = []
images1 = glob.glob('data/**/*0[1-3].jpg', recursive=True)
images1.sort()
for i in range(0, images_to_use):  # TODO: cambiar el images_to_use por len(images)
    face = cv.imread(images1[i])
    face = fd.get_face(face)
    face = cv.cvtColor(face, cv.COLOR_RGB2GRAY)
    weights.append(get_weights(face, eigenfaces, avg))
print('Weights calculated')
images2 = glob.glob('data/**/*4.jpg', recursive=True)
images2.sort()
random.seed()
rand = int(random.random() * (images_to_use//3))
print('Imagen de prueba:', images2[rand])
test_face = cv.imread(images2[rand])
test_face = fd.get_face(test_face)
test_face = cv.cvtColor(test_face, cv.COLOR_RGB2GRAY)
test_weight = np.array(get_weights(test_face, eigenfaces, avg))
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
print('La imagen es:', images1[min_i])
print('Tardé', end - start, 's')
cv.imshow('test', cv.imread(images2[rand]))
cv.imshow('result', cv.imread(images1[min_i]))
cv.waitKey(0)

import cv2 as cv
import glob
import numpy as np
from itertools import chain
import random
import time

images_to_use = 500
# Source: http://ijarcet.org/wp-content/uploads/IJARCET-VOL-1-ISSUE-9-135-139.pdf
def generate_eigenfaces(root_dir, keep_percentage=0.5):
    images = glob.glob(root_dir + '/**/*.jpg', recursive=True)
    faces = []
    for i in range(0, images_to_use):  # TODO: cambiar el images_to_use por len(faces)
        # Step 1
        face = cv.imread(images[i])
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
    l = faces_not_average.transpose() @ faces_not_average
    eigval, l_eigvec = np.linalg.eig(l)  # TODO: usar nuestra funcion de autovals y autovecs
    # Quiero los eigvec ordenados por mayor eigval
    ordered_l_eigvec = [vec for val, vec in sorted(zip(eigval,l_eigvec), reverse=True)]
    ordered_eigvec = faces_not_average @ ordered_l_eigvec
    # Step 7
    sliced_eigvec = ordered_eigvec.transpose()
    return sliced_eigvec[0:int(len(sliced_eigvec) * keep_percentage), :], avg


def get_weights(face, eigenfaces, avg):
    face = np.array(list(chain.from_iterable(face)))
    weights = []
    for i in range(0, len(face)):
        face[i] -= avg[i]
    for i in range(0, len(eigenfaces)):
        weights.append(float(eigenfaces[i] @ face.reshape(-1, 1)))
    return weights


start = time.time()
eigenfaces, avg = generate_eigenfaces('data')
print('Eigenfaces generated')
weights = []
images = glob.glob('data/**/*.jpg', recursive=True)
for i in range(0, images_to_use):  # TODO: cambiar el images_to_use por len(images)
    face = cv.imread(images[i])
    face = cv.cvtColor(face, cv.COLOR_RGB2GRAY)
    weights.append(get_weights(face, eigenfaces, avg))
print('Weights calculated')
random.seed()
rand = int(random.random() * len(eigenfaces))
print('Imagen de prueba:', images[rand])
test_face = cv.imread(images[rand])
test_face = cv.cvtColor(test_face, cv.COLOR_RGB2GRAY)
test_weight = np.array(get_weights(test_face, eigenfaces, avg))
min_distance = np.linalg.norm(test_weight - weights[0])
min_i = 0
for i, w in enumerate(weights):
    aux = np.linalg.norm(test_weight - w)
    if aux < min_distance:
        min_distance = aux
        min_i = i
print('La imagen es:', images[min_i])
end = time.time()
print('Tardé', end - start, 's')

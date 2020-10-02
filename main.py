import cv2 as cv
import glob
import numpy as np
from itertools import chain


# Source: http://ijarcet.org/wp-content/uploads/IJARCET-VOL-1-ISSUE-9-135-139.pdf
def generate_eigenfaces(root_dir, keep_percentage=0.5):
    images = glob.glob(root_dir + '/**/*.jpg', recursive=True)
    faces = []
    for i in range(0,1000): # TODO: cambiar el 1000 por len(faces)
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
    eigval, l_eigvec = np.linalg.eig(l) # TODO: usar nuestra funcion de autovals y autovecs
    ordered_l_eigvec = [vec for val,vec in sorted(zip(eigval,l_eigvec), reverse=True)] # Quiero los eigvec ordenados por mayor eigval
    ordered_eigvec = faces_not_average @ ordered_l_eigvec
    # Step 7
    sliced_eigvec = ordered_eigvec.transpose()
    return sliced_eigvec[0:int(len(sliced_eigvec) * keep_percentage),:], avg

def get_weigths(face, eigenfaces, avg):
    face = np.array(list(chain.from_iterable(face)))
    weights = []
    for i in range(0, len(face)):
        face[i] -= avg[i]
    for i in range(0, len(eigenfaces)):
        weights.append(float(eigenfaces[i] @ face.reshape(-1,1)))
    return weights

eigenfaces, avg = generate_eigenfaces('data')
face = cv.imread('data/image.jpg')
face = cv.cvtColor(face, cv.COLOR_RGB2GRAY)
weigths = get_weigths(face, eigenfaces, avg)
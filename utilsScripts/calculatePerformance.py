import sys
import time
import cv2 as cv
import numpy as np
sys.path.append("../")
import Matrix_Handling_Functions as mhf

# read image file
img = cv.imread("../data/image.jpg")
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# measure performance of eigen calculation
start_time = time.time()
eigval,eigvec = mhf.getEigenFromQR(img,30)
end_time = time.time()
print("Our implementation:", end_time - start_time )

# print numpy's version as a comparison point
start_time = time.time()
eigval,eigvec = np.linalg.eig(img)
end_time = time.time()
print( "Numpy's implementation" , end_time - start_time )

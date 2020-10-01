import numpy as np
import QR_Householder as qr

def getStandarized(A):
  Z = (A - np.mean(A, axis=0)) / np.std(A, axis=0, ddof=1)
  return Z
  
def getCovariance(A, Z):
  Cz = (1/(len(A)-1)) * Z.transpose() @ Z
  return Cz

# TODO: check how eigvec is calculated
def getEigenFromQR(A,iterations=50):
  Q,R = qr.qr_decomp(A) 
  A = np.transpose(Q) @ A @ Q
  eigvec = Q
  for i in range(0,iterations-1):
    Q,R = qr.qr_decomp(A)
    A = np.transpose(Q) @ A @ Q
    eigvec = eigvec @ Q
  Q,R = qr.qr_decomp(A)
  eigvec = eigvec @ Q
  eigval = np.diag(A)
  return eigval,eigvec
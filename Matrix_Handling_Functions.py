import numpy as np

def getStandarized(A):
  Z = (A - np.mean(A, axis=0)) / np.std(A, axis=0, ddof=1)
  return Z
  
def getCovariance(A, Z):
  Cz = (1/(len(A)-1)) * Z.transpose() @ Z
  return Cz
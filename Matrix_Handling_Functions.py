import numpy as np
import QR_Householder as qr

def get_standarized(A):
    Z = (A - np.mean(A, axis=0)) / np.std(A, axis=0, ddof=1)
    return Z


def get_covariance(A, Z):
    cz = (1 / (len(A) - 1)) * Z.transpose() @ Z
    return cz


def get_eigen_from_qr(A):
    Q, R = qr.qr_decomp(A)
    A = np.transpose(Q) @ A @ Q
    prev_eigvec = eigvec = Q

    error = 1
    while error > 0.1:
        Q, R = qr.qr_decomp(A)
        A = np.transpose(Q) @ A @ Q
        eigvec = eigvec @ Q
        error = np.linalg.norm(eigvec + prev_eigvec)
        prev_eigvec = eigvec

    Q, R = qr.qr_decomp(A)
    eigvec = eigvec @ Q
    eigval = np.diag(A)
    return eigval, eigvec


def average_matrix(matrices):
    total_sum = np.zeros((len(matrices[0]), len(matrices[0][0])))
    qty=0
    for image in matrices:
        total_sum += image
        qty+=1
    print(total_sum)
    print(qty)
    ans = np.zeros((len(total_sum), len(total_sum[0])), dtype=np.uint8)
    for i in range(0, len(total_sum)):
        for j in range(0, len(total_sum[0])):
            ans[i][j] = total_sum[i][j] / qty
    return ans
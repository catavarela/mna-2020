import numpy as np
import QR_Householder as qr


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

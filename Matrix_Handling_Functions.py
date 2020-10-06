import numpy as np
import QR_Householder as qr


# TODO: check how eigvec is calculated
def get_eigen_from_qr(A, iterations=50):
    Q, R = qr.qr_decomp(A)
    A = np.transpose(Q) @ A @ Q
    eigvec = Q
    for i in range(0, iterations - 1):
        Q, R = qr.qr_decomp(A)
        A = np.transpose(Q) @ A @ Q
        eigvec = eigvec @ Q
    Q, R = qr.qr_decomp(A)
    eigvec = eigvec @ Q
    eigval = np.diag(A)
    return eigval, eigvec


def get_covariance_eigenvectors(A):
    """Dado una matriz de NxM, siendo N>>M, devuelve los M autovectores significativos (con autovalor distinto a 0),
    ordenados por mayor autovalor"""
    # Calculo los mayores autovectores de la covarianza usando el truco del paper
    L = A.dot(A.transpose())  # MxM

    # Calculo los autovectores de L
    eigval, L_eigvec = np.linalg.eig(L)  # TODO: usar nuestra funcion de autovals y autovecs
    # eigval, L_eigvec = get_eigen_from_qr(L)

    # Los paso a autovectores de la covarianza
    # Demo de esto en la pagina 2: http://ijarcet.org/wp-content/uploads/IJARCET-VOL-1-ISSUE-9-135-139.pdf
    # Transponemos porque en el paper ordena una foto por columna y nosotros hacemos una por fila
    eigfaces = A.transpose().dot(L_eigvec.transpose()).transpose()

    # Los ordeno por mayor |autovalor|
    eigfaces = [eigface for val, eigface in sorted(zip(np.absolute(eigval), eigfaces), reverse=True)]

    # Normalizo las eigenfaces dividiendo por su norma
    eigfaces = np.divide(eigfaces, np.linalg.norm(eigfaces, axis=1).reshape((-1, 1)))
    return eigfaces

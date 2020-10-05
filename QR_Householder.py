# Descomposicion QR usando la transformación Householder
# Disclaimer: ver si cuando calculemos los autovalores y autovectores
# el hecho de que R no tenga ceros "perfectos" en el triángulo hace
# que nos de algún error. Además a veces los signos son opuestos en algunas
# columnas de Q obtenido con este algoritmo que con np.linalg.qr. Según
# lo que leí no debería haber problema (medio que da lo mismo). Igual
# si tenemos que cambiar algo de esto ver:
# https://stackoverflow.com/questions/53489237/how-can-you-implement-householder-based-qr-decomposition-in-python

import numpy as np


def householder(a):
    u = a.copy()
    # TODO: check change from .. += -- to .. = ... + ---
    u[0] = u[0] + np.copysign(np.linalg.norm(a), a[0])
    return np.identity(len(u)) - ((2 * (u @ u.transpose())) / (u.transpose() @ u))


def qr_decomp(A):
    m, n = A.shape
    Q = np.identity(m)
    R = A.copy()

    min_dim = min(m, n)
    for k in range(0, min_dim):
        h = householder(R[k:, k, np.newaxis])
        H = np.identity(m)
        H[k:, k:] = h
        Q = Q @ H
        R = H @ R

    return Q[:, :n], np.triu(R[:m])

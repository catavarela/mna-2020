import numpy as np


class PCA:
    # Images es una matriz col imagenes
    def __init__(self, images):
        self.images = images
        self.mean = np.mean(images)
        self.centered = images - self.mean
        self.eigenvalues, self.eigenfaces = self._get_covariance_eigenvectors()

    def _get_covariance_eigenvectors(self, keep_percentage=0.3):
        """Dado una matriz de MxN, siendo M>>N, devuelve los N autovectores significativos (con autovalor distinto a 0),
        ordenados por mayor autovalor"""
        # Calculo los mayores autovectores de la covarianza
        L = self.centered.transpose().dot(self.centered)  # MxM

        # Calculo los autovectores de L
        eigval, L_eigvec = np.linalg.eig(L)  # TODO: usar nuestra funcion de autovals y autovecs
        # eigval, L_eigvec = get_eigen_from_qr(L)

        # Los paso a autovectores de la covarianza
        # Demo de esto en la pagina 2: http://ijarcet.org/wp-content/uploads/IJARCET-VOL-1-ISSUE-9-135-139.pdf
        eigfaces = self.centered.dot(L_eigvec)

        # Los ordeno por mayor |autovalor|
        eigfaces = [eigface for val, eigface in sorted(zip(np.absolute(eigval), eigfaces), reverse=True)]

        # Normalizo las eigenfaces dividiendo por su norma
        eigfaces = np.divide(eigfaces, np.linalg.norm(eigfaces, axis=1).reshape((-1, 1)))

        # Paso de autovectores de L a autovectores de la matriz centrada
        eigfaces = self.centered.dot(eigfaces)
        floor = int(len(eigfaces) * keep_percentage)
        return eigval[0:floor if floor == len(eigfaces) else floor + 1], eigfaces[0:floor if floor == len(eigfaces) else floor + 1]

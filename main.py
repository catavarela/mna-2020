import pca as pca
import kpca as kpca

def main():
    rootdir = 'fotos/'
    kernel_degree = 2
    kernel_ctx = 1
    kernel_denom = 30
    people_number = 4
    train_number = 4 
    test_number = 6

    pca.classify_face_by_pca(rootdir, people_number, 6, 'catalina', 2)
    kpca.classify_face_by_kpca(rootdir, people_number, 4, 'catalina', 8)

if __name__ == '__main__':
    main()

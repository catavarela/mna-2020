import pca as pca
import kpca as kpca

def main():
    rootdir = 'fotos'
    
    kernel_degree = 2
    kernel_ctx = 1
    kernel_denom = 30
    
    people_number = 6

    train_number_kpca = 6 
    test_number_kpca = 4

    train_number_pca = 3 
    test_number_pca = 7

    action_op = input('Elegi la accion que queres realizar:\n 1 -> Testear un metodo(pca o kpca) \n 2 -> Clasificar una imagen \n Eleccion(1 o 2): ')
    method_op = input('Elegi el metodo:\n 1 -> PCA \n 2 -> KPCA \n Eleccion(1 o 2): ')

    if action_op == '1':
        if method_op == '1':
            pca.pca(rootdir, people_number, train_number_pca, test_number_pca)
        elif method_op == '2':
            kpca.kpca(rootdir, people_number, train_number_kpca, test_number_kpca, kernel_denom, kernel_ctx, kernel_degree)
        else:
            print("Invalid method")
            exit(1)

    elif action_op == '2':
        name_face = input('Introducir nombre de la persona.\nOpciones:\tagustin\n\t\taugusto\n\t\tcatalina\n\t\tfrancisco\n\t\tguido\n\t\tnicolas\nSu eleccion: ')
        number_face = input('Introducir numero de la foto[1-10]: ')

        if method_op == '1':
            pca.classify_face_by_pca(rootdir, people_number, 6, name_face, number_face)
        elif method_op == '2':
            kpca.classify_face_by_kpca(rootdir, people_number, 4, name_face, number_face)
        else:
            print("Invalid method")
            exit(1)
    else:
        print("Invalid action")
        exit(1)

if __name__ == '__main__':
    main()

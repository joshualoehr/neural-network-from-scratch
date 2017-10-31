#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

VectorXi OutputToClass(MatrixXi output_vectors);
MatrixXi ClassToOutput(VectorXi class_vector);

int main()
{
    MatrixXi m(5,2);
    m(0,1) = 1;
    m(1,0) = 1;
    m(2,1) = 1;
    m(3,1) = 1;
    m(4,0) = 1;

    VectorXi class_vector = OutputToClass(m);
    cout << "output matrix: " << endl << m << endl;
    cout << "class vector: " << endl << class_vector << endl;
    cout << "output matrix: " << endl << ClassToOutput(class_vector) << endl;
    return 0;
}

VectorXi OutputToClass(MatrixXi output_vectors)
{
    int num_rows = output_vectors.rows();
    int num_cols = output_vectors.cols();
    VectorXi class_vector(num_rows);

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++) {
            if (output_vectors(i, j) == 1) {
                class_vector(i) = j + 1;
                break;
            }
        }
    }

    return class_vector;
}

MatrixXi ClassToOutput(VectorXi class_vector)
{
    int num_rows = class_vector.rows();
    int num_cols = class_vector.maxCoeff();
    MatrixXi output_vectors = MatrixXi::Constant(num_rows, num_cols, 0);

    for (int i = 0; i < num_rows; i++) {
        int class_val = class_vector(i);
        output_vectors(i, class_val - 1) = 1;
    }

    return output_vectors;
}

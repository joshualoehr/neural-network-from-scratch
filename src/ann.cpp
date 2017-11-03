#include "proto.hpp"

struct DataSets
{
    int input_count;
    int output_count;
    DataSet training_set;
    DataSet validation_set;
    DataSet test_set;
};

int main()
{
    // MatrixXi csv_contents = ParseInputCSV("mushrooms.csv");
    MatrixXf m = MatrixXf::Random(5,5) + MatrixXf::Constant(5,5,1.0f);
    cout << m << endl;
    cout << Sigmoid(m) << endl;
    return 0;
}

MatrixXf Sigmoid(MatrixXf X)
{
    MatrixXf ones = MatrixXf::Constant(X.rows(), X.cols(), 1.0f);
    return ((X * -1).array().exp().matrix() + ones).cwiseInverse();
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
    MatrixXi output_vectors = MatrixXi::Zero(num_rows, num_cols);

    for (int i = 0; i < num_rows; i++) {
        int class_val = class_vector(i);
        output_vectors(i, class_val - 1) = 1;
    }

    return output_vectors;
}

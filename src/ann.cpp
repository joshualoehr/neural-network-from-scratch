#include "proto.hpp"

MatrixXf FeedForward(MatrixXf inputs, MatrixXf weights, VectorXf bias);

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
    // MatrixXf csv_contents = ParseInputCSV(INPUT_FILENAME_SMALL);
    // cout << csv_contents << endl;
    // cout << endl << endl;

    MatrixXf m(2,1);
    m(0,0) = 1;
    m(1,0) = 0;
    cout << "inputs: " << endl << m << endl << endl;

    MatrixXf weights(2,1);
    weights(0,0) = 20;
    weights(1,0) = 20;
    cout << "weights: " << endl << weights << endl << endl;

    VectorXf bias = VectorXf::Constant(m.cols(), -10);
    cout << "bias: " << endl << bias << endl << endl;

    MatrixXf ff1 = FeedForward(m, weights, bias);
    cout << ff1.rows() << "x" << ff1.cols() << endl;
    cout << ff1 << endl;
    return 0;
}

MatrixXf FeedForward(MatrixXf inputs, MatrixXf weights, VectorXf bias)
{
    return Sigmoid(weights.transpose() * inputs + bias);
}

MatrixXf Sigmoid(MatrixXf X)
{
    MatrixXf ones = MatrixXf::Constant(X.rows(), X.cols(), 1.0f);
    return ((X * -1).array().exp().matrix() + ones).cwiseInverse();
}

VectorXi OutputToClass(MatrixXf output_vectors)
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

MatrixXf ClassToOutput(VectorXi class_vector)
{
    int num_rows = class_vector.rows();
    int num_cols = class_vector.maxCoeff();
    MatrixXf output_vectors = MatrixXf::Zero(num_rows, num_cols);

    for (int i = 0; i < num_rows; i++) {
        int class_val = class_vector(i);
        output_vectors(i, class_val - 1) = 1.0f;
    }

    return output_vectors;
}

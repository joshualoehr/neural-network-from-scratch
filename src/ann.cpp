#include "proto.hpp"

MatrixXf BackProp(MatrixXf inputs, MatrixXf weights, VectorXf bias, float lr);
float SquaredError(VectorXf outputs, VectorXf target_outputs);
MatrixXf FeedForward_Matrix(MatrixXf inputs, MatrixXf weights, VectorXf bias);
VectorXf FeedForward_Vector(VectorXf inputs, MatrixXf weights, VectorXf bias);
MatrixXf InitWeights(int rows, int cols, float max_weight);

struct DataSets
{
    int input_count;
    int output_count;
    DataSet training_set;
    DataSet validation_set;
    DataSet test_set;
};

string mdim(MatrixXf m)
{
    stringstream ss;
    ss << m.rows() << "x" << m.cols();
    return ss.str();
}

string vdim(VectorXf v)
{
    stringstream ss;
    ss << v.rows() << "x" << v.cols();
    return ss.str();
}

int main()
{
    // Reset the random generator seed
    srand((unsigned int) time(0));

    // MatrixXf csv_contents = ParseInputCSV(INPUT_FILENAME_SMALL);
    // cout << csv_contents << endl;
    // cout << endl << endl;

    MatrixXf X(4,2);
    X(0,0) = 0;
    X(0,1) = 0;
    X(1,0) = 0;
    X(1,1) = 1;
    X(2,0) = 1;
    X(2,1) = 0;
    X(3,0) = 1;
    X(3,1) = 1;
    cout << "inputs: " << endl << X << endl << endl;

    int L_0 = X.cols();

    MatrixXf Y(4,1);
    Y(0,0) = 0;
    Y(1,0) = 1;
    Y(2,0) = 1;
    Y(3,0) = 0;
    cout << "target output: " << endl << Y << endl << endl;

    int L_1 = 2;
    int L_2 = 1;

    MatrixXf W_1 = InitWeights(L_0, L_1, 20);
    cout << "W_1: " << endl << W_1 << endl << endl;
    VectorXf b_1 = VectorXf::Ones(L_1);

    MatrixXf Z_1 = FeedForward_Matrix(X, W_1, b_1);
    cout << "Z_1: " << endl << Z_1 << endl << endl;

    MatrixXf A_1 = Sigmoid(Z_1);
    cout << "A_1: " << endl << A_1 << endl << endl;

    MatrixXf W_2 = InitWeights(L_1, L_2, 20);
    cout << "W_2: " << endl << W_2 << endl << endl;
    VectorXf b_2 = VectorXf::Ones(L_2);

    MatrixXf Z_2 = FeedForward_Matrix(A_1, W_2, b_2);
    cout << "Z_2: " << endl << Z_2 << endl << endl;

    MatrixXf A_2 = Sigmoid(Z_2);
    cout << "A_2: " << endl << A_2 << endl << endl;

    cout << "error: " << SquaredError(A_2, Y) << endl;

    return 0;
}

MatrixXf BackProp(MatrixXf inputs, MatrixXf weights, VectorXf bias, float lr)
{


    return weights;
}

float SquaredError(VectorXf outputs, VectorXf targets)
{
    assert(outputs.rows() == targets.rows() && "output dimension did not match target dimension");
    return (targets - outputs).array().pow(2).matrix().sum();
}


MatrixXf InitWeights(int rows, int cols, float max_weight)
{
    return MatrixXf::Random(rows, cols) * max_weight;
}

MatrixXf FeedForward_Matrix(MatrixXf inputs, MatrixXf weights, VectorXf bias)
{
    MatrixXf Z(inputs.rows(), weights.cols());
    for (int r = 0; r < inputs.rows(); r++)
    {
        VectorXf x = inputs.row(r).transpose();
        VectorXf z = FeedForward_Vector(x, weights, bias);
        Z.row(r) = z.transpose();
    }
    return Z;
}

VectorXf FeedForward_Vector(VectorXf inputs, MatrixXf weights, VectorXf bias)
{
    return weights.transpose() * inputs + bias;
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

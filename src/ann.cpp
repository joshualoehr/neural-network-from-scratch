#include "proto.hpp"

vector<MatrixXf> TrainOneEpoch(MatrixXf X, vector<MatrixXf> WW, vector<VectorXf> bb, MatrixXf Y, float lr, int num_layers, float& error);
MatrixXf Evaluate(MatrixXf X, vector<MatrixXf> WW, vector<VectorXf> bb, int num_layers);
MatrixXf UpdateWeights(MatrixXf W, VectorXf A, VectorXf delta, float lr);
VectorXf DeltaOutput(VectorXf A, MatrixXf Y, int index);
VectorXf DeltaHidden(VectorXf a, MatrixXf weights, VectorXf delta_prevs);
vector<MatrixXf> BackProp(vector<MatrixXf> AA, vector<MatrixXf> WW, MatrixXf Y, int num_layers, int num_samples, float lr);
float BinaryCrossEntropy(VectorXf A, VectorXf Y);
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

string mstr(string name, MatrixXf m)
{
    stringstream ss;
    ss << name << " (" << mdim(m) << "): \n" << m << "\n";
    return ss.str();
}

string vstr(string name, VectorXf v)
{
    stringstream ss;
    ss << name << " (" << vdim(v) << "): \n" << v << "\n";
    return ss.str();
}

int main()
{
    // Reset the random generator seed
    srand((unsigned int) time(0));

    // MatrixXf csv_contents = ParseInputCSV(INPUT_FILENAME_SMALL);
    // cout << csv_contents << endl;
    // cout << endl << endl;

    // Hyperparameters
    int num_layers = 2;
    float error_threshold = 0.5;
    float lr = 0.1;
    float max_weight_val = 20.0;

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

    MatrixXf Y(4,1);
    Y(0,0) = 0;
    Y(1,0) = 1;
    Y(2,0) = 1;
    Y(3,0) = 0;
    cout << "target output: " << endl << Y << endl << endl;



    // Initialization
    cout << "Initializing... ";
    vector<int> L(num_layers+1);
    L[0] = X.cols();
    L[1] = 2;
    L[2] = 1;

    vector<MatrixXf> WW(num_layers+1);
    vector<VectorXf> bb(num_layers+1);
    for (int l = 1; l <= num_layers; l++)
    {
        WW[l] = InitWeights(L[l-1], L[l], max_weight_val);
        bb[l] = VectorXf::Ones(L[l]);
    }
    cout << "done\n";

    // Training
    cout << "Training... \n";
    int epoch = 0;
    float error = 20;
    while (error > error_threshold)
    {
        cout << "Epoch #" << (++epoch) << " -- ";
        WW = TrainOneEpoch(X, WW, bb, Y, lr, num_layers, error);

        if (epoch > 100)
            break;
    }
    cout << "Training done\n";

    // Evaluation
    cout << "Evaluating... \n";
    MatrixXf output = Evaluate(X, WW, bb, num_layers);
    cout << mstr("Output", output) << endl;
    cout << "Final Error: " << BinaryCrossEntropy(output, Y);

    return 0;
}

float BinaryCrossEntropy(VectorXf A, VectorXf Y)
{
    // Prevent NaN's by avoiding computing log(0)
    for (int i = 0; i < A.rows(); i++)
    {
        if (A(i) == 1)
            A(i) -= 0.000001;
        else if (A(i) == 0)
            A(i) += 0.000001;
    }

    VectorXf ones = VectorXf::Ones(Y.rows());
    VectorXf loss = (-1 * Y).cwiseProduct(A.array().log().matrix()) - (ones - Y).cwiseProduct((ones - A).array().log().matrix());
    return loss.sum() / loss.rows();
}

vector<MatrixXf> TrainOneEpoch(MatrixXf X, vector<MatrixXf> WW, vector<VectorXf> bb, MatrixXf Y, float lr, int num_layers, float& error)
{
    cout << "(Forward pass (" << num_layers << " layers)... ";
    vector<MatrixXf> ZZ(num_layers+1);
    vector<MatrixXf> AA(num_layers+1);
    AA[0] = X;

    for (int i = 1; i < num_layers+1; i++)
    {
        ZZ[i] = FeedForward_Matrix(AA[i-1], WW[i], bb[i]);
        AA[i] = Sigmoid(ZZ[i]);
    }
    cout << "done) ";

    cout << "(BackProp... ";
    WW = BackProp(AA, WW, Y, num_layers, X.rows(), lr);
    cout << "done) ";

    error = BinaryCrossEntropy(AA[num_layers], Y);
    cout << "Error: " << error << endl;

    return WW;
}

MatrixXf Evaluate(MatrixXf X, vector<MatrixXf> WW, vector<VectorXf> bb, int num_layers)
{
    vector<MatrixXf> ZZ(num_layers+1);
    vector<MatrixXf> AA(num_layers+1);
    AA[0] = X;

    for (int i = 1; i < num_layers+1; i++)
    {
        ZZ[i] = FeedForward_Matrix(AA[i-1], WW[i], bb[i]);
        AA[i] = Sigmoid(ZZ[i]);
    }

    MatrixXf A_final = AA[num_layers];
    return A_final;
}


vector<MatrixXf> BackProp(vector<MatrixXf> AA, vector<MatrixXf> WW, MatrixXf Y, int num_layers, int num_samples, float lr)
{
    for (int s = 0; s < num_samples; s++)
    {
        vector<VectorXf> deltas(num_layers);
        deltas.push_back(DeltaOutput(AA[num_layers], Y, s));

        for (int i = num_layers; i > 0; i--)
        {
            VectorXf A_s = AA[i-1].row(s).transpose();
            VectorXf delta = deltas[i];
            MatrixXf W = WW[i];

            deltas[i-1] = DeltaHidden(A_s, W, delta);
        }

        // (Purely) Stochastic Gradient Descent - batch size = 1
        for (int j = num_layers; j > 0; j--)
        {
            VectorXf A_s = AA[j-1].row(s).transpose();
            VectorXf delta = deltas[j];
            WW[j] = UpdateWeights(WW[j], A_s, delta, lr);
        }
    }

    return WW;
}



MatrixXf UpdateWeights(MatrixXf W, VectorXf A, VectorXf delta, float lr)
{
    return W + lr * A * delta.transpose();

    // for (int r = 0; r < W_new.rows(); r++)
    // {
    //     float delta = deltas(r);
    //     W_new.row(r) = W_new.row(r) + lr * RowVectorXf::Constant(W_new.cols(), delta);
    // }
    //
    // return W_new.transpose();
}

VectorXf DeltaOutput(VectorXf A, MatrixXf Y, int index)
{
    float a = A(index), y = Y(index);
    VectorXf delta(1);
    delta << a * (1 - a) * (y - a);
    return delta;
}

VectorXf DeltaHidden(VectorXf A, MatrixXf W, VectorXf delta)
{
    VectorXf ones = VectorXf::Ones(A.rows());
    return A.cwiseProduct(ones - A).cwiseProduct(W * delta);
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
        VectorXf z = weights.transpose() * x + bias;
        //VectorXf z = FeedForward_Vector(x, weights, bias);
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

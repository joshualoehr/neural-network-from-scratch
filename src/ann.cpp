#include "proto.hpp"

void TrainOneEpoch(MatrixXf X, vector<MatrixXf>& WW, vector<VectorXf>& bb, MatrixXf Y, float lr, int num_layers, float& error);
MatrixXf Evaluate(MatrixXf X, vector<MatrixXf> WW, vector<VectorXf> bb, int num_layers);
VectorXf ConvertOutputLayer(VectorXf A);
MatrixXf UpdateWeights(MatrixXf W, VectorXf A, VectorXf delta, float lr);
VectorXf UpdateBiases(VectorXf b, VectorXf delta, float lr);
VectorXf DeltaOutput(VectorXf A, MatrixXf Y, int index);
VectorXf DeltaHidden(VectorXf a, MatrixXf weights, VectorXf delta_prevs);
void BackProp(vector<MatrixXf> AA, vector<MatrixXf>& WW, vector<VectorXf>& bb, MatrixXf Y, int num_layers, int num_samples, float lr);
float BinaryCrossEntropy(VectorXf A, VectorXf Y);
MatrixXf FeedForward_Matrix(MatrixXf inputs, MatrixXf weights, VectorXf bias);
VectorXf FeedForward_Vector(VectorXf inputs, MatrixXf weights, VectorXf bias);
MatrixXf InitWeights(int rows, int cols, float max_weight);
MatrixXf Sigmoid(MatrixXf X);

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
    MatrixXf X;
    VectorXf Y;

    ParseInputCSV(INPUT_FILENAME, X, Y);
    cout << mstr("X", X) << endl;
    cout << vstr("Y", Y) << endl;

    // Hyperparameters
    float error_threshold = 0.1;
    int max_epochs = 100000;
    float lr = 0.2;
    float max_weight_val = 1.0;

    // Network Topology
    int num_layers = 5;
    vector<int> L(num_layers+1);
    L[0] = X.cols();
    L[1] = 90;
    L[2] = 60;
    L[3] = 30;
    L[4] = 10;
    L[num_layers] = 1; // final layer should always be of dim 1

    // Initialization
    vector<MatrixXf> WW(num_layers+1);
    vector<VectorXf> bb(num_layers+1);
    for (int l = 1; l <= num_layers; l++)
    {
        WW[l] = InitWeights(L[l-1], L[l], max_weight_val);
        bb[l] = VectorXf::Ones(L[l]);
    }

    // Training
    float error = 20;
    for (int epoch = 0; epoch < max_epochs; epoch++)
    {
        // cout << "Epoch #" << (++epoch) << " -- ";
        TrainOneEpoch(X, WW, bb, Y, lr, num_layers, error);
        if (error <= error_threshold)
        {
            cout << "Converged at epoch " << epoch << endl << endl;
            break;
        }
    }

    if (error <= error_threshold)
    {
        // Evaluation
        MatrixXf output = Evaluate(X, WW, bb, num_layers);
        cout << mstr("Final Layer", output) << endl;
        cout << mstr("Final Output", ConvertOutputLayer(output)) << endl;
        cout << "Final Error: " << BinaryCrossEntropy(ConvertOutputLayer(output), Y) << endl << endl;

        // for (int i = 1; i <= num_layers; i++)
        // {
        //     cout << "<<< Layer " << i << " >>>\n\n";
        //     cout << mstr("W", WW[i]) << endl;
        //     cout << vstr("b", bb[i]) << endl << endl;
        // }
    }
    else
    {
        cout << "Did not converge after " << max_epochs << " epochs :(\n";
    }

    return 0;
}

int DoXOR()
{
    // Reset the random generator seed
    srand((unsigned int) time(0));

    MatrixXf X(8,3);
    X(0,0) = 0;
    X(0,1) = 0;
    X(0,2) = 0;
    X(1,0) = 0;
    X(1,1) = 0;
    X(1,2) = 1;
    X(2,0) = 0;
    X(2,1) = 1;
    X(2,2) = 0;
    X(3,0) = 0;
    X(3,1) = 1;
    X(3,2) = 1;
    X(4,0) = 1;
    X(4,1) = 0;
    X(4,2) = 0;
    X(5,0) = 1;
    X(5,1) = 0;
    X(5,2) = 1;
    X(6,0) = 1;
    X(6,1) = 1;
    X(6,2) = 0;
    X(7,0) = 1;
    X(7,1) = 1;
    X(7,2) = 1;
    cout << "inputs: " << endl << X << endl << endl;

    MatrixXf Y(8,1);
    Y(0,0) = 0;
    Y(1,0) = 0;
    Y(2,0) = 1;
    Y(3,0) = 1;
    Y(4,0) = 1;
    Y(5,0) = 1;
    Y(6,0) = 0;
    Y(7,0) = 0;
    cout << "target output: " << endl << Y << endl << endl;

    // Hyperparameters
    float error_threshold = 0.1;
    int max_epochs = 100000;
    float lr = 0.2;
    float max_weight_val = 1.0;

    // Network Topology
    int num_layers = 3;
    vector<int> L(num_layers+1);
    L[0] = X.cols();
    L[1] = 3;
    L[2] = 2;
    L[3] = 1;

    // Initialization
    vector<MatrixXf> WW(num_layers+1);
    vector<VectorXf> bb(num_layers+1);
    for (int l = 1; l <= num_layers; l++)
    {
        WW[l] = InitWeights(L[l-1], L[l], max_weight_val);
        bb[l] = VectorXf::Ones(L[l]);
    }

    // Training
    float error = 20;
    for (int epoch = 0; epoch < max_epochs; epoch++)
    {
        // cout << "Epoch #" << (++epoch) << " -- ";
        TrainOneEpoch(X, WW, bb, Y, lr, num_layers, error);
        if (error <= error_threshold)
        {
            cout << "Converged at epoch " << epoch << endl << endl;
            break;
        }
    }

    if (error <= error_threshold)
    {
        // Evaluation
        MatrixXf output = Evaluate(X, WW, bb, num_layers);
        cout << mstr("Final Layer", output) << endl;
        cout << mstr("Final Output", ConvertOutputLayer(output)) << endl;
        cout << "Final Error: " << BinaryCrossEntropy(ConvertOutputLayer(output), Y) << endl << endl;

        for (int i = 1; i <= num_layers; i++)
        {
            cout << "<<< Layer " << i << " >>>\n\n";
            cout << mstr("W", WW[i]) << endl;
            cout << vstr("b", bb[i]) << endl << endl;
        }
    }
    else
    {
        cout << "Did not converge after " << max_epochs << " epochs :(\n";
    }

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

void TrainOneEpoch(MatrixXf X, vector<MatrixXf>& WW, vector<VectorXf>& bb, MatrixXf Y, float lr, int num_layers, float& error)
{
    vector<MatrixXf> ZZ(num_layers+1);
    vector<MatrixXf> AA(num_layers+1);
    AA[0] = X;

    for (int i = 1; i < num_layers+1; i++)
    {
        ZZ[i] = FeedForward_Matrix(AA[i-1], WW[i], bb[i]);
        AA[i] = Sigmoid(ZZ[i]);
    }

    BackProp(AA, WW, bb, Y, num_layers, X.rows(), lr);

    // cout << "Error: " << BinaryCrossEntropy(AA[num_layers], Y) << endl;
    error = BinaryCrossEntropy(ConvertOutputLayer(AA[num_layers]), Y);
}

void BackProp(vector<MatrixXf> AA, vector<MatrixXf>& WW, vector<VectorXf>& bb, MatrixXf Y, int num_layers, int num_samples, float lr)
{
    VectorXf output = ConvertOutputLayer(AA[num_layers]);

    for (int s = 0; s < num_samples; s++)
    {
        if (output(s) == Y(s))
            continue;

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
            bb[j] = UpdateBiases(bb[j], delta, lr);
        }
    }
}

VectorXf ConvertOutputLayer(VectorXf A)
{
    for (int i = 0; i < A.rows(); i++)
    {
        A(i) = (A(i) >= 0.5) ? 1 : 0;
    }
    return A;
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

MatrixXf UpdateWeights(MatrixXf W, VectorXf A, VectorXf delta, float lr)
{
    return W + lr * A * delta.transpose();
}

VectorXf UpdateBiases(VectorXf b, VectorXf delta, float lr)
{
    return b + lr * delta;
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
        Z.row(r) = z.transpose();
    }
    return Z;
}

MatrixXf Sigmoid(MatrixXf X)
{
    MatrixXf ones = MatrixXf::Constant(X.rows(), X.cols(), 1.0f);
    return ((X * -1).array().exp().matrix() + ones).cwiseInverse();
}

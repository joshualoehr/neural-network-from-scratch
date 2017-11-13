#include "proto.hpp"

void SaveWeightsAndBiases(vector<MatrixXf> WW, vector<VectorXf> bb);
string GetFilename(string name);
void WriteWeights(string filename, vector<MatrixXf> WW);
void WriteBiases(string filename, vector<VectorXf> bb);

MatrixXf Evaluate(MatrixXf X, vector<MatrixXf> WW, vector<VectorXf> bb, int num_layers);
VectorXf ConvertOutputLayer(VectorXf A);
float BinaryCrossEntropy(VectorXf A, VectorXf Y);
float Error(VectorXf A, VectorXf Y);
float PercentCorrect(VectorXf A, VectorXf Y);
string ErrorReport(VectorXf A, VectorXf Y);

void TrainOneEpoch(MatrixXf X, vector<MatrixXf>& WW, vector<VectorXf>& bb, MatrixXf Y, float lr, int num_layers, float& error);

MatrixXf FeedForward_Matrix(MatrixXf inputs, MatrixXf weights, VectorXf bias);
VectorXf FeedForward_Vector(VectorXf inputs, MatrixXf weights, VectorXf bias);
MatrixXf InitWeights(int rows, int cols, float max_weight);
MatrixXf Sigmoid(MatrixXf X);

void BackProp(vector<MatrixXf> AA, vector<MatrixXf>& WW, vector<VectorXf>& bb, MatrixXf Y, int num_layers, int num_samples, float lr);
VectorXf DeltaOutput(VectorXf A, MatrixXf Y, int index);
VectorXf DeltaHidden(VectorXf a, MatrixXf weights, VectorXf delta_prevs);
MatrixXf UpdateWeights(MatrixXf W, VectorXf A, VectorXf delta, float lr);
VectorXf UpdateBiases(VectorXf b, VectorXf delta, float lr);

string mdim(MatrixXf m) {
    stringstream ss;
    ss << m.rows() << "x" << m.cols();
    return ss.str();
}

string vdim(VectorXf v) {
    stringstream ss;
    ss << v.rows() << "x" << v.cols();
    return ss.str();
}

string mstr(string name, MatrixXf m) {
    stringstream ss;
    ss << name << " (" << mdim(m) << "): \n" << m << "\n";
    return ss.str();
}

string vstr(string name, VectorXf v) {
    stringstream ss;
    ss << name << " (" << vdim(v) << "): \n" << v << "\n";
    return ss.str();
}

int main(int argc, char **argv) {
    if (argc > 2) {
        cout << USAGE << endl;
        exit(0);
    }

    // Initialize random seed
    uint seed = time(0);
    if (argc == 2) {
        seed = (uint) atoi(argv[1]);
    }
    srand(seed);
    cout << "Initializing random seed to: " << seed << endl;
    cout << endl;

    // Hyperparameters
    cout << "<<< Hyperparameters >>>\n";
    float lr = 0.001;
    float max_weight_val = 0.8;
    float error_threshold = 0.02;
    int max_epochs = 1000;
    int max_divergence = 50;
    cout << "Learning Rate:                " << lr << endl;
    cout << "Initial Weight Range:         [" << (-1 * max_weight_val) << ", " << max_weight_val << "]" << endl;
    cout << "Error Convergence Threshold:  " << error_threshold << endl;
    cout << "Maximum Training Epochs:      " << max_epochs << endl;
    cout << "Epochs Before Divergence:     " << max_divergence << endl;
    cout << endl;

    // Initialize train and test data
    cout << "<<< Loading Data >>>\n";
    MatrixXf X_train, X_test;
    VectorXf Y_train, Y_test;
    ParseInputCSV(INPUT_FILENAME_TRAIN, X_train, Y_train);
    ParseInputCSV(INPUT_FILENAME_TEST, X_test, Y_test);
    cout << "Sample Features: X_train (" << mdim(X_train) << "), X_test (" << mdim(X_test) << ")" << endl;
    cout << "Sample Outputs:  Y_train (" << vdim(Y_train) << "), Y_test (" << mdim(Y_test) << ")" << endl;
    cout << endl;

    // Network Topology
    cout << "<<< Topology >>>\n";
    int num_layers = 2;
    vector<int> L(num_layers+1);
    L[0] = X_train.cols(); // Zero'th layer is the input vector
    L[1] = 64;
    L[num_layers] = 1; // final layer should always be of dim 1
    cout << "Hidden Layers: " << num_layers << endl << endl;
    cout << "Input Layer:    " << L[0] << " nodes" << endl;
    for (int i = 1; i <= num_layers; i++) {
        cout << "Hidden Layer" << i << ":  " << L[i] << " nodes" << endl;
    }
    cout << endl;

    // Weight Initialization
    cout << "<<< Initializing Weights & Biases >>>\n";
    vector<MatrixXf> WW(num_layers+1);
    vector<VectorXf> bb(num_layers+1);
    for (int l = 1; l <= num_layers; l++) {
        WW[l] = InitWeights(L[l-1], L[l], max_weight_val);
        bb[l] = VectorXf::Ones(L[l]);

        cout << "W[" << l << "]: " << mdim(WW[l]) << endl;
        cout << "B[" << l << "]: " << vdim(bb[l]) << endl;
    }
    SaveWeightsAndBiases(WW, bb);
    cout << endl;

    // Initial Evaluation
    cout << "<<< Initial Evaluation >>>\n";
    VectorXf H_output = Evaluate(X_train, WW, bb, num_layers);
    float bce = BinaryCrossEntropy(H_output, Y_train);
    float error = Error(H_output, Y_train);
    float acc = PercentCorrect(H_output, Y_train);
    cout << "Initial Error -- " << ErrorReport(H_output, Y_train) << endl;
    cout << endl;

    // Training
    cout << "<<< Training >>>\n";
    vector<MatrixXf> best_WW = WW;
    vector<VectorXf> best_bb = bb;
    float old_bce = bce, best_bce = bce;
    float acc_report_threshold = 81.0;
    float divergence_count = 0;
    int epoch = 0;

    while (epoch < max_epochs) {
        // Train for one epoch, get the new hidden output layer, and update error value
        TrainOneEpoch(X_train, WW, bb, Y_train, lr, num_layers, bce);
        H_output = Evaluate(X_train, WW, bb, num_layers);
        error = Error(H_output, Y_train);
        acc = PercentCorrect(H_output, Y_train);
        cout << "Epoch #" << epoch << " -- Accuracy: " << acc << "%\n";

        // If the error improved, reset the divergence counter; otherwise, increment
        divergence_count = (bce < old_bce) ? 0 : divergence_count + 1;
        if (divergence_count >= max_divergence) {
            cout << "Diverged at epoch " << epoch << endl << endl;
            break;
        }

        // If the error is a new best, save the weights and biases
        if (bce < best_bce) {
            best_bce = bce;
            best_WW = WW;
            best_bb = bb;

            if (acc >= acc_report_threshold) {
                cout << endl;
                cout << "**Saving** -- " << ErrorReport(H_output, Y_train) << endl;
                SaveWeightsAndBiases(WW, bb);
                cout << endl;

                acc_report_threshold += 3.0;
            }

            // If the error is under the threshold, we have converged
            if (error <= error_threshold || acc >= 99.8) {
                cout << endl << "Saving Final Weights..." << endl;
                SaveWeightsAndBiases(WW, bb);
                cout << endl;
                break;
            }

        }

        old_bce = bce;
        epoch++;
    }

    if (epoch == max_epochs)
        cout << "Exceeded max training epochs without converging" << endl;
    else if (divergence_count >= max_divergence)
        cout << "Training diverged at epoch " << epoch << endl;
    else
        cout << "Training converged at epoch " << epoch << endl;
    cout << endl;

    // Evaluation
    cout << "<<< Final Evaluation >>>\n";
    H_output = Evaluate(X_train, best_WW, best_bb, num_layers);
    cout << "Final Training Error -- " << ErrorReport(H_output, Y_train) << endl;
    H_output = Evaluate(X_test, best_WW, best_bb, num_layers);
    cout << "Final Test Error -- " << ErrorReport(H_output, Y_test) << endl;
    cout << endl;

    return 0;
}

int write_counter = 0;
void SaveWeightsAndBiases(vector<MatrixXf> WW, vector<VectorXf> bb) {
    string weights_file = GetFilename("weights");
    string biases_file = GetFilename("biases");

    WriteWeights(weights_file, WW);
    cout << "Weights saved to: " << weights_file << endl;
    WriteBiases(biases_file, bb);
    cout << "Biases saved to: " << biases_file << endl;

    write_counter++;
}

string GetFilename(string name) {
    stringstream ss;
    ss << name << write_counter << ".txt";
    return ss.str();
}

void WriteWeights(string filename, vector<MatrixXf> WW) {
    ofstream f;
    f.open(filename.c_str());
    for (uint i = 1; i < WW.size(); i++) {
        stringstream ss;
        ss << "W[" << i << "]";
        f << mstr(ss.str(), WW[i]) << endl << endl;
    }
    f.close();
}

void WriteBiases(string filename, vector<VectorXf> bb) {
    ofstream f;
    f.open(filename.c_str());
    for (uint i = 1; i < bb.size(); i++) {
        stringstream ss;
        ss << "B[" << i << "]";
        f << vstr(ss.str(), bb[i]) << endl << endl;
    }
    f.close();
}

float BinaryCrossEntropy(VectorXf A, VectorXf Y) {
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

float Error(VectorXf A, VectorXf Y) {
    return BinaryCrossEntropy(ConvertOutputLayer(A), Y);
}

string ErrorReport(VectorXf A, VectorXf Y) {
    stringstream ss;
    ss << "BCE: " << BinaryCrossEntropy(A, Y) << ", ";
    ss << "Error: " << Error(A, Y) << ", ";
    ss << "Correct: " << PercentCorrect(A, Y) << "%";
    return ss.str();
}

float PercentCorrect(VectorXf A, VectorXf Y) {
    VectorXf output = ConvertOutputLayer(A);

    float num_correct = 0;
    for (int i = 0; i < Y.rows(); i++) {
        if (output(i) == Y(i))
            num_correct++;
    }
    return 100.0 * (num_correct / (float)Y.rows());
}

void TrainOneEpoch(MatrixXf X, vector<MatrixXf>& WW, vector<VectorXf>& bb, MatrixXf Y, float lr, int num_layers, float& bce) {
    vector<MatrixXf> ZZ(num_layers+1);
    vector<MatrixXf> AA(num_layers+1);
    AA[0] = X;

    for (int i = 1; i < num_layers+1; i++) {
        ZZ[i] = FeedForward_Matrix(AA[i-1], WW[i], bb[i]);
        AA[i] = Sigmoid(ZZ[i]);
    }

    bce = BinaryCrossEntropy(AA[num_layers], Y);
    BackProp(AA, WW, bb, Y, num_layers, X.rows(), lr);

    MatrixXf new_results(Y.rows(), 3);
    VectorXf new_AA_final = Evaluate(X, WW, bb, num_layers);
    new_results.col(0) = new_AA_final;
    new_results.col(1) = ConvertOutputLayer(new_AA_final);
    new_results.col(2) = Y;
}

void BackProp(vector<MatrixXf> AA, vector<MatrixXf>& WW, vector<VectorXf>& bb, MatrixXf Y, int num_layers, int num_samples, float lr) {
    VectorXf output = ConvertOutputLayer(AA[num_layers]);
    MatrixXf results(Y.rows(), 3);
    results.col(0) = AA[num_layers];
    results.col(1) = output;
    results.col(2) = Y;

    for (int s = 0; s < num_samples; s++) {
        if (output(s) == Y(s))
            continue;

        vector<VectorXf> deltas(num_layers);
        deltas.push_back(DeltaOutput(AA[num_layers], Y, s));

        for (int i = num_layers; i > 0; i--) {
            VectorXf A_s = AA[i-1].row(s).transpose();
            VectorXf delta = deltas[i];
            MatrixXf W = WW[i];

            deltas[i-1] = DeltaHidden(A_s, W, delta);
        }

        for (int j = num_layers; j > 0; j--) {
            VectorXf A_s = AA[j-1].row(s).transpose();
            VectorXf delta = deltas[j];
            WW[j] = UpdateWeights(WW[j], A_s, delta, lr);
            bb[j] = UpdateBiases(bb[j], delta, lr);
        }
    }
}

VectorXf ConvertOutputLayer(VectorXf A) {
    for (int i = 0; i < A.rows(); i++) {
        A(i) = (A(i) >= 0.5) ? 1 : 0;
    }
    return A;
}

MatrixXf Evaluate(MatrixXf X, vector<MatrixXf> WW, vector<VectorXf> bb, int num_layers) {
    vector<MatrixXf> ZZ(num_layers+1);
    vector<MatrixXf> AA(num_layers+1);
    AA[0] = X;

    for (int i = 1; i < num_layers+1; i++) {
        ZZ[i] = FeedForward_Matrix(AA[i-1], WW[i], bb[i]);
        AA[i] = Sigmoid(ZZ[i]);
    }

    return AA[num_layers];
}

MatrixXf UpdateWeights(MatrixXf W, VectorXf A, VectorXf delta, float lr) {
    MatrixXf gradients = A * delta.transpose();
    return W + lr * A * delta.transpose();
}

VectorXf UpdateBiases(VectorXf b, VectorXf delta, float lr) {
    return b + lr * delta;
}

VectorXf DeltaOutput(VectorXf A, MatrixXf Y, int index) {
    float a = A(index), y = Y(index);
    VectorXf delta(1);
    delta << a * (1 - a) * (y - a);
    return delta;
}

VectorXf DeltaHidden(VectorXf A, MatrixXf W, VectorXf delta) {
    VectorXf ones = VectorXf::Ones(A.rows());
    return A.cwiseProduct(ones - A).cwiseProduct(W * delta);
}

MatrixXf InitWeights(int rows, int cols, float max_weight) {
    return MatrixXf::Random(rows, cols) * max_weight;
}

MatrixXf FeedForward_Matrix(MatrixXf inputs, MatrixXf weights, VectorXf bias) {
    MatrixXf Z(inputs.rows(), weights.cols());
    for (int r = 0; r < inputs.rows(); r++) {
        VectorXf x = inputs.row(r).transpose();
        VectorXf z = weights.transpose() * x + bias;
        Z.row(r) = z.transpose();
    }
    return Z;
}

MatrixXf Sigmoid(MatrixXf X) {
    MatrixXf ones = MatrixXf::Constant(X.rows(), X.cols(), 1.0f);
    return ((X * -1).array().exp().matrix() + ones).cwiseInverse();
}

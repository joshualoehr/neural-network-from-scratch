// proto.hpp
#ifndef __PROTO_HPP_INCL__
#define __PROTO_HPP_INCL__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// constants
const string USAGE = "Usage: ./ann [SEED]";
const float E = 2.71828182845904523536; // mathematical constant e

const string INPUT_FILENAME_TRAIN = "mushrooms-train.csv";
const string INPUT_FILENAME_TEST = "mushrooms-test.csv";
const int NUM_MUSHROOM_FEATURES = 23;
const string FEATURE_SPACES[] =
{
    "ep", // class (edible, poisonous)
    "bcxfks", // cap-shape
    "fgsy", // cap-surface
    "nbcgrpuewy", // cap-color
    "tf", // bruises
    "alcyfmnps", // odor
    "adfn", // gill-attach
    "wcd", // gill-space
    "bn", // gill-size
    "knbhgropuewy", // gill-color
    "et", // stalk-shape
    "bcuezr?", // stalk-root
    "fyks", // stalk-surface-above-ring
    "fyks", // stalk-surface-below-ring
    "nbcgopewy", // stalk-color-above-ring
    "nbcgopewy", // stalk-color-below-ring
    "pu", // veil-type
    "nowy", // veil-color
    "not", // ring-number
    "ceflnpsz", // ring-type
    "knbhrouwy", // spore-print-color
    "acnsuy", // population
    "glmpuwd" // habitat
};

// ann.cpp
MatrixXf InitWeights(int rows, int cols, float max_weight);
void PrintSeed(uint seed);
void PrintHyperparameters(float lr, float max_weight_val, float error_threshold, int max_epochs, int max_divergence);
void PrintLoadedData(MatrixXf X_train, MatrixXf X_test, VectorXf Y_train, VectorXf Y_test);
void PrintTopology(int num_layers, vector<int> L);
void PrintWeightInit(vector<MatrixXf> WW, vector<VectorXf> bb, int num_layers);
void PrintInitialEval(VectorXf H_output, VectorXf Y_train);
void PrintFinalEval(VectorXf H_output_train, VectorXf H_output_test, VectorXf Y_train, VectorXf Y_test);

// parse.cpp
void ParseInputCSV(string csv_filename, MatrixXf& samples, VectorXf& labels);
vector<char> NextLine(ifstream& ifs);
void OneHotEncode(vector<char> line, float& label, RowVectorXf& encoding);
RowVectorXf EncodeFeature(char feature, string possibilities, RowVectorXf encoding);

// save.cpp
void SaveWeightsAndBiases(vector<MatrixXf> WW, vector<VectorXf> bb);
string GetFilename(string name);
void WriteWeights(string filename, vector<MatrixXf> WW);
void WriteBiases(string filename, vector<VectorXf> bb);

// eval.cpp
MatrixXf Evaluate(MatrixXf X, vector<MatrixXf> WW, vector<VectorXf> bb, int num_layers);
VectorXf ConvertOutputLayer(VectorXf A);
float BinaryCrossEntropy(VectorXf A, VectorXf Y);
float Error(VectorXf A, VectorXf Y);
float PercentCorrect(VectorXf A, VectorXf Y);
string ErrorReport(VectorXf A, VectorXf Y);

// train.cpp
void TrainOneEpoch(MatrixXf X, vector<MatrixXf>& WW, vector<VectorXf>& bb, MatrixXf Y, float lr, int num_layers, float& error);
void BackProp(vector<MatrixXf> AA, vector<MatrixXf>& WW, vector<VectorXf>& bb, MatrixXf Y, int num_layers, int num_samples, float lr);
VectorXf DeltaOutput(VectorXf A, MatrixXf Y, int index);
VectorXf DeltaHidden(VectorXf a, MatrixXf weights, VectorXf delta_prevs);
MatrixXf UpdateWeights(MatrixXf W, VectorXf A, VectorXf delta, float lr);
VectorXf UpdateBiases(VectorXf b, VectorXf delta, float lr);

// feedforward.cpp
MatrixXf FeedForward(MatrixXf inputs, MatrixXf weights, VectorXf bias);
MatrixXf Sigmoid(MatrixXf X);

// utils.cpp
string mdim(MatrixXf m);
string vdim(VectorXf v);
string mstr(string name, MatrixXf m);
string vstr(string name, VectorXf v);

#endif

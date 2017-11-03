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
const float E = 2.71828182845904523536; // mathematical constant e
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

// uncategorized (tbd)
MatrixXf Sigmoid(MatrixXf X);
VectorXi OutputToClass(MatrixXi output_vectors);
MatrixXi ClassToOutput(VectorXi class_vector);

// parse.hpp
MatrixXi ParseInputCSV(string csv_filename);
vector<char> NextLine(ifstream& ifs);
RowVectorXi OneHotEncode(vector<char> line);
RowVectorXi EncodeFeature(char feature, string possibilities, RowVectorXi encoding);

// data_set.hpp
class DataSet
{
    MatrixXi inputs;
    VectorXi outputs;
    int count;
    int bias;
public:
    DataSet(MatrixXi, VectorXi);
};

#endif

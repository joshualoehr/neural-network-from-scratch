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

// parse.hpp
void ParseInputCSV(string csv_filename, MatrixXf& samples, VectorXf& labels);
vector<char> NextLine(ifstream& ifs);
void OneHotEncode(vector<char> line, float& label, RowVectorXf& encoding);
RowVectorXf EncodeFeature(char feature, string possibilities, RowVectorXf encoding);

#endif

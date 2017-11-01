#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

VectorXi OutputToClass(MatrixXi output_vectors);
MatrixXi ClassToOutput(VectorXi class_vector);
MatrixXi ParseInputCSV(string csv_filename);
vector<char> NextLine(ifstream& ifs);
RowVectorXi OneHotEncode(vector<char> line);
RowVectorXi EncodeFeature(char feature, string possibilities, RowVectorXi encoding);

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

int main()
{
    cout << ParseInputCSV("mushrooms.csv") << endl;
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
    MatrixXi output_vectors = MatrixXi::Zero(num_rows, num_cols);

    for (int i = 0; i < num_rows; i++) {
        int class_val = class_vector(i);
        output_vectors(i, class_val - 1) = 1;
    }

    return output_vectors;
}

MatrixXi ParseInputCSV(string csv_filename)
{
    vector<RowVectorXi> samples;
    ifstream ifs(csv_filename.c_str(), ifstream::in);

    // Consume header
    while (ifs.good() && ifs.get() != '\n');

    while (ifs.good())
    {
        vector<char> line = NextLine(ifs);
        RowVectorXi sample = OneHotEncode(line);
        samples.push_back(sample);
    }
    ifs.close();

    int rows = samples.size();
    int cols = samples[0].cols();
    MatrixXi sample_matrix = MatrixXi::Zero(rows, cols);
    for (int r = 0; r < rows; r++) {
        sample_matrix.row(r) = samples[r];
    }

    return sample_matrix;
}

vector<char> NextLine(ifstream& ifs)
{
    vector<char> line;
    char c;

    while (ifs.good() && (c = ifs.get()) != '\n' && c != -1)
    {
        if (c != ',')
            line.push_back(c);
    }

    return line;
}

/* This encoding scheme is specific to the mushroom classification problem */
RowVectorXi OneHotEncode(vector<char> line)
{
    assert(line.size() == NUM_MUSHROOM_FEATURES && "Unexpected number of features");

    RowVectorXi encoding(0);
    for (int i = 0; i < NUM_MUSHROOM_FEATURES; i++)
    {
        encoding = EncodeFeature(line[i], FEATURE_SPACES[i], encoding);
    }
    return encoding;
}

RowVectorXi EncodeFeature(char feature, string possibilities, RowVectorXi encoding)
{
    RowVectorXi f = RowVectorXi::Zero(possibilities.length());
    for (uint i = 0; i < possibilities.length(); i++)
    {
        if (feature == possibilities[i])
        {
            f(i) = 1;
            break;
        }
    }

    RowVectorXi new_encoding(encoding.cols() + f.cols());
    new_encoding << encoding, f;
    return new_encoding;
}

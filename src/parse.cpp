#include "proto.hpp"

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

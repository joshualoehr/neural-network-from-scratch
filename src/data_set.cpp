#include "proto.hpp"

DataSet::DataSet(MatrixXf inputs, VectorXf outputs)
{
    this->inputs = inputs;
    this->outputs = outputs;
    this->count = inputs.rows() + outputs.rows();
    this->bias = 1;
}

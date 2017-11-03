#include "proto.hpp"

DataSet::DataSet(MatrixXi inputs, VectorXi outputs)
{
    this->inputs = inputs;
    this->outputs = outputs;
    this->count = inputs.rows() + outputs.rows();
    this->bias = 1;
}

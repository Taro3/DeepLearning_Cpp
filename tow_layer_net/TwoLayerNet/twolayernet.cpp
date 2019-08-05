#include <Eigen/Dense>

#include "twolayernet.h"

TwoLayerNet::TwoLayerNet(int input_size, int hidden_size, int output_size, double weigt_init_std, QObject *parent) :
    QObject(parent),
    _params(std::map<std::string, Eigen::MatrixXd>())
{
    _params["W1"] = weigt_init_std * Eigen::MatrixXd::Random(input_size, hidden_size);
    _params["b1"] = Eigen::VectorXd::Zero(hidden_size);
    _params["W2"] = weigt_init_std * Eigen::MatrixXd::Random(hidden_size, output_size);
    _params["b2"] = Eigen::VectorXd::Zero(output_size);
}

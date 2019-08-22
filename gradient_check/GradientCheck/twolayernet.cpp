#include <iostream>

#include <Eigen/Dense>

#include "twolayernet.h"
#include "layers.h"

TwoLayerNet::TwoLayerNet(int input_size, int hidden_size, int output_size, double weight_init_std, QObject *parent) :
    QObject(parent),
    _params(std::map<std::string, Eigen::MatrixXd>()),
    _layers(OrderdDict<std::string, layerBase*>()),
    _lastLayer(new SoftmaxWithLoss(this))
{
    // 重みの初期化
    _params["W1"] = weight_init_std * Eigen::MatrixXd::Random(input_size, hidden_size);
    _params["b1"] = Eigen::MatrixXd(hidden_size, 1).setZero();
    _params["W2"] = weight_init_std * Eigen::MatrixXd::Random(hidden_size, output_size);
    _params["b2"] = Eigen::MatrixXd(output_size, 1).setZero();

    // レイヤの生成
    _layers["Affine1"] = new Affine(_params["W1"], _params["b1"], this);
    _layers["Relu1"] = new Relu(this);
    _layers["Affine2"] = new Affine(_params["W2"], _params["b2"], this);
}

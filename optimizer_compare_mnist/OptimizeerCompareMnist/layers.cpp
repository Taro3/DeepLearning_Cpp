#include <Eigen/Dense>

#include "layers.h"

layerBase::layerBase(QObject *parent) : QObject(parent)
{

}

Relu::Relu(QObject *parent) : layerBase(parent),
    _mask(Eigen::MatrixXi())
{

}

Sigmoid::Sigmoid(QObject *parent) : layerBase(parent),
    _out(Eigen::MatrixXd())
{

}

Affine::Affine(Eigen::MatrixXd &W, Eigen::MatrixXd &b, QObject *parent) : layerBase(parent),
    _W(&W), _b(&b), _x(Eigen::MatrixXd()), _dW(Eigen::MatrixXd()), _db(Eigen::MatrixXd())
{

}

SoftmaxWithLoss::SoftmaxWithLoss(QObject *parent) : QObject(parent),
    _loss(0), _y(Eigen::MatrixXd()), _t(Eigen::MatrixXd())
{

}

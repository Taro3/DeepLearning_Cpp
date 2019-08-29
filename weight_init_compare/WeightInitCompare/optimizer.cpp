#include <Eigen/Dense>

#include "optimizer.h"

OptimizerBase::OptimizerBase(QObject *parent) : QObject(parent)
{

}

SGD::SGD(double lr, QObject *parent) : OptimizerBase(parent),
    _lr(lr)
{

}

Momentum::Momentum(double lr, double momentum, QObject *parent) : OptimizerBase(parent),
    _lr(lr),
    _momentum(momentum),
    _v(std::map<std::string, Eigen::MatrixXd>())
{

}

AdaGrad::AdaGrad(double lr, QObject *parent) : OptimizerBase(parent),
    _lr(lr),
    _h(std::map<std::string, Eigen::MatrixXd>())
{

}

Adam::Adam(double lr, double beta1, double beta2, QObject *parent) : OptimizerBase(parent),
    _lr(lr),
    _beta1(beta1),
    _beta2(beta2),
    _iter(0),
    _m(std::map<std::string, Eigen::MatrixXd>()),
    _v(std::map<std::string, Eigen::MatrixXd>())
{

}

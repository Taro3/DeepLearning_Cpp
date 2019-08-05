#include <random>

#include <Eigen/Dense>

#include "simplenet.h"

simpleNet::simpleNet(QObject *parent) : QObject(parent), _W(Eigen::MatrixXd::Random(2, 3))
{
    // -1～1の乱数を0～1の乱数に変更する
    for (int i = 0; i < _W.size(); ++i) {
        _W.data()[i] = _W.data()[i] / 2 + 0.5;
    }
}

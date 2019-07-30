#include <iostream>

#include <QCoreApplication>

#include <Eigen/Dense>

#include "lossfunction.h"

int main(int argc, char *argv[])
{
    Q_UNUSED(argc)
    Q_UNUSED(argv)

    LossFunction lf;
    Eigen::VectorXd t(10);
    // 「2」を正解とする
    t << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;

    // 例1:「2」の確率が最も高い場合(0.6)
    Eigen::VectorXd y(10);
    y << 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0;
    std::cout << lf.mean_squared_error(y, t) << std::endl;

    // 例2:「7」の確率が最も高い場合(0.6)
    y << 0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0;
    std::cout << lf.mean_squared_error(y, t) << std::endl;

    return 0;
}

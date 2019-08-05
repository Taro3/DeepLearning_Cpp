#include <iostream>

#include <QCoreApplication>

#include <Eigen/Dense>

#include "simplenet.h"

Eigen::MatrixXd numerical_gradient(std::function<double(Eigen::MatrixXd)> f, Eigen::MatrixXd x) {
    double h = 1e-4;    // 0.0001
    Eigen::MatrixXd grad(x);
    grad.setZero();

    for (int idx = 0; idx < x.size(); ++idx) {
        double tmp_val = x.data()[idx];
        // f(x+h)の計算
        x.data()[idx] = tmp_val + h;
        double fxh1 = f(x.transpose());

        // f(x-h)の計算
        x.data()[idx] = tmp_val - h;
        double fxh2 = f(x.transpose());

        grad.data()[idx] = (fxh1 - fxh2) / (2 * h);
        x.data()[idx] = tmp_val;   // 値を元に戻す
    }

    return grad;
}

int main(int argc, char *argv[])
{
    Q_UNUSED(argc)
    Q_UNUSED(argv)

    /*
     * simpleNetを使う
     */
    simpleNet net;
    std::cout << net._W << std::endl;   // 重みパラメータ

    std::cout << std::endl;

    Eigen::RowVector2d x(0.6, 0.9);
    auto p = net.predict(x);
    std::cout << p << std::endl;

    std::cout << std::endl;

    int max_index;
    p.row(0).maxCoeff(&max_index);
    std::cout << max_index << std::endl;    // 最大値のインデックス

    std::cout << std::endl;

    Eigen::RowVector3d t(0, 0, 1);  // 正解ラベル
    std::cout << net.loss(x, t) << std::endl;

    std::cout << std::endl;

    /*
     * 勾配を求めてみよう
     */
    std::function<double(Eigen::MatrixXd)> f = [&](Eigen::MatrixXd x) { return net.loss(x, t); };

    Eigen::MatrixXd dW = numerical_gradient(f, net._W);
    std::cout << dW << std::endl;

    return 0;
}

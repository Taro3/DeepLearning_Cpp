#include <iostream>

#include <QCoreApplication>

#include <Eigen/Dense>

#include "twolayernet.h"

int main(int argc, char *argv[])
{
    Q_UNUSED(argc)
    Q_UNUSED(argv)

    // パラメータの確認
    TwoLayerNet net(784, 100, 10);
    std::cout << net._params["W1"].rows() << " " << net._params["W1"].cols() << std::endl;
    std::cout << net._params["b1"].rows() << " " << net._params["b1"].cols() << std::endl;
    std::cout << net._params["W2"].rows() << " " << net._params["W2"].cols() << std::endl;
    std::cout << net._params["b2"].rows() << " " << net._params["b2"].cols() << std::endl;

    std::cout << std::endl;

    // 勾配情報生成
    Eigen::MatrixXd x = Eigen::MatrixXd::Random(100, 784);  // ダミーの入力データ(100枚分)
    x /= 2;
    x += 0.5 * Eigen::MatrixXd::Ones(x.rows(), x.cols());
    Eigen::MatrixXd t = Eigen::MatrixXd::Random(100, 10);   // ダミーの正解ラベル(100枚分)
    t /= 2;
    t += 0.5 * Eigen::MatrixXd::Ones(t.rows(), t.cols());

    std::map<std::string, Eigen::MatrixXd> grads = net.numerical_gradient(x, t);

    std::cout << grads["W1"].rows() << " " << grads["W1"].cols() << std::endl;
    std::cout << grads["b1"].rows() << " " << grads["b1"].cols() << std::endl;
    std::cout << grads["W2"].rows() << " " << grads["W2"].cols() << std::endl;
    std::cout << grads["b2"].rows() << " " << grads["b2"].cols() << std::endl;

    return 0;
}

#include <QCoreApplication>

#include "twolayernet.h"
#include "mnist.h"

Eigen::MatrixXd abs(const Eigen::MatrixXd &x) {
    Eigen::MatrixXd r(x.rows(), x.cols());
    for (int i = 0; i < x.size(); ++i) {
        r.data()[i] = abs(x.data()[i]);
    }

    return r;
}

double average(const Eigen::MatrixXd &x) {
    double r = 0;
    for (int i = 0; i < x.size(); ++i) {
        r += x.data()[i];
    }

    return r / x.size();
}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    // データの読み込み
    Mnist m;
    std::tuple<std::vector<Eigen::MatrixXd>, Eigen::MatrixXi, std::vector<Eigen::MatrixXd>, Eigen::MatrixXi> mnist;
    mnist = m.load_mnist(true, true, true);

    TwoLayerNet network(784, 50, 10);

    std::vector<Eigen::MatrixXd> x_train = std::get<0>(mnist);
    Eigen::MatrixXi t_train = std::get<1>(mnist);
    Eigen::MatrixXd x_batch(3, x_train[0].rows());
    for (uint i = 0; i < 3; ++i) {
        x_batch.row(i) = x_train[i].col(0);
    }
    Eigen::MatrixXd t_batch(3, 10);
    for (uint i = 0; i < 3; ++i) {
        t_batch.row(i) = t_train.row(i).cast<double>();
    }

    std::map<std::string, Eigen::MatrixXd> grad_numerical = network.numerical_gradient(x_batch, t_batch);
    std::map<std::string, Eigen::MatrixXd> grad_backprop = network.gradient(x_batch, t_batch);

    // 各重みの絶対誤差の平均を求める
    foreach (auto x, grad_numerical) {
        std::string key = x.first;
        double diff = average(abs(grad_backprop[key] - grad_numerical[key]));
        std::cout << key << ":" << diff << std::endl;
    }
}

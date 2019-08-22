#include <iostream>
#include <array>
#include <vector>

#include <QCoreApplication>
#include <QRandomGenerator>

#include <Eigen/Dense>

#include "mnist.h"
#include "twolayernet.h"

int main(int argc, char *argv[])
{
    Q_UNUSED(argc)
    Q_UNUSED(argv)

    Mnist m;
    std::tuple<std::vector<Eigen::MatrixXd>, Eigen::MatrixXi, std::vector<Eigen::MatrixXd>, Eigen::MatrixXi> mnist =
            m.load_mnist(true, true, true);

    std::vector<double> train_loss_list;

    // ハイパーパラメータ
    uint iters_num = 10; //10000;
    uint train_size = static_cast<uint>(std::get<0>(mnist).size());
    uint  batch_size = 10;  //100;
    double learning_rate = 0.1;

    TwoLayerNet network(784, 50, 10);

    for (uint i = 0; i < iters_num; ++i) {
        // ミニバッチの取得
        std::vector<uint> batch_mask(batch_size);
        for (uint j = 0; j < batch_size; ++j) {
            batch_mask[j] = static_cast<uint>(QRandomGenerator::global()->generate() % train_size);
        }
        Eigen::MatrixXd x_batch(batch_size, std::get<0>(mnist)[0].rows());
        Eigen::MatrixXi t_batch(batch_size, 10);
        for (uint j = 0; j < batch_size; ++j) {
            x_batch.row(j) = std::get<0>(mnist)[batch_mask[j]].col(0);
            t_batch.row(j) = std::get<1>(mnist).cast<int>().row(batch_mask[j]);
        }

        // 勾配の計算
        //std::map<std::string, Eigen::MatrixXd> grad = network.numerical_gradient(x_batch, t_batch.cast<double>());
        std::map<std::string, Eigen::MatrixXd> grad = network.gradient(x_batch, t_batch.cast<double>());

        // パラメータの更新
        const static std::array<std::string, 4> KEYS = { "W1", "b1", "W2", "b2" };
        foreach (std::string key, KEYS) {
            network.params()[key] -= (learning_rate * grad[key]);
        }

        // 学習経過の記録
        double loss = network.loss(x_batch, t_batch.cast<double>());
        train_loss_list.push_back(loss);
    }

    foreach (double l, train_loss_list) {
        std::cout << l << " ";
    }
    std::cout << std::endl;

    return 0;
}

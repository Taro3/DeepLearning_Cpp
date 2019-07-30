#include <iostream>

#include <QCoreApplication>
#include <QRandomGenerator>

#include <Eigen/Dense>

#include <mnist.h>

int main(int argc, char *argv[])
{
    Q_UNUSED(argc)
    Q_UNUSED(argv)

    Mnist m;

    // MNISTデータ読み込み
    std::tuple<std::vector<Eigen::MatrixXd>, Eigen::MatrixXi, std::vector<Eigen::MatrixXd>, Eigen::MatrixXi> data;
    data = m.load_mnist(true, true, true);

    std::cout << std::get<0>(data).size() << " " << std::get<0>(data)[0].size() << std::endl;
    std::cout << std::get<1>(data).rows() << " " << std::get<1>(data).cols() << std::endl;

    // バッチ抽出
    const std::vector<Eigen::MatrixXd> x_train = std::get<0>(data);
    const Eigen::MatrixXi t_train = std::get<1>(data);
    const uint train_size = static_cast<uint>(x_train.size());
    const uint batch_size = 10;
    uint batch_mask[batch_size];
    for (uint i = 0; i < batch_size; ++i) {
        batch_mask[i] = QRandomGenerator::global()->generate() % train_size;
    }
    std::vector<Eigen::MatrixXd> x_batch(batch_size);
    Eigen::MatrixXi t_batch(batch_size, 10);
    for (uint i = 0; i < batch_size; ++i) {
        x_batch[i] = x_train[batch_mask[i]];
        t_batch.row(i) = t_train.row(batch_mask[i]);
    }

    // テスト表示
    std::cout << x_batch[0] << std::endl;
    std::cout << t_batch << std::endl;

    return 0;
}

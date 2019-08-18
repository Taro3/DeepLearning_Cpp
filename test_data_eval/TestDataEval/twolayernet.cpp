#include <random>

#include <Eigen/Dense>

#include "twolayernet.h"

TwoLayerNet::TwoLayerNet(int input_size, int hidden_size, int output_size, double weigt_init_std, QObject *parent) :
    QObject(parent),
    _params(std::map<std::string, Eigen::MatrixXd>())
{
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());

    // 平均0.0、標準偏差1.0で分布させる
    std::normal_distribution<> dist(0.0, 1.0);

    Eigen::MatrixXd r1(input_size, hidden_size);
    for (int i = 0; i < r1.size(); ++i) {
        // 正規分布で乱数を生成する
        double result = dist(engine);
        r1.data()[i] = result;
    }
    Eigen::MatrixXd r2(hidden_size, output_size);
    for (int i = 0; i < r2.size(); ++i) {
        // 正規分布で乱数を生成する
        double result = dist(engine);
        r2.data()[i] = result;
    }

    //_params["W1"] = weigt_init_std * Eigen::MatrixXd::Random(input_size, hidden_size);
    _params["W1"] = weigt_init_std * r1;
    _params["b1"] = Eigen::VectorXd::Zero(hidden_size);
    //_params["W2"] = weigt_init_std * Eigen::MatrixXd::Random(hidden_size, output_size);
    _params["W2"] = weigt_init_std * r2;
    _params["b2"] = Eigen::VectorXd::Zero(output_size);
}

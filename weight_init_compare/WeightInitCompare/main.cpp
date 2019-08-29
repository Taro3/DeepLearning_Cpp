#include <QApplication>
#include <QRandomGenerator>

#include <Eigen/Dense>

#include "mainwindow.h"
#include "mnist.h"
#include "optimizer.h"
#include "multilayernet.h"
#include "plotgraph.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;

    // 0:MNISTデータの読み込み==========
    Mnist m;
    std::tuple<std::vector<Eigen::MatrixXd>, Eigen::MatrixXi, std::vector<Eigen::MatrixXd>, Eigen::MatrixXi> mnist;
    mnist = m.load_mnist();
    std::vector<Eigen::MatrixXd> x_train = std::get<0>(mnist);
    Eigen::MatrixXi t_train = std::get<1>(mnist);
    std::vector<Eigen::MatrixXd> x_test = std::get<2>(mnist);
    Eigen::MatrixXi t_test = std::get<3>(mnist);

    uint train_size = static_cast<uint>(x_train.size());
    uint batch_size = 128;
    uint max_iteration = 2000;

    // 1:実験の設定==========
    std::vector<std::pair<std::string, std::string> > weight_init_types = {
        { "std=0.01", "0.01" }, { "Xavier", "sigmoid" }, { "He", "relu" }, };
    OptimizerBase *optimizer = new SGD(0.01, &w);

    std::map<std::string, MultiLayerNet*> network;
    std::map<std::string, std::vector<double> > train_loss;
    for (std::pair<std::string, std::string> n : weight_init_types) {
        network[n.first] = new MultiLayerNet(784, std::vector<int>() = { 100, 100, 100, 100 }, 10, "relu", n.second, 0,
                                             &w);
        train_loss[n.first] = std::vector<double>();
    }

    // 2:訓練の開始==========
    for (uint i = 0; i < max_iteration; ++i) {
        std::vector<uint> batch_mask;
        for (uint j = 0; j < batch_size; ++j) {
            batch_mask.push_back(QRandomGenerator::global()->generate() % train_size);
        }
        Eigen::MatrixXd x_batch(batch_size, x_train[0].rows());
        for (uint j = 0; j < batch_size; ++j) {
            x_batch.row(j) = x_train[batch_mask[j]].col(0);
        }
        Eigen::MatrixXd t_batch(batch_size, t_train.cols());
        for (uint j = 0; j < batch_size; ++j) {
            t_batch.row(j) = t_train.row(batch_mask[j]).cast<double>();
        }

        for (std::pair<std::string, std::string> n : weight_init_types) {
            std::map<std::string, Eigen::MatrixXd> grads = network[n.first]->gradient(x_batch, t_batch);
            optimizer->update(network[n.first]->params(), grads);

            double loss = network[n.first]->loss(x_batch, t_batch);
            train_loss[n.first].push_back(loss);
        }

        if (i % 100 == 0) {
            std::cout << "========== iteration:" << std::to_string(i) << " ==========" << std::endl;
            for (std::pair<std::string, std::string> n : weight_init_types) {
                double loss = network[n.first]->loss(x_batch, t_batch);
                std::cout << n.first << ":" << std::to_string(loss) << std::endl;
            }
        }
    }

    // 3:グラフの描画==========
    std::vector<std::string> names = { "std=0.01", "Xavier", "He", };
    std::vector<PlotGraph*> pgs;
    uint idx = 0;
    for (std::string title : names) {
        PlotGraph *p = new PlotGraph(title.c_str());
        pgs.push_back(p);
        Eigen::VectorXd x(max_iteration);
        Eigen::VectorXd y(max_iteration);
        for (uint i = 0; i < max_iteration; ++i) {
            x(i) = i;
            y(i) = train_loss[weight_init_types[idx].first][i];
        }
        p->plot(x, y);
        ++idx;
    }

    w.show();

    int ret = a.exec();
    for (PlotGraph *p : pgs) {
        delete p;
    }

    return ret;
}

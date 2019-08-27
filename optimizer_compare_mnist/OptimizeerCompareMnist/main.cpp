#include <QApplication>
#include <QRandomGenerator>

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
    mnist = m.load_mnist(true);
    std::vector<Eigen::MatrixXd> x_train = std::get<0>(mnist);
    Eigen::MatrixXi t_train = std::get<1>(mnist);
    std::vector<Eigen::MatrixXd> x_test = std::get<2>(mnist);
    Eigen::MatrixXi t_test = std::get<3>(mnist);

    uint train_size = static_cast<uint>(x_train.size());
    uint batch_size = 128;
    int max_iterations = 2000;

    // 1:実験の設定==========
    std::vector<std::pair<std::string, OptimizerBase*> > optimizers;
    optimizers.push_back(std::pair<std::string, OptimizerBase*>("SGD", new SGD(0.01, &w)));
    optimizers.push_back(std::pair<std::string, OptimizerBase*>("Momentum", new Momentum(0.01, 0.9, &w)));
    optimizers.push_back(std::pair<std::string, OptimizerBase*>("AdaGrad", new AdaGrad(0.01, &w)));
    optimizers.push_back(std::pair<std::string, OptimizerBase*>("Adam", new Adam(0.001, 0.9, 0.999, &w)));

    std::map<std::string, MultiLayerNet*> network;
    std::map<std::string, std::vector<double>> train_loss;
    for (const std::pair<std::string, OptimizerBase*> &x : optimizers) {
        network[x.first] = new MultiLayerNet(784, std::vector<int>() = { 100, 100, 100, 100 }, 10, "relu", "relu", 0,
                                               &w);
        train_loss[x.first] = std::vector<double>();
    }

    // 2:訓練の開始==========
    for (int i = 0; i < max_iterations; ++i) {
        std::vector<uint> batch_mask;
        for (uint j = 0; j < batch_size; ++j) {
            batch_mask.push_back(QRandomGenerator::global()->generate() % train_size);
        }
        Eigen::MatrixXd x_batch(batch_size, x_train[0].rows());
        Eigen::MatrixXd t_batch(batch_size, t_train.cols());
        for (uint k = 0; k < batch_size; ++k) {
            x_batch.row(k) = x_train[batch_mask[k]].col(0);
            t_batch.row(k) = t_train.row(batch_mask[k]).cast<double>();
        }

        for (std::vector<std::pair<std::string, OptimizerBase*> >::const_iterator it = optimizers.begin();
             it != optimizers.end(); ++it) {
            std::map<std::string, Eigen::MatrixXd> grads = network[it->first]->gradient(x_batch, t_batch);
            it->second->update(network[it->first]->params(), grads);

            double loss = network[it->first]->loss(x_batch, t_batch);
            train_loss[it->first].push_back(loss);
        }

        if (i % 100 == 0) {
            std::cout << "========== iteration:" << std::to_string(i) << " ==========" << std::endl;
            std::for_each (optimizers.begin(), optimizers.end(),
                           [&network, &x_batch, &t_batch](std::pair<std::string, OptimizerBase*> x) {
                double loss = network[x.first]->loss(x_batch, t_batch);
                std::cout << x.first << ":" << std::to_string(loss) << std::endl;
            });
        }
    }

    // 3:グラフの描画==========
    //std::vector<std::string> markers = { "SGD", "Momentum", "AdaGrad", "Adam" };
    std::vector<PlotGraph*> widgets;
    Eigen::VectorXd px(max_iterations);
    for (int i = 0; i < max_iterations; ++i) {
        px(i) = i;
    }
    std::for_each (optimizers.begin(), optimizers.end(), [&widgets, &px, &train_loss](std::pair<std::string,
                   OptimizerBase*> x) {
        //for (uint i = 0; i < train_loss.size(); ++i) {
        Eigen::VectorXd py(train_loss[x.first].size());
        for (uint i = 0; i < train_loss[x.first].size(); ++i) {
            py(i) = train_loss[x.first][i];
        }
        PlotGraph *pg = new PlotGraph(x.first);
        pg->plot(px, py);
        widgets.push_back(pg);
    });

    w.show();

    int r = a.exec();
    for (PlotGraph *p : widgets) {
        delete p;
    }

    return r;
}

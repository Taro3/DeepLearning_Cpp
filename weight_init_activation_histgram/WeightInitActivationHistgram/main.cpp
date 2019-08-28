#include <iostream>

#include <QApplication>
#include <QtCharts>

#include <Eigen/Dense>

#include "mainwindow.h"

Eigen::MatrixXd sigmoid(Eigen::MatrixXd x) {
    return 1 / (1 + (-x).array().exp());
}

Eigen::MatrixXd ReLU(Eigen::MatrixXd x) {
    Eigen::MatrixXd r(x.rows(), x.cols());
    for (int i = 0; i < x.size(); ++i) {
        r.data()[i] = std::max(0.0, x.data()[i]);
    }
    return r;
}

Eigen::MatrixXd tanh(Eigen::MatrixXd x) {
    Eigen::MatrixXd r(x.rows(), x.cols());
    for (int i = 0; i < x.size(); ++i) {
        r.data()[i] = std::tanh(x.data()[i]);
    }
    return r;
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow win;

    Eigen::MatrixXd input_data = Eigen::MatrixXd::Random(1000, 100);    // 1000個のデータ
    uint node_num = 100;                                                // 各隠れ層のノード(ニューロン)の数
    uint hidden_layer_size = 5;                                         // 隠れ層が5層
    std::vector<Eigen::MatrixXd> activation;                            // ここにアクティベーションの結果を格納する

    Eigen::MatrixXd x = input_data;

    std::vector<QChartView*> cvs;

    for (uint i = 0; i < hidden_layer_size; ++i) {
        if (i) {
            x = activation[i - 1];
        }

        // 初期値の値をいろいろ変えて実験しよう！
//        Eigen::MatrixXd w = Eigen::MatrixXd::Random(node_num, node_num) * 1;
//        Eigen::MatrixXd w = Eigen::MatrixXd::Random(node_num, node_num) * 0.01;
//        Eigen::MatrixXd w = Eigen::MatrixXd::Random(node_num, node_num) * std::sqrt(1.0 / node_num);
        Eigen::MatrixXd w = Eigen::MatrixXd::Random(node_num, node_num) * std::sqrt(2.0 / node_num);

        Eigen::MatrixXd a = x * w;

        // 活性化関数の種類も実験しよう！
//        Eigen::MatrixXd z = sigmoid(a);
        Eigen::MatrixXd z = ReLU(a);
//        Eigen::MatrixXd z = tanh(a);

        activation.push_back(z);

        // ヒストグラムを描画
        QChartView *cv = new QChartView;
        cvs.push_back(cv);
        QBarSeries *bss = new QBarSeries(&win);
        QBarSet *bs = new QBarSet("", &win);
        Eigen::VectorXd n(30);
        n.setZero();
        double max = activation[i].maxCoeff();
        for (uint j = 0; j < activation[i].size(); ++j) {
            int idx = static_cast<int>(activation[i].data()[j] / max * 30);
            if (idx >= 0 && idx < 30) {
                ++n(idx);
            }
        }
        for (uint j = 0; j < 30; ++j) {
            *bs << n(j);
        }
        bss->append(bs);
        cv->chart()->addSeries(bss);
        cv->setWindowTitle(QString::number(i + 1) + "layer");
        cv->setGeometry(0, 0, 640, 480);
        cv->show();
    }

    win.show();

    int ret = a.exec();

    for (QChartView *v : cvs) {
        v->deleteLater();
    }

    return ret;
}

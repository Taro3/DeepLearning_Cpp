#include <tuple>
#include <memory>
#include <iostream>

#include <QApplication>
#include <QImage>

#include <Eigen/Dense>

#include "mainwindow.h"
#include "mnist.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    Mnist m;

    // MNISTデータ読み込み
    std::tuple<std::vector<Eigen::MatrixXd>, Eigen::MatrixXi, std::vector<Eigen::MatrixXd>, Eigen::MatrixXi> data;
    data = m.load_mnist(false, false);
    std::vector<Eigen::MatrixXd> imgs = std::get<0>(data);

    // 試しに画像として表示してみる
    std::shared_ptr<QImage> image(new QImage(static_cast<int>(imgs[0].cols()), static_cast<int>(imgs[0].rows()),
            QImage::Format_Grayscale8));
    for (int r = 0; r < imgs[0].rows(); ++r) {
        uchar *line = image->scanLine(r);
        for (int c = 0; c < imgs[0].cols(); ++c) {
            uchar p = static_cast<uchar>(imgs[0](r, c));
            line[c] = p;
        }
        std::cerr << std::endl;
    }

    MainWindow w;
    w.setImage(image);
    w.show();

    return a.exec();
}

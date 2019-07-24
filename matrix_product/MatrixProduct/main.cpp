#include <iostream>

#include <QCoreApplication>
#include <QDebug>

#include <Eigen/Dense>

int main(int argc, char *argv[])
{
    Q_UNUSED(argc);
    Q_UNUSED(argv);

    // 2 * 2のマトリクス同士の積(ドット積)
    {
        Eigen::MatrixXi A(2, 2);
        A <<    1, 2,
                3, 4;
        Eigen::MatrixXi B(2, 2);
        B <<    5, 6,
                7, 8;
        std::cout << A * B << std::endl;
    }

    std::cout << std::endl;

    // 3 * 2と2 * 3のマトリクスの積(ドット積)
    {
        Eigen::MatrixXi A(2, 3);
        A <<    1, 2, 3,
                4, 5, 6;
        Eigen::MatrixXi B(3, 2);
        B <<    1, 2,
                3, 4,
                5, 6;
        std::cout << A * B << std::endl;
    }

    std::cout << std::endl;

    // Aの1次元とBの0次元が異なる場合(エラー)
//    {
//        Eigen::MatrixXi A(3, 2);
//        A <<    1, 2, 3,
//                4, 5, 6;
//        Eigen::MatrixXi C(2, 2);
//        C <<    1, 2,
//                3, 4;
//        std::cout << A * C << std::endl;
//    }

//    std::cout << std::endl;

    // Aが2次元、Bが1次元の場合
    {
        Eigen::MatrixXi A(3, 2);
        A <<    1, 2,
                3, 4,
                5, 6;
        Eigen::VectorXi B(2);
        B << 7, 8;

        std::cout << A * B;
    }

    std::cout << std::endl;

    // ニューラルネットワークの行列の積
    {
        Eigen::RowVectorXi X(2);
        X << 1, 2;
        Eigen::MatrixXi W(2, 3);
        W <<    1, 3, 5,
                2, 4, 6;
        auto Y = X * W;
        std::cout << Y;
    }
}

#include <iostream>

#include <QCoreApplication>

#include <Eigen/Dense>

#include "softmaxfunction.h"

int main(int argc, char *argv[])
{
    Q_UNUSED(argc)
    Q_UNUSED(argv)

    {
        // 最初のソフトマックス関数のテスト
        Eigen::ArrayXd a(3);
        a << .3, 2.9, 4.0;
        Eigen::ArrayXd exp_a = a.exp();
        std::cout << exp_a << std::endl;

        std::cout << std::endl;

        double sum_exp_a = exp_a.sum();
        std::cout << sum_exp_a << std::endl;

        std::cout << std::endl;

        Eigen::ArrayXd y = exp_a / sum_exp_a;
        std::cout << y << std::endl;

        std::cout << std::endl <<std::endl;

        // 最初のソフトマックス関数の使用
        SoftmaxFunction sf;
        std::cout << sf.softmax1(a);
    }

    std::cout << std::endl <<std::endl;

    {
        // 2つ目のソフトマックス関数のテスト
        Eigen::ArrayXd a(3);
        a << 1010, 1000, 990;
        std::cout << a.exp() / a.exp().sum() << std::endl;  // 正しく計算されない(nanになってしまう)

        std::cout << std::endl;

        double c = a.maxCoeff();
        std::cout << a - c << std::endl;

        std::cout << std::endl;

        std::cout << (a - c).exp() / (a - c).exp().sum() << std::endl;

        std::cout << std::endl <<std::endl;

        // 2つ目のソフトマックス関数の使用
        SoftmaxFunction sf;
        std::cout << sf.softmax2(a) << std::endl;
    }

    std::cout << std::endl <<std::endl;

    {
        // ソフトマックス関数の使用(特徴)
        Eigen::ArrayXd a(3);
        a << 0.3, 2.9, 4.0;
        SoftmaxFunction sf;
        Eigen::ArrayXd y = sf.softmax2(a);
        std::cout << y << std::endl;
        std::cout << y.sum() << std::endl;
    }
}

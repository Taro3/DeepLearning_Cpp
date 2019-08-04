#include <iostream>

#include <QCoreApplication>

#include <Eigen/Dense>

// 勾配の算出処理
Eigen::ArrayXd numerical_gradient(std::function<double(Eigen::ArrayXd)> f, Eigen::ArrayXd x) {
    double h = 1e-4;    // 0.0001
    Eigen::ArrayXd grad(x.size());
    grad.setZero();

    for (int idx = 0; idx < x.size(); ++idx) {
        double tmp_val = x(idx);
        // f(x+h)の計算
        x(idx) = tmp_val + h;
        double fxh1 = f(x);

        // f(x-h)の計算
        x(idx) = tmp_val - h;
        double fxh2 = f(x);

        grad(idx) = (fxh1 - fxh2) / (2 * h);
        x(idx) = tmp_val;   // 値を元に戻す
    }

    return grad;
}

int main(int argc, char *argv[])
{
    Q_UNUSED(argc)
    Q_UNUSED(argv)

    // 式(4.6)の関数(勾配を求める関数)
    std::function<double(Eigen::ArrayXd x)> function_2 = [](Eigen::ArrayXd x) { return pow(x(0), 2) + pow(x(1), 2); };

    Eigen::ArrayXd x(2);
    x << 3.0, 4.0;
    std::cout << numerical_gradient(function_2, x) << std::endl;

    x << 0.0, 2.0;
    std::cout << numerical_gradient(function_2, x) << std::endl;

    x << 3.0, 0.0;
    std::cout << numerical_gradient(function_2, x) << std::endl;

    return 0;
}

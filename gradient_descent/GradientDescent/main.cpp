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

// 勾配降下法算出処理
Eigen::ArrayXd gradient_descent(std::function<double(Eigen::ArrayXd)> f, Eigen::ArrayXd init_x, double lr = 0.01,
                                int step_num = 100) {
    Eigen::ArrayXd x = init_x;

    for (int i = 0; i < step_num; ++i) {
        Eigen::ArrayXd grad = numerical_gradient(f, x);
        x -= lr * grad;
    }

    return x;
}

int main(int argc, char *argv[])
{
    Q_UNUSED(argc)
    Q_UNUSED(argv)

    std::function<double(Eigen::ArrayXd)> function_2 = [](Eigen::ArrayXd x) { return pow(x(0), 2) + pow(x(1), 2); };

    // 問:f(x0,x1)=x0^2+x1^2の最小値を勾配法で求めよ。
    Eigen::ArrayXd init_x(2);
    init_x << -3.0, 4.0;
    std::cout << gradient_descent(function_2, init_x, 0.1, 100) << std::endl;

    std::cout << std::endl;

    // 学習率が大きすぎる例:lr=10.0
    std::cout << gradient_descent(function_2, init_x, 10.0, 100) << std::endl;

    std::cout << std::endl;

    // 学習率が小さすぎる例:lr=1e-10
    std::cout << gradient_descent(function_2, init_x, 1e-10, 100) << std::endl;

    return 0;
}

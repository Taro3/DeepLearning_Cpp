#include <iostream>

#include <QCoreApplication>

#include <Eigen/Dense>

// 式(4.6)の関数
double function_2(Eigen::ArrayXd x) {
    return pow(x(0), 2) + pow(x(1), 2);
}

double numerical_diff(std::function<double(double)> f, double x)
{
    double h = 1e-004;  // 0.0001;
    return  (f(x + h) - f(x - h)) / (2 * h);
}

int main(int argc, char *argv[])
{
    Q_UNUSED(argc)
    Q_UNUSED(argv)

    Eigen::ArrayXd x1(2);
    x1 << 1, 2;
    std::cout << function_2(x1) << std::endl;

    // 問1:x0=3,x1=4のときのx0に対する偏微分∂f/∂x0を求める。
    std::cout << numerical_diff([](double x0) { return x0 * x0 + pow(4.0, 2); }, 3.0) << std::endl;

    // 問2:x0=3,x1=4のときのx1に対する偏微分∂f/∂x1を求める。
    std::cout << numerical_diff([](double x1) { return pow(3.0, 2) + x1 * x1; }, 4.0) << std::endl;

    return 0;
}

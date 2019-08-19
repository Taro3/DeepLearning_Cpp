#include <iostream>

#include <QCoreApplication>

#include "layer_naive.h"

int main(int argc, char *argv[])
{
    Q_UNUSED(argc)
    Q_UNUSED(argv)

    double apple = 100;
    double apple_num = 2;
    double tax = 1.1;

    MulLayer mul_apple_layer;
    MulLayer mul_tax_layer;

    // forword
    double apple_price = mul_apple_layer.forword(apple, apple_num);
    double price = mul_tax_layer.forword(apple_price, tax);

    // backword
    double dprice = 1;
    std::tuple<double, double> d1 = mul_tax_layer.backword(dprice);
    double dapple_price = std::get<0>(d1);
    double dtax = std::get<1>(d1);
    std::tuple<double, double> d2 = mul_apple_layer.backword(dapple_price);
    double dapple = std::get<0>(d2);
    double dapple_num = std::get<1>(d2);

    std::cout << "price:" << static_cast<int>(price) << std::endl;
    std::cout << "dApple:" << dapple << std::endl;
    std::cout << "dApple_num:" << static_cast<int>(dapple_num) << std::endl;
    std::cout << "dTax:" << dtax << std::endl;
}

#include <iostream>

#include <QCoreApplication>

#include "layer_naive.h"

int main(int argc, char *argv[])
{
    Q_UNUSED(argc)
    Q_UNUSED(argv)

    double apple = 100;
    double apple_num = 2;
    double orange = 150;
    double orange_num = 3;
    double tax = 1.1;

    // layer
    MulLayer mul_apple_layer;
    MulLayer mul_orange_layer;
    AddLayer add_apple_layer;
    MulLayer mul_tax_layer;

    // forword
    double apple_price = mul_apple_layer.forword(apple, apple_num);
    double orange_price = mul_orange_layer.forword(orange, orange_num);
    double all_price = add_apple_layer.forword(apple_price, orange_price);
    double price = mul_tax_layer.forword(all_price, tax);

    // backword
    double dprice = 1;
    std::tuple<double, double> d1 = mul_tax_layer.backword(dprice);
    double dall_price = std::get<0>(d1);
    double dtax = std::get<1>(d1);
    std::tuple<double, double> d2 = add_apple_layer.backword(dall_price);
    double dapple_price = std::get<0>(d2);
    double dorange_price = std::get<1>(d2);
    std::tuple<double, double> d3 = mul_orange_layer.backword(dorange_price);
    double dorange = std::get<0>(d3);
    double dorange_num = std::get<1>(d3);
    std::tuple<double, double> d4 = mul_apple_layer.backword(dapple_price);
    double dapple = std::get<0>(d4);
    double dapple_num = std::get<1>(d4);

    std::cout << "price:" << static_cast<int>(price) << std::endl;
    std::cout << "dApple:" << dapple << std::endl;
    std::cout << "dApple_num:" << static_cast<int>(dapple_num) << std::endl;
    std::cout << "dOrange:" << dorange << std::endl;
    std::cout << "dOrange_num:" << static_cast<int>(dorange_num) << std::endl;
    std::cout << "dTax:" << dtax << std::endl;

    return 0;
}

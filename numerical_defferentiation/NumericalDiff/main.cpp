#include <iostream>

#include <QApplication>
#include <QDebug>

#include <Eigen/Dense>

#include "mainwindow.h"
#include "plotgraph.h"

template<typename T>
T function_1(T x)
{
    return 0.01 * x * x + 0.1 * x;
}

template<typename T>
double numerical_diff(T x)
{
    double h = 1e-004;  // 0.0001;
    return  (function_1(x + h) - function_1(x - h)) / (2 * h);
}

int main(int argc, char *argv[])
{
    Eigen::ArrayXd x(201);
    x.setLinSpaced(201, 0, 20);
    Eigen::ArrayXd y = function_1(x);

    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    PlotGraph pg;
    pg.plot(x, y, &w);  // 0.01x^2+0.1xのグラフを描画する

    qInfo() << numerical_diff(5.0);     // 5の微分
    qInfo() << numerical_diff(10.0);    // 10の微分

    return a.exec();
}

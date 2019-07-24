#include <iostream>

#include <QCoreApplication>
#include <QMap>

#include <Eigen/Dense>

#include "activationfunction.h"

QMap<QString, Eigen::MatrixXd> init_network()
{
    QMap<QString, Eigen::MatrixXd> network;
    network["W1"] = Eigen::MatrixXd(2, 3);
    network["W1"] <<    .1, .3, .5,
                        .2, .4, .6;
    network["b1"] = Eigen::RowVectorXd(3);
    network["b1"] << .1, .2, .3;
    network["W2"] = Eigen::MatrixXd(3, 2);
    network["W2"] <<    .1, .4,
                        .2, .5,
                        .3, .6;
    network["b2"] = Eigen::RowVectorXd(2);
    network["b2"] << .1, .2;
    network["W3"] = Eigen::MatrixXd(2, 2);
    network["W3"] <<    .1, .3,
                        .2, .4;
    network["b3"] = Eigen::RowVectorXd(2);
    network["b3"] << .1, .2;

    return network;
}

Eigen::VectorXd identity_function(Eigen::VectorXd x)
{
    return x;
}

Eigen::RowVectorXd forword(QMap<QString, Eigen::MatrixXd> network, Eigen::RowVectorXd x)
{
    Eigen::MatrixXd W1 = network["W1"];
    Eigen::MatrixXd W2 = network["W2"];
    Eigen::MatrixXd W3 = network["W3"];
    Eigen::RowVectorXd b1 = network["b1"];
    Eigen::RowVectorXd b2 = network["b2"];
    Eigen::RowVectorXd b3 = network["b3"];

    ActivationFunction af;

    Eigen::VectorXd a1 = x * W1 + b1;
    Eigen::RowVectorXd z1 = af.sigmoid(a1.array());
    Eigen::VectorXd a2 = z1 * W2 + b2;
    Eigen::RowVectorXd z2 = af.sigmoid(a2.array());
    Eigen::VectorXd a3 = z2 * W3 + b3;
    Eigen::VectorXd y = identity_function(a3);

    return y;
}

int main(int argc, char *argv[])
{
    Q_UNUSED(argc)
    Q_UNUSED(argv)

    QMap<QString, Eigen::MatrixXd> network = init_network();
    Eigen::RowVectorXd x(2);
    x << 1.0, 0.5;
    Eigen::RowVectorXd y = forword(network, x);
    std::cout << y << std::endl;
}

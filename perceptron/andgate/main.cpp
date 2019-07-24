#include <iostream>

#include <QCoreApplication>
#include <QDebug>

#include <Eigen/Dense>

#include "gate.h"

int main(int argc, char *argv[])
{
    Q_UNUSED(argc);
    Q_UNUSED(argv);

    Gate gate;
    qInfo() << gate.AND(0, 0);
    qInfo() << gate.AND(1, 0);
    qInfo() << gate.AND(0, 1);
    qInfo() << gate.AND(1, 1);

    qInfo() << gate.XOR(0, 0);
    qInfo() << gate.XOR(1, 0);
    qInfo() << gate.XOR(0, 1);
    qInfo() << gate.XOR(1, 1);
}

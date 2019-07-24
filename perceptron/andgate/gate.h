#ifndef GATE_H
#define GATE_H

#include <QObject>

#include <Eigen/Dense>

class Gate : public QObject
{
    Q_OBJECT
public:
    explicit Gate(QObject *parent = nullptr);
    int AND(double x1, double x2) {
        Eigen::ArrayXd x(2);
        x << x1, x2;
        Eigen::ArrayXd w(2);
        w << 0.5, 0.5;
        double b = -0.7;
        double tmp = (x * w).sum() + b;
        if (tmp <= 0) {
            return 0;
        } else {
            return 1;
        }
    }
    int NAND(double x1, double x2) {
        Eigen::ArrayXd x(2);
        x << x1, x2;
        Eigen::ArrayXd w(2);
        w << -0.5, -0.5;
        double b = 0.7;
        double tmp = (w * x).sum() + b;
        if (tmp <= 0) {
            return 0;
        } else {
            return 1;
        }
    }
    int OR(double x1, double x2) {
        Eigen::ArrayXd x(2);
        x << x1, x2;
        Eigen::ArrayXd w(2);
        w << 0.5, 0.5;
        double b = -0.2;
        double tmp = (w * x).sum() + b;
        if (tmp <= 0) {
            return 0;
        } else {
            return 1;
        }
    }
    int XOR(double x1, double x2) {
        int s1 = NAND(x1, x2);
        int s2 = OR(x1, x2);
        int y = AND(s1, s2);
        return y;
    }

signals:

public slots:
};

#endif // GATE_H

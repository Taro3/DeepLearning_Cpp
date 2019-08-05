#ifndef SIMPLENET_H
#define SIMPLENET_H

#include <iostream>

#include <QObject>

#include <Eigen/Dense>

class simpleNet : public QObject
{
    Q_OBJECT
public:
    Eigen::MatrixXd _W;

    explicit simpleNet(QObject *parent = nullptr);
    Eigen::MatrixXd predict(Eigen::MatrixXd x) {
        return x * _W;
    }
    double loss(Eigen::MatrixXd x, Eigen::MatrixXd t) {
        Eigen::MatrixXd z = predict(x);
        Eigen::MatrixXd y = softmax(z);
        double loss = cross_entry_error(y, t);

        return loss;
    }

signals:

public slots:

private:
    Eigen::MatrixXd softmax(Eigen::MatrixXd x) {
        if (x.rows() > 1 && x.cols() > 1) {
            Eigen::ArrayXd xx(x.rows());
            for (int r = 0; r < x.rows(); ++r) {
                xx(r) = x.row(r).maxCoeff();
            }
            Eigen::MatrixXd y = xx.exp() / xx.exp().sum();
            return y;
        }

        auto xx = x - x.col(0).maxCoeff() * Eigen::MatrixXd::Ones(x.rows(), x.cols());
        return xx.array().exp() / xx.array().exp().sum();
    }

    double cross_entry_error(Eigen::MatrixXd y, Eigen::MatrixXd t) {
        double delta = 1e-7;
        Eigen::MatrixXd x = delta * Eigen::MatrixXd::Ones(y.rows(), y.cols());
        Eigen::MatrixXd xx = y + x;
        xx = xx.array().log();
        if (xx.rows() != 1) {
            xx.transposeInPlace();
        }
        return -(t * xx.transpose()).sum();
    }
};

#endif // SIMPLENET_H

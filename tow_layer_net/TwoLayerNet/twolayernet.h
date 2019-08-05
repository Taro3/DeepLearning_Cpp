#ifndef TWOLAYERNET_H
#define TWOLAYERNET_H

#include <iostream>

#include <QObject>
#include <QMap>

#include <Eigen/Dense>

class TwoLayerNet : public QObject
{
    Q_OBJECT
public:
    std::map<std::string, Eigen::MatrixXd> _params;

    explicit TwoLayerNet(int input_size, int hidden_size, int output_size, double weigt_init_std = 0.01, QObject *parent = nullptr);
    Eigen::MatrixXd predict(Eigen::MatrixXd x) {
        Eigen::MatrixXd W1 = _params["W1"];
        Eigen::MatrixXd W2 = _params["W2"];
        Eigen::MatrixXd b1 = _params["b1"];
        Eigen::MatrixXd b2 = _params["b2"];

        Eigen::MatrixXd at1 = x * W1;
        b1.resize(at1.rows(), at1.cols());
        Eigen::MatrixXd a1 = at1 + b1;
        Eigen::MatrixXd z1 = sigmoid(a1);
        Eigen::MatrixXd at2 = z1 * W2;
        b2.resize(at2.rows(), at2.cols());
        Eigen::MatrixXd a2 = at2 + b2;
        Eigen::MatrixXd y = softmax(a1);

        return y;
    }

    // x:入力データ t:教師データ
    double loss(Eigen::MatrixXd x, Eigen::MatrixXd t) {
        Eigen::MatrixXd y = predict(x);

        return cross_entry_error(y, t);
    }

    double accuracy(Eigen::MatrixXd x, Eigen::MatrixXd t) {
        Eigen::MatrixXd y = predict(x);
        y = y.colwise().maxCoeff();
        t = t.colwise().maxCoeff();

        double accuracy = (y.array() == t.array()).cast<int>().sum() / x.rows();
        return accuracy;
    }

    // x:入力データ t:教師データ
    std::map<std::string, Eigen::MatrixXd> numerical_gradient(Eigen::MatrixXd x, Eigen::MatrixXd t) {
        std::function<double(Eigen::MatrixXd)> loss_W = [this, &x, &t](Eigen::MatrixXd) { return this->loss(x, t); };
        std::map<std::string, Eigen::MatrixXd> grads;
        grads["W1"] = numerical_gradient(loss_W, _params["W1"]);
        grads["b1"] = numerical_gradient(loss_W, _params["b1"]);
        grads["W2"] = numerical_gradient(loss_W, _params["W2"]);
        grads["b2"] = numerical_gradient(loss_W, _params["b2"]);

        return grads;
    }

signals:

public slots:

private:
    Eigen::MatrixXd sigmoid(Eigen::MatrixXd x) {
        return 1 / (1 + (-x.array()).exp());
    }

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
        return -(xx * t).sum();
    }

    Eigen::MatrixXd numerical_gradient(std::function<double(Eigen::MatrixXd)> f, Eigen::MatrixXd x) {
        double h = 1e-4;    // 0.0001
        Eigen::MatrixXd grad(x);
        grad.setZero();

        for (int idx = 0; idx < x.size(); ++idx) {
            double tmp_val = x.data()[idx];
            // f(x+h)の計算
            x.data()[idx] = tmp_val + h;
            double fxh1 = f(x);

            // f(x-h)の計算
            x.data()[idx] = tmp_val - h;
            double fxh2 = f(x);

            grad.data()[idx] = (fxh1 - fxh2) / (2 * h);
            x.data()[idx] = tmp_val;   // 値を元に戻す
            std::cout << idx << "/" << x.size() << '\r';
        }
        std::cout << std::endl;

        return grad;
    }
};

#endif // TWOLAYERNET_H

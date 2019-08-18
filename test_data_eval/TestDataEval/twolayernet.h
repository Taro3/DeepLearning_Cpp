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
    Eigen::MatrixXd predict(const Eigen::MatrixXd &x) {
        const Eigen::MatrixXd &W1 = _params["W1"];
        const Eigen::MatrixXd &W2 = _params["W2"];
        Eigen::MatrixXd b1 = _params["b1"];
        Eigen::MatrixXd b2 = _params["b2"];

        Eigen::MatrixXd at1 = x * W1;
        b1.transposeInPlace();
        b1.conservativeResize(at1.rows(), Eigen::NoChange);
        for (int i = 0; i < b1.rows() - 1; ++i) {
            b1.row(i + 1) = b1.row(0);
        }
        Eigen::MatrixXd a1 = at1 + b1;
        Eigen::MatrixXd z1 = sigmoid(a1);
        Eigen::MatrixXd at2 = z1 * W2;
        b2.transposeInPlace();
        b2.conservativeResize(at2.rows(), Eigen::NoChange);
        for (int i = 0; i < b2.rows() - 1; ++i) {
            b2.row(i + 1) = b2.row(0);
        }
        Eigen::MatrixXd a2 = at2 + b2;
        Eigen::MatrixXd y = softmax(a2);

        return y;
    }

    // x:入力データ t:教師データ
    double loss(const Eigen::MatrixXd &x, const Eigen::MatrixXd &t) {
        Eigen::MatrixXd y = predict(x);

        return cross_entry_error(y, t);
    }

    double accuracy(const Eigen::MatrixXd &x, Eigen::MatrixXd t) {
        Eigen::MatrixXd y = predict(x);
        double ymin = y.minCoeff();
        Eigen::MatrixXi yargmax(y.rows(), 1);
        for (int i = 0; i < y.rows(); ++i) {
            double ymax = ymin;
            int ymaxidx = 0;
            for (int c = 0; c < y.cols(); ++c) {
                if (ymax < y(i, c)) {
                    ymax = y(i,c);
                    ymaxidx = c;
                }
            }
            yargmax(i, 0) = ymaxidx;
        }
        Eigen::MatrixXi targmax(t.rows(), 1);
        double tmin = t.minCoeff();
        for (int i = 0; i < t.rows(); ++i) {
            double tmax = tmin;
            int tmaxidx = 0;
            for (int c = 0; c < t.cols(); ++c) {
                if (tmax < t(i, c)) {
                    tmax = t(i, c);
                    tmaxidx = c;
                }
            }
            targmax(i, 0) = tmaxidx;
        }
        //y = y.rowwise().maxCoeff();
        //t = t.rowwise().maxCoeff();

        double accuracy = static_cast<double>((yargmax.array() == targmax.array()).cast<int>().sum()) / x.rows();
        return accuracy;
    }

    // x:入力データ t:教師データ
    std::map<std::string, Eigen::MatrixXd> numerical_gradient(const Eigen::MatrixXd &x, const Eigen::MatrixXd &t) {
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
    Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &x) {
        return 1 / (1 + ((-x).array()).exp());
    }

    Eigen::MatrixXd softmax(Eigen::MatrixXd x) {
        if (x.rows() > 1 && x.cols() > 1) {
            x.transposeInPlace();
            Eigen::MatrixXd x_max = x.colwise().maxCoeff();
            for (int r = 0; r < x.rows(); ++r) {
                for (int c = 0; c < x.cols(); ++c) {
                    x(r, c) = x(r, c) - x_max(0, c);
                }
            }
            Eigen::MatrixXd x_exp = x.array().exp();
            Eigen::VectorXd x_sum(x.cols());
            for (int c = 0; c < x_exp.cols(); ++c) {
                x_sum(c) = x_exp.col(c).sum();
            }
            Eigen::MatrixXd y(x.rows(), x.cols());
            for (int r = 0; r < x_exp.rows(); ++r) {
                for (int c = 0; c < x_exp.cols(); ++c) {
                    y(r, c) = x_exp(r, c) / x_sum(c);
                }
            }
            return y.transpose();
        }

        auto xx = x - x.col(0).maxCoeff() * Eigen::MatrixXd::Ones(x.rows(), x.cols());
        return xx.array().exp() / xx.array().exp().sum();
    }

    double cross_entry_error(const Eigen::MatrixXd &y, const Eigen::MatrixXd &t) {
        Eigen::MatrixXi tt(t.rows(), 1);
        Eigen::MatrixXi ti = t.cast<int>();
        for (int r = 0; r < ti.rows(); ++r) {
            for (int c = 0; c < ti.cols(); ++c) {
                if (ti(r, c)) {
                    tt(r, 0) = c;
                    break;
                }
            }
        }

        int batch_size = static_cast<int>(y.rows());
        Eigen::RowVectorXd ly(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            ly(i) = log(y(i, tt(i, 0)) + 1e-7);
        }

        return (-ly).sum() / batch_size;
    }

    Eigen::MatrixXd numerical_gradient(std::function<double(Eigen::MatrixXd)> f, Eigen::MatrixXd &x) {
        double h = 1e-4;    // 0.0001
        Eigen::MatrixXd grad(x.rows(), x.cols());
        grad.setZero();

        double *px = x.data();
        for (int i = 0; i < x.size(); ++i, ++px) {
            double tmp_val = *px;
            // f(x+h)の計算
            *px = tmp_val + h;
            double fxh1 = f(x);

            // f(x-h)の計算
            *px = tmp_val - h;
            double fxh2 = f(x);

            grad.data()[i] = (fxh1 - fxh2) / (2 * h);
            *px = tmp_val;   // 値を元に戻す
            //std::cout << x.cols() * r + c << "/" << x.size() << '\r';
        }

        return grad;
    }
};

#endif // TWOLAYERNET_H

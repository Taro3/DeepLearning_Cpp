#ifndef LAYERS_H
#define LAYERS_H

#include <QObject>

#include <Eigen/Dense>

class layerBase : public QObject
{
    Q_OBJECT
public:
    layerBase(QObject *parent = nullptr);
    virtual Eigen::MatrixXd forword(Eigen::MatrixXd &x) = 0;
    virtual Eigen::MatrixXd backword(Eigen::MatrixXd &dout) = 0;

    virtual ~layerBase() {}
};

/**
 * @brief The Relu class
 */
class Relu : public layerBase
{
    Q_OBJECT
public:
    explicit Relu(QObject *parent = nullptr);
    Eigen::MatrixXd forword(Eigen::MatrixXd &x) {
        _mask = (x.array() <= 0).cast<int>();
        Eigen::MatrixXd out = x;
        for (int i = 0; i < out.size(); ++i) {
            if (_mask.data()[i]) {
                out.data()[i] = 0;
            }
        }

        return out;
    }
    Eigen::MatrixXd backword(Eigen::MatrixXd &dout) {
        for (int i = 0; i < dout.size(); ++i) {
            if (_mask.data()[i]) {
                dout.data()[i] = 0;
            }
        }
        Eigen::MatrixXd dx = dout;

        return dx;
    }

signals:

public slots:

private:
    Eigen::MatrixXi _mask;
};

/**
 * @brief The Sigmoid class
 */
class Sigmoid : public layerBase
{
    Q_OBJECT
public:
    explicit Sigmoid(QObject *parent = nullptr);
    Eigen::MatrixXd forword(Eigen::MatrixXd &x) {
        Eigen::MatrixXd out = 1 / (1 + (-x).array().exp());
        _out = out;

        return out;
    }
    Eigen::MatrixXd backword(Eigen::MatrixXd &dout) {
        Eigen::MatrixXd t = 1 * Eigen::MatrixXd::Ones(_out.rows(), _out.cols());
        Eigen::MatrixXd dx = dout * (t - _out) * _out;

        return dx;
    }

signals:

public slots:

private:
    Eigen::MatrixXd _out;
};

/**
 * @brief The Affine class
 */
class Affine : public layerBase
{
    Q_OBJECT
public:
    explicit Affine(Eigen::MatrixXd &W, Eigen::MatrixXd &b, QObject *parent = nullptr);
    Eigen::MatrixXd forword(Eigen::MatrixXd &x) {
        _x = x;
        Eigen::MatrixXd out = (x * *_W);
        Eigen::MatrixXd temp_b(out.rows(), out.cols());
        Eigen::MatrixXd trans_b = (*_b).transpose();
        for (int i = 0; i < temp_b.rows(); ++i) {
            temp_b.row(i) = trans_b.row(0);
        }
        out += temp_b;

        return out;
    }
    Eigen::MatrixXd backword(Eigen::MatrixXd &dout) {
        Eigen::MatrixXd dx = dout * (*_W).transpose();
        _dW = _x.transpose() * dout;
        _db.resize(dout.cols(), 1);
        for (int i = 0; i < dout.cols(); ++i) {
            _db(i, 0) = dout.col(i).sum();
        }

        return dx;
    }
    Eigen::MatrixXd dW() const {
        return _dW;
    }
    Eigen::MatrixXd db() const {
        return _db;
    }

signals:

public slots:

private:
    Eigen::MatrixXd *_W;
    Eigen::MatrixXd *_b;
    Eigen::MatrixXd _x;
    Eigen::MatrixXd _dW;
    Eigen::MatrixXd _db;
};

/**
 * @brief The SoftmaxWithLoss class
 */
class SoftmaxWithLoss : public QObject
{
    Q_OBJECT
public:
    explicit SoftmaxWithLoss(QObject *parent = nullptr);
    double forword(const Eigen::MatrixXd &x, const Eigen::MatrixXd &t) {
        _t = t;
        _y = softmax(x);
        _loss = cross_entropy_error(_y, _t);

        return _loss;
    }
    Eigen::MatrixXd backword(const double &dout = 1) {
        (void)dout;
        int batch_size = static_cast<int>(_t.rows());
        Eigen::MatrixXd dx = (_y - _t) / batch_size;

        return dx;
    }

signals:

public slots:

private:
    double _loss;       // 損失
    Eigen::MatrixXd _y; // softmaxの出力
    Eigen::MatrixXd _t; // 教師データ(one-hot vector)

    double cross_entropy_error(const Eigen::MatrixXd &y, const Eigen::MatrixXd &t) {
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
};

#endif // LAYERS_H

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <QObject>

#include <Eigen/Dense>

/**
 * @brief The OptimizerBase class
 */
class OptimizerBase : public QObject
{
    Q_OBJECT
public:
    OptimizerBase(QObject *parent = nullptr);
    virtual void update(std::map<std::string, Eigen::MatrixXd> &params, std::map<std::string, Eigen::MatrixXd> grads) = 0;
    virtual ~OptimizerBase() {}
};

/**
 * @brief The SDG class
 */
class SGD : public OptimizerBase
{
    Q_OBJECT
public:
    explicit SGD(double lr = 0.01, QObject *parent = nullptr);

    void update(std::map<std::string, Eigen::MatrixXd> &params, std::map<std::string, Eigen::MatrixXd> grads)
    {
        for (std::map<std::string, Eigen::MatrixXd>::iterator it = params.begin(); it != params.end(); ++it) {
            it->second -= _lr * grads[it->first];
        }
    }

signals:

public slots:

private:
    double _lr;
};

/**
 * @brief The Momentum class
 */
class Momentum : public OptimizerBase
{
    Q_OBJECT
public:
    explicit Momentum(double lr = 0.01, double momentum = 0.9, QObject *parent = nullptr);

    void update(std::map<std::string, Eigen::MatrixXd> &params, std::map<std::string, Eigen::MatrixXd> grads)
    {
        if (_v.empty()) {
            std::for_each (params.begin(), params.end(), [this](std::pair<std::string, Eigen::MatrixXd> x) {
                Eigen::MatrixXd s(x.second.rows(), x.second.cols());
                s.setZero();
                _v[x.first] = s;
            });
        }

        for (std::map<std::string, Eigen::MatrixXd>::iterator it = params.begin(); it != params.end(); ++it) {
            _v[it->first] = _momentum * _v[it->first] - _lr * grads[it->first];
            it->second += _v[it->first];
        }
    }

signals:

public slots:

private:
    double _lr;
    double _momentum;
    std::map<std::string, Eigen::MatrixXd> _v;
};

class AdaGrad : public OptimizerBase
{
    Q_OBJECT
public:
    explicit AdaGrad(double lr = 0.01, QObject *parent = nullptr);

    void update(std::map<std::string, Eigen::MatrixXd> &params, std::map<std::string, Eigen::MatrixXd> grads)
    {
        if (_h.empty()) {
            std::for_each (params.begin(), params.end(), [this](std::pair<std::string, Eigen::MatrixXd> x) {
                Eigen::MatrixXd v(x.second.rows(), x.second.cols());
                v.setZero();
                _h[x.first] = v;
            });
        }

        for (std::map<std::string, Eigen::MatrixXd>::iterator it = params.begin(); it != params.end(); ++it) {
            for (int i = 0; i < grads[it->first].size(); ++i) {
                double t = grads[it->first].data()[i];
                _h[it->first].data()[i] += t * t;
            }
            Eigen::MatrixXd s = _lr * grads[it->first];
            Eigen::MatrixXd t = _h[it->first].array().sqrt();
            for (int r = 0; r < s.rows(); ++r) {
                for (int c = 0; c < s.cols(); ++c) {
                    s(r, c) /= (t(r, c) + 1e-7);
                }
            }
            it->second -= s;
        }
    }

signals:

public slots:

private:
    double _lr;
    std::map<std::string, Eigen::MatrixXd> _h;
};

/**
 * @brief The Adam class
 */
class Adam : public OptimizerBase
{
    Q_OBJECT
public:
    explicit Adam(double lr = 0.01, double beta1 = 0.9, double beta2 = 0.999, QObject *parent = nullptr);

    void update(std::map<std::string, Eigen::MatrixXd> &params, std::map<std::string, Eigen::MatrixXd> grads)
    {
        if (_m.empty()) {
            std::for_each (params.begin(), params.end(), [this](std::pair<std::string, Eigen::MatrixXd> x) {
                Eigen::MatrixXd m = Eigen::MatrixXd::Zero(x.second.rows(), x.second.cols());
                _m[x.first] = m;
                Eigen::MatrixXd v = Eigen::MatrixXd::Zero(x.second.rows(), x.second.cols());
                _v[x.first] = v;
            });
        }

        ++_iter;
        double lr_t = _lr * sqrt(1.0 - pow(_beta2, _iter)) / (1.0 - pow(_beta1, _iter));

        for (std::map<std::string, Eigen::MatrixXd>::iterator it = params.begin(); it != params.end(); ++it) {
            _m[it->first] += (1 - _beta1) * (grads[it->first] - _m[it->first]);
            Eigen::MatrixXd vt = grads[it->first].array().pow(2);
            _v[it->first] += (1 - _beta2) * (vt - _v[it->first]);

            Eigen::MatrixXd t1 = lr_t * _m[it->first];
            Eigen::MatrixXd t2 = _v[it->first].array().sqrt() + 1e-7;
            for (int r = 0; r < t1.rows(); ++r) {
                for (int c = 0; c < t1.cols(); ++c) {
                    t1(r, c) /= t2(r, c);
                }
            }
            it->second -= t1;
        }
    }

signals:

public slots:

private:
    double _lr;
    double _beta1;
    double _beta2;
    int _iter;
    std::map<std::string, Eigen::MatrixXd> _m;
    std::map<std::string, Eigen::MatrixXd> _v;
};

#endif // OPTIMIZER_H

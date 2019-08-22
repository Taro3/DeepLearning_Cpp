#ifndef TWOLAYERNET_H
#define TWOLAYERNET_H

#include <vector>
#include <iostream>

#include <QObject>

#include <Eigen/Dense>

#include "layers.h"

template<typename TKEY, typename TVALUE>
class OrderdDict : public QObject
{
public:
    OrderdDict(QObject *parent = nullptr) : QObject(parent),
        _dict(std::vector<std::pair<TKEY, TVALUE> >())
    {
    }
    /**
     * @brief   OrderdDict
     *          コピーコンストラクタ
     * @param   x
     */
    OrderdDict(const OrderdDict &x) :QObject(x.parent()),
        _dict(std::vector<std::pair<TKEY, TVALUE> >())
    {
        std::for_each(x._dict.begin(), x._dict.end(), [this](std::pair<TKEY, TVALUE> x) {
            _dict.push_back(x);
        });
    }
    TVALUE & operator[](const TKEY key) {
        for (uint i = 0; i < _dict.size(); ++i) {
            if (_dict[i].first == key) {
                return _dict[i].second;
            }
        }
        std::pair<TKEY, TVALUE> vv(key, nullptr);
        _dict.push_back(vv);
        return _dict.at(_dict.size() - 1).second;
    }
    std::pair<TKEY, TVALUE> & at(const int idx) {
        return _dict[static_cast<unsigned long long>(idx)];
    }
    int size() const {
        return static_cast<int>(_dict.size());
    }
    void reverse() {
        std::reverse(_dict.begin(), _dict.end());
    }

private:
    std::vector<std::pair<TKEY, TVALUE> > _dict;
};

class TwoLayerNet : public QObject
{
    Q_OBJECT
public:
    explicit TwoLayerNet(int input_size, int hidden_size, int output_size, double weight_init_std = 0.01, QObject *parent = nullptr);

    Eigen::MatrixXd predict(Eigen::MatrixXd x) {
        for (int i = 0; i < _layers.size(); ++i) {
            x = _layers.at(i).second->forword(x);
        }

        return x;
    }

    // x:入力データ, t:教師データ
    double loss(const Eigen::MatrixXd &x, const Eigen::MatrixXd &t) {
        Eigen::MatrixXd y = predict(x);
        return _lastLayer->forword(y, t);
    }

    double accuracy(Eigen::MatrixXd x, Eigen::MatrixXd t) {
        Eigen::MatrixXd y = predict(x);
        y = argmax(0, y).cast<double>();
        if (t.rows() > 1 && t.cols() > 1) {
            t = argmax(1, t).cast<double>();
        }

        double accuracy = (y.array() == t.array()).cast<int>().sum() / static_cast<double>(x.rows());
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

    std::map<std::string, Eigen::MatrixXd> gradient(const Eigen::MatrixXd &x, const Eigen::MatrixXd &t) {
        // forword
        loss(x, t);

        // backword
        double doutd = 1;
        Eigen::MatrixXd dout = _lastLayer->backword(doutd);

        OrderdDict<std::string, layerBase*> layers = _layers;
        layers.reverse();
        for (int i = 0; i < layers.size(); ++i) {
            dout = layers.at(i).second->backword(dout);
        }

        // 設定
        std::map<std::string, Eigen::MatrixXd> grads;
        Affine *a1 = qobject_cast<Affine*>(_layers["Affine1"]);
        grads["W1"] = a1->dW();
        grads["b1"] = a1->db();
        Affine *a2 = qobject_cast<Affine*>(_layers["Affine2"]);
        grads["W2"] = a2->dW();
        grads["b2"] = a2->db();

        return grads;
    }

signals:

public slots:

private:
    std::map<std::string, Eigen::MatrixXd> _params;
    OrderdDict<std::string, layerBase*> _layers;
    SoftmaxWithLoss *_lastLayer;

    Eigen::MatrixXi argmax(const int dim, const Eigen::MatrixXd x) {
        if (dim == 0) {
            Eigen::MatrixXi r(x.rows(), 1);
            for (int i = 0; i < x.rows(); ++i) {
                double xmax = x.minCoeff();
                int xidx = 0;
                for (int j = 0; j < x.cols(); ++j) {
                    if (x(i, j) > xmax) {
                        xidx = j;
                    }
                }
                r(i, 0) = xidx;
            }

            return r;
        } else {
            Eigen::MatrixXi r(1, x.cols());
            for (int i = 0; i < x.cols(); ++i) {
                double xmax = x.minCoeff();
                int xidx = 0;
                for (int j = 0; j < x.rows(); ++j) {
                    if (x(i, j) > xmax) {
                        xidx = j;
                    }
                }
                r(0, i) = xidx;
            }

            return r;
        }
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

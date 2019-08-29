#ifndef MULTILAYERNET_H
#define MULTILAYERNET_H

#include <vector>
#include <iostream>
#include <algorithm>
#include <random>

#include <QObject>
#include <QDebug>

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

/**
 * @brief The MultiLayerNet class
 */
class MultiLayerNet : public QObject
{
    /* 全結合による多層ニューラルネットワーク

    Parameters
    ----------
    input_size : 入力サイズ（MNISTの場合は784）
    hidden_size_list : 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
    output_size : 出力サイズ（MNISTの場合は10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
        'relu'または'he'を指定した場合は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
    weight_decay_lambda : Weight Decay（L2ノルム）の強さ
    */
    Q_OBJECT
public:
    explicit MultiLayerNet(int input_size, std::vector<int> hidden_size_list, int output_size, std::string activation = "relu", std::string weight_init_std = "relu", int weight_dcay_lambda = 0, QObject *parent = nullptr);

    Eigen::MatrixXd predict(Eigen::MatrixXd x) {
        for (int i = 0; i < _layers.size(); ++i) {
            std::pair<std::string, layerBase*> layer = _layers.at(i);
            Eigen::MatrixXd x2 = layer.second->forword(x);
            x = x2;
        }

        return x;
    }

    double loss(const Eigen::MatrixXd &x, const Eigen::MatrixXd &t) {
        /* 損失関数を求める

        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        損失関数の値
        */
        Eigen::MatrixXd y = predict(x);

        double weight_decay = 0;
        for (int idx = 1; idx < _hidden_layer_num + 2; ++idx) {
            Eigen::MatrixXd W = _params["W" + std::to_string(idx)];
            weight_decay += 0.5 * _weight_decay_lambda * W.array().pow(2).sum();
        }

        return _last_layer->forword(y, t) + weight_decay;
    }

    double accuracy(Eigen::MatrixXd x, Eigen::MatrixXd t) {
        Eigen::MatrixXd y = predict(x);
        y = argmax(1, y).cast<double>();
        if (t.rows() > 1 && t.cols() > 1) {
            t = argmax(1, t).cast<double>();
        }

        double accuracy = (y.array() == t.array()).cast<int>().sum() / x.rows();
        return accuracy;
    }

    // x:入力データ t:教師データ
    std::map<std::string, Eigen::MatrixXd> numerical_gradient(const Eigen::MatrixXd &x, const Eigen::MatrixXd &t) {
        /* 勾配を求める（数値微分）

        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        */
        std::function<double(Eigen::MatrixXd)> loss_W = [this, &x, &t](Eigen::MatrixXd) { return this->loss(x, t); };

        std::map<std::string, Eigen::MatrixXd> grads;
        for (int idx = 1; idx < _hidden_layer_num + 2; ++idx) {
            grads["W" + std::to_string(idx)] = numerical_gradient(loss_W, _params["W" + std::to_string(idx)]);
            grads["b" + std::to_string(idx)] = numerical_gradient(loss_W, _params["b" + std::to_string(idx)]);
        }

        return grads;
    }

    std::map<std::string, Eigen::MatrixXd> gradient(const Eigen::MatrixXd &x, const Eigen::MatrixXd &t) {
        /* 勾配を求める（誤差逆伝搬法）

        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        */
        // forword
        loss(x, t);

        // backword
        double doutd = 1;
        Eigen::MatrixXd dout = _last_layer->backword(doutd);

        OrderdDict<std::string, layerBase*> layers = _layers;
        layers.reverse();
        for (int i = 0; i < layers.size(); ++i) {
            Eigen::MatrixXd doutt = layers.at(i).second->backword(dout);
            dout = doutt;
        }

        // 設定
        std::map<std::string, Eigen::MatrixXd> grads;
        for (int idx = 1; idx < _hidden_layer_num + 2; ++idx) {
            grads["W" + std::to_string(idx)] = qobject_cast<Affine*>(_layers["Affine" + std::to_string(idx)])->dW() +
                    _weight_decay_lambda * qobject_cast<Affine*>(_layers["Affine" + std::to_string(idx)])->W();
            grads["b" + std::to_string(idx)] = qobject_cast<Affine*>(_layers["Affine" + std::to_string(idx)])->db();
        }

        return grads;
    }

    std::map<std::string, Eigen::MatrixXd> &params() {
        return _params;
    }

signals:

public slots:

private:
    int _input_size;
    int _output_size;
    std::vector<int> _hidden_size_list;
    int _hidden_layer_num;
    int _weight_decay_lambda;
    std::map<std::string, Eigen::MatrixXd> _params;
    OrderdDict<std::string, layerBase*> _layers;
    SoftmaxWithLoss *_last_layer;

    void __init_weight(std::string weight_init_std) {
        /* 重みの初期値設定

        Parameters
        ----------
        weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
            'relu'または'he'を指定した場合は「Heの初期値」を設定
            'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
        */
        std::vector<int> all_size_list;
        all_size_list.push_back(_input_size);
        std::copy(_hidden_size_list.begin(), _hidden_size_list.end(), std::back_inserter(all_size_list));
        all_size_list.push_back(_output_size);
        for (uint idx = 1; idx < static_cast<uint>(all_size_list.size()); ++idx) {
            double scale = 0;
            std::string lower_weight_init_std = weight_init_std;
            std::transform(lower_weight_init_std.begin(), lower_weight_init_std.end(), lower_weight_init_std.begin(),
                           tolower);
            if (lower_weight_init_std == "relu" || lower_weight_init_std == "he") {
                scale = std::sqrt(2.0 / all_size_list[idx - 1]);    // ReLUを使う場合に推奨される初期値
            } else if (lower_weight_init_std == "sigmoid" || lower_weight_init_std == "xavier") {
                scale = std::sqrt(1.0 / all_size_list[idx - 1]);    // sigmoidを使う場合に推奨される初期値
            } else {
                scale = std::stod(weight_init_std);
            }

			std::random_device seed_gen;
			std::default_random_engine engine(seed_gen());
			std::normal_distribution<> dist(0.0, 1.0);
			Eigen::MatrixXd Wn(all_size_list[idx - 1], all_size_list[idx]);
			for (int i = 0; i < Wn.size(); ++i) {
				Wn.data()[i] = dist(engine);
			}
            //_params["W" + std::to_string(idx)] = scale * Eigen::MatrixXd::Random(all_size_list[idx - 1],
            //        all_size_list[idx]);
			_params["W" + std::to_string(idx)] = scale * Wn;
            _params["b" + std::to_string(idx)] = Eigen::MatrixXd::Zero(all_size_list[idx], 1);
        }
    }

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

#endif // MULTILAYERNET_H

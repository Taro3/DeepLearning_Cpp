#include <QApplication>

#include <Eigen/Dense>

#include "mainwindow.h"
#include "optimizer.h"
#include "plotgraph.h"

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

Eigen::MatrixXd f(Eigen::MatrixXd x, Eigen::MatrixXd y) {
    return x.array().pow(2) / 20.0 + y.array().pow(2);
}

std::pair<double, double> df(double x, double y) {
    return std::pair<double, double>(x / 10.0, 2.0 * y);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> meshgrid(Eigen::MatrixXd x, Eigen::MatrixXd y) {
    Eigen::MatrixXd rx = Eigen::MatrixXd::Zero(y.rows(), x.rows());
    Eigen::MatrixXd ry = Eigen::MatrixXd::Zero(y.rows(), x.rows());
    x.transposeInPlace();
    for (int i = 0; i < rx.rows(); ++i) {
        rx.row(i) = x.row(0);
    }
    for (int i = 0; i < ry.cols(); ++i) {
        ry.col(i) = y.col(0);
    }
    return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>(rx, ry);
}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    QApplication a(argc, argv);
    MainWindow w;
    w.setVisible(false);

    std::pair<double, double> init_pos(-7.0, 2.0);
    std::map<std::string, double> params;
    params["x"] = init_pos.first;
    params["y"] = init_pos.second;
    std::map<std::string, double> grads;
    grads["x"] = 0;
    grads["y"] = 0;

    OrderdDict<std::string, OptimizerBase*> optimizers;
    optimizers["SGD"] = new SGD(0.95, &w);
    optimizers["Momentum"] = new Momentum(0.1, 0.9, &w);
    optimizers["AdaGrad"] = new AdaGrad(1.5, &w);
    optimizers["Adam"] = new Adam(0.3, 0.9, 0.999, &w);

//    int idx = 1;
    std::vector<PlotGraph*> plots;

    for (int k = 0; k < optimizers.size(); ++k) {
        OptimizerBase *optimizer = optimizers.at(k).second;
        std::vector<double> x_history;
        std::vector<double> y_history;
        params["x"] = init_pos.first;
        params["y"] = init_pos.second;

        for (int i = 0; i < 30; ++i) {
            x_history.push_back(params["x"]);
            y_history.push_back(params["y"]);

            std::pair<double, double> t = df(params["x"], params["y"]);
            grads["x"] = t.first;
            grads["y"] = t.second;
            std::map<std::string, Eigen::MatrixXd> tm;
            Eigen::MatrixXd tx(1, 1);
            tx << params["x"];
            Eigen::MatrixXd ty(1, 1);
            ty << params["y"];
            tm["x"] = tx;
            tm["y"] = ty;
            std::map<std::string, Eigen::MatrixXd> tg;
            Eigen::MatrixXd tx2(1, 1);
            tx2(0, 0) = grads["x"];
            Eigen::MatrixXd ty2(1, 1);
            ty2(0, 0) = grads["y"];
            tg["x"] = tx2;
            tg["y"] = ty2;
            optimizer->update(tm, tg);
            params["x"] = tm["x"](0, 0);
            params["y"] = tm["y"](0, 0);
        }

        Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(2000, -10, 10);
        Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(1000, -5, 5);

        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> m = meshgrid(x, y);
        Eigen::MatrixXd X = m.first;
        Eigen::MatrixXd Y = m.second;
        Eigen::MatrixXd Z = f(X, Y);

        // for sinple contour line
        Eigen::MatrixXi mask = (Z.array() > 7).cast<int>();
        for (int i = 0; i < Z.size(); ++i) {
            if (mask.data()[i]) {
                Z.data()[i] = 0;
            }
        }

        // plot
        auto p = new PlotGraph;
        Eigen::ArrayXd px(x_history.size());
        Eigen::ArrayXd py(y_history.size());
        for (uint i = 0; i < x_history.size(); ++i) {
            px(i) = x_history[i];
            py(i) = y_history[i];
        }
        p->plot(px, py);
        plots.push_back(p);
    }

    w.show();

    int r = a.exec();
    for (PlotGraph *p : plots) {
        delete p;
    }

    return r;
}

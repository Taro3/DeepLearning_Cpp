#ifndef SOFTMAXFUNCTION_H
#define SOFTMAXFUNCTION_H

#include <QObject>

#include <Eigen/Dense>

class SoftmaxFunction : public QObject
{
    Q_OBJECT
public:
    explicit SoftmaxFunction(QObject *parent = nullptr);
    Eigen::ArrayXd softmax1(Eigen::ArrayXd a) {
        Eigen::ArrayXd exp_a = a.exp();
        double sum_exp_a = exp_a.sum();
        Eigen::ArrayXd y = exp_a / sum_exp_a;

        return y;
    }
    Eigen::ArrayXd softmax2(Eigen::ArrayXd a) {
        double c = a.maxCoeff();
        Eigen::ArrayXd exp_a = (a - c).exp();
        double sum_exp_a = exp_a.sum();
        Eigen::ArrayXd y = exp_a / sum_exp_a;

        return y;
    }

signals:

public slots:
};

#endif // SOFTMAXFUNCTION_H

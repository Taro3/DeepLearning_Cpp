#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include <QObject>

#include <Eigen/Dense>

class ActivationFunction : public QObject
{
    Q_OBJECT
public:
    explicit ActivationFunction(QObject *parent = nullptr);
    /**
     * @brief       step_function
     *              入力値xのステップ関数
     * @param[in]   x   入力値
     * @return      出力値
     */
    Eigen::VectorXd step_function(Eigen::ArrayXd x) {
        auto y = x > 0;
        return y.cast<double>();
    }
    /**
     * @brief       sigmoid
     *              入力値xのシグモイド関数
     * @param[in]   x   入力値
     * @return      出力値
     */
    Eigen::VectorXd sigmoid(Eigen::ArrayXd x) {
        return 1 / (1 + Eigen::exp(-x));
    }
    /**
     * @brief       relu
     *              入力値xのReLU関数
     * @param[in]   x   入力値
     * @return      出力値
     */
    Eigen::ArrayXd relu(Eigen::ArrayXd x) {
        return x.max(0);
    }

signals:

public slots:
};

#endif // ACTIVATIONFUNCTION_H

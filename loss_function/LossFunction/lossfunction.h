#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#include <QObject>

#include <Eigen/Dense>

class LossFunction : public QObject
{
    Q_OBJECT
public:
    explicit LossFunction(QObject *parent = nullptr);
    /**
     * @brief       mean_squared_error
     *              2情話誤差を計算する
     * @param[in]   y   ニューラルネットワークの出力
     * @param[in]   t   教師データ
     * @return      2乗和誤差値
     */
    double mean_squared_error(Eigen::VectorXd y, Eigen::VectorXd t) {
        return 0.5 * (y - t).squaredNorm();
    }

signals:

public slots:
};

#endif // LOSSFUNCTION_H

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
    /**
     * @brief       cross_entropy_error
     *              交差エントロピー誤差を計算する
     * @param[in]   y   ニューラルネットワークの出力
     * @param[in]   t   教師データ
     * @return      交差エントロピー誤差値
     */
    double cross_entropy_error(Eigen::VectorXd y, Eigen::VectorXd t) {
//        double delta = 1e-007;
//        return -(t.array() * (y.array() + delta).log()).sum();
        std::vector<Eigen::VectorXd> y2 = { y };
        std::vector<Eigen::VectorXd> t2 = { t };
        return cross_entropy_error(y2, t2);
    }
    /**
     * @brief       cross_entropy_error
     *              バッチ対応版の交差エントロピー誤差の計算
     * @param[in]   y   ニューラルネットワークの出力(配列)
     * @param[in]   t   教師データ(配列)
     * @return      交差エントロピー誤差値
     */
    double cross_entropy_error(std::vector<Eigen::VectorXd> y, std::vector<Eigen::VectorXd> t) {
        unsigned long long batch_size = y.size();
        double r = 0;
        for (unsigned long long i = 0; i < y.size(); ++i) {
            Eigen::VectorXd yy = y[i];
            Eigen::VectorXd tt = t[i];
            r += (tt.array() * (yy.array() + 1e-007).log()).sum();
        }
        return -r / batch_size;
    }
    /**
     * @brief       cross_entropy_error_label
     *              教師データtがラベル値の配列の場合の交差エントロピー誤差の計算
     * @param[in]   y   ニューラルネットワークの出力(配列)
     * @param[in]   t   教師データ(配列)
     * @return      交差エントロピー誤差値
     */
    double cross_entropy_error_label(std::vector<Eigen::VectorXd> y, Eigen::VectorXd t) {
        unsigned long long batch_size = y.size();
        double r = 0;
        for (int i = 0; i < static_cast<int>(y.size()); ++i) {
            Eigen::VectorXd yy = y[static_cast<unsigned long long>(i)];
            r += log(yy(t.cast<int>()(i)) + 1e+007);
        }
        return -r / batch_size;
    }


signals:

public slots:
};

#endif // LOSSFUNCTION_H

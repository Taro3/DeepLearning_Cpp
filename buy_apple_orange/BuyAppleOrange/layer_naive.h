#ifndef LAYER_NAIVE_H
#define LAYER_NAIVE_H

#include <QObject>

class MulLayer : public QObject
{
    Q_OBJECT
public:
    explicit MulLayer(QObject *parent = nullptr);
    double forword(const double x, const double y) {
        _x = x;
        _y = y;
        double out = x * y;

        return out;
    }
    std::tuple<double, double> backword(const double dout) const {
        double dx = dout * _y;
        double dy = dout * _x;

        return std::tuple<double, double>(dx, dy);
    }

signals:

public slots:

private:
    double  _x;
    double  _y;
};

class AddLayer : public QObject
{
    Q_OBJECT
public:
    explicit AddLayer(QObject *parent = nullptr);
    double forword(const double x, const double y) {
        double out = x + y;

        return out;
    }
    std::tuple<double, double> backword(const double dout) const {
        double dx = dout * 1;
        double dy = dout * 1;

        return std::tuple<double, double>(dx, dy);
    }

signals:

public slots:

private:
    double  _x;
    double  _y;
};

#endif // LAYER_NAIVE_H

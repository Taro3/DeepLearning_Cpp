#ifndef PLOTGRAPH_H
#define PLOTGRAPH_H

#include <QtCharts>

#include <Eigen/Dense>

class PlotGraph
{
public:
    PlotGraph();
    ~PlotGraph();
    void plot(Eigen::ArrayXd x, Eigen::ArrayXd y, QWidget *parent = nullptr);

private:
    QChartView *_chartView;
};

#endif // PLOTGRAPH_H

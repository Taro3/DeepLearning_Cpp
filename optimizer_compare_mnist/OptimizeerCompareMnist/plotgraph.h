#ifndef PLOTGRAPH_H
#define PLOTGRAPH_H

#include <QtCharts>

#include <Eigen/Dense>

class PlotGraph
{
public:
    PlotGraph(std::string title = "");
    ~PlotGraph();
    void plot(Eigen::ArrayXd x, Eigen::ArrayXd y);

private:
    QChartView *_chartView;
    std::string _title;
};

#endif // PLOTGRAPH_H

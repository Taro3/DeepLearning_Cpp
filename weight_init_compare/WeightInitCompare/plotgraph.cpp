#include <QtCharts>

#include <Eigen/Dense>

#include "plotgraph.h"

PlotGraph::PlotGraph(std::string title) : _chartView(new QChartView()), _title(title)
{
    _chartView->setWindowTitle(_title.c_str());
}

PlotGraph::~PlotGraph()
{
    _chartView->deleteLater();
}

/**
 * @brief       PlotGraph::plot
 *              引数で与えられた座標軍のグラフを描画する
 * @param[in]   x   X座標軍
 * @param[in]   y   Y座標軍
 * @return      なし
 */
void PlotGraph::plot(Eigen::ArrayXd x, Eigen::ArrayXd y)
{
    QChart *c = _chartView->chart();
    c->removeAllSeries();
    c->legend()->hide();
    //auto ls = new QtCharts::QLineSeries(_chartView);
    auto ss = new QSplineSeries(_chartView);
    for (auto i = 0; i < x.rows(); ++i)
    {
        //ls->append(static_cast<qreal>(x(i)), static_cast<qreal>(y(i)));
        ss->append(static_cast<qreal>(x(i)), static_cast<qreal>(y(i)));
    }
    //c->addSeries(ls);
    c->addSeries(ss);
    c->createDefaultAxes();
    _chartView->setRenderHint(QPainter::Antialiasing);
    _chartView->resize(640, 480);
    _chartView->show();
}

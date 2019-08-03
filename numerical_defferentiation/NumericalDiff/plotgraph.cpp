#include <QtCharts>

#include <Eigen/Dense>

#include "plotgraph.h"

PlotGraph::PlotGraph() : _chartView(nullptr)
{

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
void PlotGraph::plot(Eigen::ArrayXd x, Eigen::ArrayXd y, QWidget *parent)
{
    if (_chartView) {
        delete _chartView;
    }
    _chartView = new QChartView(parent);
    QChart *c = _chartView->chart();
    c->removeAllSeries();
    c->legend()->hide();
    auto ls = new QtCharts::QLineSeries(_chartView);
    for (auto i = 0; i < x.rows(); ++i)
    {
        ls->append(static_cast<qreal>(x(i)), static_cast<qreal>(y(i)));
    }
    c->addSeries(ls);
    c->createDefaultAxes();
    _chartView->setRenderHint(QPainter::Antialiasing);
    _chartView->resize(640, 480);
    _chartView->show();
}

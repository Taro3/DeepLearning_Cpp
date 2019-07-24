#include <QtCharts>
#include <Eigen/Dense>

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "activationfunction.h"
#include "plotgraph.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    _plotGraph(new PlotGraph())
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
    delete _plotGraph;
}

/**
 * @brief   MainWindow::on_pushButton_clicked
 *          ステップ関数グラフ描画
 */
void MainWindow::on_pushButton_clicked()
{
    Eigen::ArrayXd x;
    x.setLinSpaced(100, -5, 5);
    ActivationFunction af;
    auto y = af.step_function(x);
    _plotGraph->plot(x, y.cast<double>());
}

/**
 * @brief   MainWindow::on_pushButton_2_clicked
 *          シグモイド関数グラフ描画
 */
void MainWindow::on_pushButton_2_clicked()
{
    Eigen::ArrayXd x;
    x.setLinSpaced(100, -5, 5);
    ActivationFunction af;
    auto y = af.sigmoid(x);
    _plotGraph->plot(x, y);
}

/**
 * @brief   MainWindow::on_pushButton_3_clicked
 *          ReLU関数グラフ描画
 */
void MainWindow::on_pushButton_3_clicked()
{
    Eigen::ArrayXd x;
    x.setLinSpaced(100, -5, 5);
    ActivationFunction af;
    auto y = af.relu(x);
    _plotGraph->plot(x, y);
}

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <memory>

#include <QMainWindow>
#include <QImage>
#include <QPainter>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void setImage(std::shared_ptr<QImage> image) {
        _image = image;
    }

protected:
    void paintEvent(QPaintEvent *e) {
        QMainWindow::paintEvent(e);
        if (_image) {
            QPainter p(this);
            p.drawImage(0, 0, *_image);
        }
    }

private:
    Ui::MainWindow *ui;
    std::shared_ptr<QImage> _image;
};

#endif // MAINWINDOW_H

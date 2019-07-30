#ifndef MNIST_H
#define MNIST_H

#include <iostream>
#include <tuple>

#include <QObject>
#include <QFile>
#include <QtEndian>

#include <Eigen/Dense>

class Mnist : public QObject
{
    Q_OBJECT
public:
    explicit Mnist(QObject *parent = nullptr);
    /**
     * @brief       load_mnist
     * @param[in]   normalize       正規化フラグ
     * @value       true            正規化(デフォルト)
     * @value       false           非正規化
     * @param[in]   flatten         1次元配列化フラグ
     * @value       true            1次元化(デフォルト)
     * @value       false           非1次元化
     * @param[in]   one_hot_label   hot-label化フラグ
     * @value       true            hot-labek化
     * @value       false           非hot-label化(デフォルト)
     * @return      生成されたMNISTデータ郡
     */
    std::tuple<std::vector<Eigen::MatrixXd>, Eigen::MatrixXi, std::vector<Eigen::MatrixXd>, Eigen::MatrixXi>
    load_mnist(bool normalize = true, bool flatten = true, bool one_hot_label = false) {
        static const std::string TRAIN_IMAGE_FILENAME = "../../../MNIST/train-images.idx3-ubyte";
        static const std::string TRAIN_LABEL_FILENAME = "../../../MNIST/train-labels.idx1-ubyte";
        static const std::string TEST_IMAGE_FILENAME = "../../../MNIST/t10k-images.idx3-ubyte";
        static const std::string TEST_LABEL_FILENAME = "../../../MNIST/t10k-labels.idx1-ubyte";

        // データ読み込み
        std::vector<Eigen::MatrixXd> trainImages = loadImages(TRAIN_IMAGE_FILENAME);
        Eigen::MatrixXi trainLabels = loadLabels(TRAIN_LABEL_FILENAME);
        std::vector<Eigen::MatrixXd> testImages = loadImages(TEST_IMAGE_FILENAME);
        Eigen::MatrixXi testLabels = loadLabels(TEST_LABEL_FILENAME);

        // 正規化処理
        if (normalize) {
            for (std::vector<Eigen::MatrixXd>::iterator it = trainImages.begin(); it != trainImages.end(); ++it) {
                for (int r = 0; r < (*it).rows(); ++r) {
                    for (int c = 0; c < (*it).cols(); ++c) {
                        (*it)(r, c) /= 255;
                    }
                }
            }
            for (std::vector<Eigen::MatrixXd>::iterator it = testImages.begin(); it != testImages.end(); ++it) {
                for (int r = 0; r < (*it).rows(); ++r) {
                    for (int c = 0; c < (*it).cols(); ++c) {
                        (*it)(r, c) /= 255;
                    }
                }
            }
        }

        // 1次元配列化処理
        if (flatten) {
            for (std::vector<Eigen::MatrixXd>::iterator it = trainImages.begin(); it != trainImages.end(); ++it) {
                (*it) = Eigen::Map<Eigen::VectorXd>((*it).data(), (*it).size());
            }
            for (std::vector<Eigen::MatrixXd>::iterator it = testImages.begin(); it != testImages.end(); ++it) {
                (*it) = Eigen::Map<Eigen::VectorXd>((*it).data(), (*it).size());
            }
        }

        // one-hotデータ変換処理
        if (one_hot_label) {
            Eigen::MatrixXi trainLabelOH(trainLabels.rows(), 10);
            trainLabelOH.setZero();
            for (int i = 0; i < trainLabels.size(); ++i) {
                trainLabelOH(i, trainLabels(i)) = 1;
            }
            trainLabels = trainLabelOH;

            Eigen::MatrixXi testLabelOH(testLabels.size(), 10);
            testLabelOH.setZero();
            for (int i = 0; i < testLabels.size(); ++i) {
                testLabelOH(i, testLabels(i)) = 1;
            }
            testLabels = testLabelOH;
        }

        return std::tuple<std::vector<Eigen::MatrixXd>, Eigen::MatrixXi, std::vector<Eigen::MatrixXd>,
                Eigen::MatrixXi>(trainImages, trainLabels, testImages, testLabels);
    }

signals:

public slots:

private:
    /**
     * @brief       loadImages
     *              画像データ読み込み処理
     * @param[in]   filename
     * @return      画像データリスト
     */
    std::vector<Eigen::MatrixXd> loadImages(std::string filename) {
        static const int MAGIC = 0x00000803;

        QFile file(filename.c_str());

        if (!file.open(QIODevice::ReadOnly)) {
            std::cerr << "cant open " << filename << std::endl;
            file.close();
            return std::vector<Eigen::MatrixXd>();
        }

        int magic = 0;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        magic = qFromBigEndian(magic);
        if (magic != MAGIC) {
            std::cerr << "invalid magic number." << std::endl;
            file.close();
            return std::vector<Eigen::MatrixXd>();
        }

        int image_count = 0;
        file.read(reinterpret_cast<char*>(&image_count), sizeof(image_count));
        image_count = qFromBigEndian(image_count);

        int image_width = 0;
        file.read(reinterpret_cast<char*>(&image_width), sizeof(image_width));
        image_width = qFromBigEndian(image_width);

        int image_height = 0;
        file.read(reinterpret_cast<char*>(&image_height), sizeof(image_height));
        image_height = qFromBigEndian(image_height);

        std::vector<Eigen::MatrixXd> imgs;
        for (int i = 0; i < image_count; ++i) {
            Eigen::MatrixXd img(image_height, image_width);
            for (int r = 0; r < image_height; ++r) {
                for (int c = 0; c < image_width; ++c) {
                    unsigned char p;
                    file.read(reinterpret_cast<char*>(&p), sizeof(p));
                    img(r, c) = p;
                }
            }
            imgs.push_back(img);
        }
        file.close();

        return imgs;
    }

    /**
     * @brief       loadLabels
     *              ラベルデータ読み込み
     * @param[in]   filename
     * @return      ラベルデータリスト
     */
    Eigen::MatrixXi loadLabels(std::string filename) {
        static const int MAGIC = 0x00000801;

        QFile file(filename.c_str());

        if (!file.open(QIODevice::ReadOnly)) {
            std::cerr << "cant open " << filename << std::endl;
            file.close();
            return Eigen::MatrixXi();
        }

        int magic = 0;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        magic = qFromBigEndian(magic);
        if (magic != MAGIC) {
            std::cerr << "invalid magic number." << std::endl;
            file.close();
            return Eigen::MatrixXi();
        }

        int label_count = 0;
        file.read(reinterpret_cast<char*>(&label_count), sizeof(label_count));
        label_count = qFromBigEndian(label_count);

        Eigen::MatrixXi labels(label_count, 1);
        for (int i = 0; i < label_count; ++i) {
            unsigned char p;
            file.read(reinterpret_cast<char*>(&p), sizeof(p));
            labels(i, 0) = p;
        }
        file.close();

        return labels;
    }
};

#endif // MNIST_H

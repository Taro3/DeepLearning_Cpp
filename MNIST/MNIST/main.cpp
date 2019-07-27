#include <QCoreApplication>

#include "mnist.h"

int main(int argc, char *argv[])
{
    Q_UNUSED(argc)
    Q_UNUSED(argv)

    Mnist m;
    auto r = m.load_mnist(false, false, true);
    auto trainImgs = std::get<0>(r);
    auto trainLbls = std::get<1>(r);

    std::cout << trainImgs[0] << std::endl;
    std::cout << "----------" << std::endl;
    std::cout << trainImgs[0].rows() << " " << trainImgs[0].cols() << std::endl;
    std::cout << "----------" << std::endl;
    std::cout << trainLbls << std::endl;

    return 0;
}

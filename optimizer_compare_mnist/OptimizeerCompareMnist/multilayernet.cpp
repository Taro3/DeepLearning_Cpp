#include "multilayernet.h"
#include "layers.h"

MultiLayerNet::MultiLayerNet(int input_size, std::vector<int> hidden_size_list, int output_size, std::string activation,
                             std::string weight_init_std, int weight_dcay_lambda, QObject *parent)
    : QObject(parent),
      _input_size(input_size), _output_size(output_size), _hidden_size_list(hidden_size_list),
      _hidden_layer_num(static_cast<int>(_hidden_size_list.size())), _weight_decay_lambda(weight_dcay_lambda),
      _params(std::map<std::string, Eigen::MatrixXd>()), _layers(OrderdDict<std::string, layerBase*>()),
      _last_layer(new SoftmaxWithLoss(this))
{
    // 重みの初期化
    __init_weight(weight_init_std);

    // レイヤの生成
    std::map<std::string, layerBase*> activation_layer = { { "sigmoid", new Sigmoid(this) }, { "relu", new Relu(this) },
                                                         };
    for (int idx = 1; idx < _hidden_layer_num + 1; ++idx) {
        std::string keyAffine = "Affine" + std::to_string(idx);
        _layers[keyAffine] = new Affine(_params["W" + std::to_string(idx)], _params["b" + std::to_string(idx)], this);
        std::string keyActivationFunction = "Activation_function" + std::to_string(idx);
        _layers[keyActivationFunction] = activation_layer[activation];
    }

    int idx = _hidden_layer_num + 1;
    _layers["Affine" + std::to_string(idx)] = new Affine(_params["W" + std::to_string(idx)],
            _params["b" + std::to_string(idx)], this);
}

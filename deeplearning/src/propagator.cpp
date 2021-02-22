#include "propagator.h"

void SimplePropagator::forward(
    const std::vector<const Layer*>& layers, const std::vector<Matrix>& weights, const Matrix& disWeight,
    const Matrix& data) {
    const FirstLayer *firstLayer = dynamic_cast<const FirstLayer*>(layers[0]);
    _xList.resize(layers.size());
    _zList.resize(layers.size());
//    std::cout << "top 5 rows of data = " << std::endl << data.block(0, 0, 5, data.cols()) << std::endl;
        
    for (int j = 0; j < layers.size(); j++) {
        if (j == 0) {
            _xList[j] = layers[j]->eval(data);
            _x0 = firstLayer->evalDiscrete(data);
        } else {
            _zList[j] = weights[j - 1] * _xList[j - 1];
            if (j == 1) {
                _zList[j] += disWeight * _x0;
            }
            _xList[j] = layers[j]->eval(_zList[j]);
        }
    }
}

void SimplePropagator::backward(const std::vector<const Layer*>& layers,
    const std::vector<Matrix>& weights, const Matrix& disWeight, const Matrix& y) {
    _weightChanges.resize(layers.size() - 1);
    
    const OutputLayer* layer = dynamic_cast<const OutputLayer*>(layers.back());
    Matrix delta = layer->delta(_xList.back(), y);
            
    for (int j = layers.size() - 2; j >= 0; j--) {
        if (_xList[j].cols() != delta.rows()) {
            throw DPLException("SimplePropagator::backward: size of matrix delta is incorrect.");
        }
//        _weightChanges[j] = (_xList[j] * delta).transpose();
        _weightChanges[j] = delta.transpose() * _xList[j].transpose();
        if (j == 0) {
            // compute change of discrete weight.
            _disWeightChange = (_x0 * delta).transpose();
        } else {
            delta = (delta * weights[j]).array() * layers[j]->gDiff(_zList[j]);
        }
    }
}

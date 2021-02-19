#include "propagator.h"

void SimplePropagator::forward(
    const std::vector<const Layer*>& layers, const std::vector<Matrix>& weights,
    const Matrix& data) {
    
    _xList.resize(layers.size());
    _zList.resize(layers.size());
        
    for (int j = 0; j < layers.size(); j++) {
        if (j == 0) {
            _xList[j] = layers[j]->eval(data);
        } else {
            _zList[j] = weights[j - 1] * _xList[j - 1];
            _xList[j] = layers[j]->eval(_zList[j]);
        }
    }
}

void SimplePropagator::backward(const std::vector<const Layer*>& layers,
    const std::vector<Matrix>& weights, const Matrix& y) {
    _weightChanges.resize(layers.size() - 1);
    
    const OutputLayer* layer = dynamic_cast<const OutputLayer*>(layers.back());
    Matrix delta = layer->delta(_xList.back(), y);
            
    for (int j = layers.size() - 2; j >= 0; j--) {
        if (_xList[j].cols() != delta.rows()) {
            throw DPLException("SimplePropagator::backward: size of matrix delta is incorrect.");
        }
//        _weightChanges[j] = (_xList[j] * delta).transpose();
        _weightChanges[j] = delta.transpose() * _xList[j].transpose();
        if (j > 0) {
            delta = (delta * weights[j]).array() * layers[j]->gDiff(_zList[j]);
        }
    }
}

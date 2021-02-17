#include "model.h"

Model::Model() {
}

Model::~Model() {
    for (size_t i = 0; i < _layers.size(); i++) {
        delete _layers[i];
    }
    _layers.clear();
}

void Model::addLayer(const Layer* layer) {
    _layers.push_back(layer);
}

void Model::prepare() {
    std::cout << "preparing model..." << std::endl;
    srand(time(NULL));
    for (int i = 0; i < _layers.size() - 1; i++) {
        _weights.push_back(Matrix::Random(_layers[i + 1]->numberOfInNodes(), _layers[i]->numberOfOutNodes()));
    }
}

Vector Model::train(const Matrix& data, const Vector& y, size_t epic, data_t learningRate) {
    std::cout << "begin training" << std::endl;
    std::vector<Matrix> x(_layers.size());
    std::vector<Matrix> z(_layers.size());
    Vector loss(epic);
    
    for (int i = 0; i < epic; i++) {
        // do forward propagator
        for (int j = 0; j < _layers.size(); j++) {
            if (j == 0) {
                x[j] = _layers[j]->eval(data);
            } else {
                z[j] = _weights[j - 1] * x[j - 1];
                x[j] = _layers[j]->eval(z[j]);
                if (j == _layers.size() - 1) {
                    const OutputLayer* layer = dynamic_cast<const OutputLayer*>(_layers[j]);
                    loss(i) = layer->loss(x[j], y);
                }
            }            
        }
                
        // backward propagator
        // for output node.
        Matrix delta;
        for (int j = _layers.size() - 1; j >= 0; j--) {
            if (j == _layers.size() - 1) {
                const OutputLayer* layer = dynamic_cast<const OutputLayer*>(_layers[j]);
                delta = layer->delta(x[j], y);
            } else if (j == 0) {
                _weights[j] -= learningRate * (x[j] * delta);
            } else {
                Matrix prev = delta;
                delta = (prev * _weights[j]).array() * _layers[j]->gDiff(z[j]);
                _weights[j] -= learningRate * (x[j] * prev);
            }            
        }
    }
    
    return loss;
}

RVector Model::eval(const RVector& input) {
    Matrix res;
    Matrix z;
    
    for (int i = 0; i < _layers.size(); i++) {
        if (i == 0) {
            res = _layers[i]->eval(input);
        } else {
            z = _weights[i - 1] * res;
            res = _layers[i]->eval(z);
        }
    }
    
    return res;
}

#include "model.h"

std::ostream& operator<< (std::ostream& out, const HyperParameter& params) {
  out << "{ epochs=" << params.epochs << ", batch=" << params.batch;
  out << ", learningRate=" << params.learningRate << ", lambda=" << params.lambda << " }";  
  return out;
}


Model::Model() {
}

Model::~Model() {
    if (_regular != NULL) {
        delete _regular;
    }
    
    if (_propagator != NULL) {
        delete _propagator;
    }
    
    for (size_t i = 0; i < _layers.size(); i++) {
        delete _layers[i];
    }
    _layers.clear();
}

void Model::addLayer(const Layer* layer) {
    _layers.push_back(layer);
}

void Model::prepare(std::string regularization) {
    std::cout << "preparing model..." << std::endl;
    srand(time(NULL));
    for (int i = 0; i < _layers.size() - 1; i++) {
        _weights.push_back(Matrix::Random(_layers[i + 1]->numberOfInNodes(), _layers[i]->numberOfOutNodes()));
    }
    _regular = Regularization::create(regularization);
    _propagator = new SimplePropagator();
}

Vector Model::train(const Matrix& data, const Matrix& y, HyperParameter params) {
    std::cout << "begin training" << std::endl;
    std::cout << "hyperparameters=" << params << std::endl;
    Vector loss = Vector::Zero(params.epochs);
    
    for (int i = 0; i < params.epochs; i++) {
        if (std::abs(params.lambda) > EPS) {
            for (int j = 0; j < _layers.size() - 1; j++) {
                    loss(i) += params.lambda * _regular->complixity(_weights[j]);
            }
        }
                
        // do forward and backward propagators
        if (params.batch <= 0 || params.batch >= data.rows()) {
            _propagator->forward(_layers, _weights, data);
            _propagator->backward(_layers, _weights, y);
            loss(i) += getOutputLayer()->loss(_propagator->getResult(), y);
        } else {
            std::vector<int> v = pickRandomIndex(data.rows(), params.batch);
            Eigen::Map<Eigen::ArrayXi> av(&v[0], v.size());
            Matrix subdata = indexing(data, av);
            Matrix suby = indexing(y, av);
            
            _propagator->forward(_layers, _weights, subdata);
            _propagator->backward(_layers, _weights, suby);
            loss(i) += getOutputLayer()->loss(_propagator->getResult(), suby);
        }
        
        for (int j = 0; j < _layers.size() - 1; j++) {
            if (_weights[j].rows() != _propagator->getWeightChange(j).rows() ||
            _weights[j].cols() != _propagator->getWeightChange(j).cols()) {
                throw DPLException("Model::train: getWeightChange matrix size not match.");
            }
            _weights[j] -= params.learningRate * (_propagator->getWeightChange(j)
                                    + params.lambda * _regular->diff(_weights[j]));
        }
    }
    
    return loss;
}

Matrix Model::eval(const Matrix& input) {
    _propagator->forward(_layers, _weights, input);
    return _propagator->getResult();    
}

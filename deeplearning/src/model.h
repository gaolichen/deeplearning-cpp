#pragma once
#include "common.h"
#include "layer.h"
#include "propagator.h"
#include "regularization.h"

struct HyperParameter {
    int epochs = 300;
    int batch = -1;
    data_t learningRate = 0.01;
    data_t lambda = 0.0;
};

std::ostream& operator<< (std::ostream& out, const HyperParameter& params);

class Model {
private:
    std::vector<Matrix> _weights;
    std::vector<const Layer*> _layers;
    Regularization* _regular = NULL;
    Propagator* _propagator = NULL;
public:
    Model();
    ~Model();
    
    const OutputLayer* getOutputLayer() {
        return dynamic_cast<const OutputLayer*>(_layers.back());
    }
    
    const std::vector<Matrix>& weights() const {
        return _weights;
    }
    
    void addLayer(const Layer* layer);
    
    void prepare(std::string regularization = "L2");
    
    Vector train(const Matrix& data, const Matrix& y, HyperParameter params);
    
//    RVector eval(const RVector& input);
    Matrix eval(const Matrix& input);
};

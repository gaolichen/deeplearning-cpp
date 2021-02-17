#pragma once
#include "common.h"
#include "layer.h"

class Model {
private:
    std::vector<Matrix> _weights;
    std::vector<const Layer*> _layers;
//    OutputNode *_output;
//    LossFunction* _loss;
//    data_t _learningRate;
public:
    Model();
    ~Model();
    
    void addLayer(const Layer* layer);
    
    void prepare();
    
    Vector train(const Matrix& data, const Vector& y, size_t epic, data_t learningRate);
    
    RVector eval(const RVector& input);
};

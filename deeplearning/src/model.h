#pragma once
#include "common.h"
#include "layer.h"
#include "loss.h"

class Model {
private:
    std::vector<Matrix> _weights;
    std::vector<const Layer*> _layers;
    OutputNode *_output;
    LossFunction* _loss;
public:
    Model(std::string loss = "L2");
    ~Model();
    
    void addLayer(const Layer* layer);
    
    void prepare();
    
    Vector train(const Matrix& data, const Vector& res, size_t batchSize, size_t epic);
    
    data_t eval(const Vector& input);
};

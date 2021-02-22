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
    data_t validation_split = 0.0;
};

std::ostream& operator<< (std::ostream& out, const HyperParameter& params);

class Model {
private:
    std::vector<Matrix> _weights;
    Matrix _discreteWeight;
    std::vector<const Layer*> _layers;
    Regularization* _regular = NULL;
    Propagator* _propagator = NULL;
    Vector _trainingLoss;
    Vector _validationLoss;
public:
    Model();
    ~Model();
    
    const OutputLayer* getOutputLayer() {
        return dynamic_cast<const OutputLayer*>(_layers.back());
    }
    
    const std::vector<Matrix>& weights() const {
        return _weights;
    }
    
    const Vector& trainingLoss() {
        return _trainingLoss;
    }
    
    const Vector& validationLoss() {
        return _validationLoss;
    }
    
    void plotLoss(bool rms = false) const;
   
    void addLayer(const Layer* layer);
    
    void prepare(std::string regularization = "L2");
    
    void train(const Matrix& data, const Matrix& y, HyperParameter params);
    
    Matrix predict(const Matrix& input);
    
    data_t evaluate(const Matrix& testX, const Matrix& testY);
};

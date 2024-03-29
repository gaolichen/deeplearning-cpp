#pragma once
#include "common.h"
#include "layer.h"
#include "propagator.h"
#include "regularization.h"

struct HyperParameter {
    int epochs = 300;
    int batchSize = 32;
    data_t learningRate = 0.01;
    data_t lambda = 0.0;
    data_t validation_split = 0.0;
};

std::ostream& operator<< (std::ostream& out, const HyperParameter& params);

class Model {
private:
    std::vector<Matrix> _weights;
    Matrix _discreteWeight;
    std::vector<Layer*> _layers;
    Regularization* _regular = NULL;
    Propagator* _propagator = NULL;
    Vector _trainingLoss;
    Vector _validationLoss;
    
    Vector _trainingAccuracy;
    Vector _validationAccuracy;
    
    void setTraining(bool isTraining);
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
    
    const Vector& validationAccuracy() {
        return _validationAccuracy;
    }
    
    const Vector& trainingAccuracy() {
        return _trainingAccuracy;
    }
    
    void plotLoss(bool rms = false) const;
   
    void addLayer(Layer* layer);
    
    void prepare(std::string regularization = "L2");
    
    void train(const Matrix& data, const Matrix& y, HyperParameter params);
    
    Matrix predict(const Matrix& input);
    
    data_t evaluate(const Matrix& testX, const Matrix& testY);
};

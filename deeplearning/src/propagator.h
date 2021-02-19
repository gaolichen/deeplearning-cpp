#pragma once
#include "common.h"
#include "layer.h"

class Propagator {
protected:
    std::vector<Matrix> _xList;
    
    std::vector<Matrix> _zList;
    
    std::vector<Matrix> _weightChanges;
public:
    const Matrix& getX(int i) {
        return _xList[i];
    }
    
    const Matrix& getResult() {
        return _xList.back();
    }
    
    const Matrix& getZ(int i) {
        return _zList[i];
    }
    
    const Matrix& getWeightChange(int i) {
        return _weightChanges[i];
    }
    
    virtual void forward(const std::vector<const Layer*>& layers, const std::vector<Matrix>& weights, const Matrix& data) = 0;
    
    virtual void backward(const std::vector<const Layer*>& layers, const std::vector<Matrix>& weights, const Matrix& y) = 0;
};

class SimplePropagator : public Propagator {
public:
    virtual void forward(
            const std::vector<const Layer*>& layers,
            const std::vector<Matrix>& weights,
            const Matrix& data);
    
    virtual void backward(const std::vector<const Layer*>& layers,
            const std::vector<Matrix>& weights, const Matrix& y);
};

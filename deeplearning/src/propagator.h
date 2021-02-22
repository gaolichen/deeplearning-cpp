#pragma once
#include "common.h"
#include "layer.h"
#include "onehot.h"

class Propagator {
protected:
    std::vector<Matrix> _xList;
    Onehot _x0;

    std::vector<Matrix> _zList;
    
    std::vector<Matrix> _weightChanges;
    Matrix _disWeightChange;
public:
    const Matrix& getX(int i) const {
        return _xList[i];
    }
    
    const Matrix& getResult() const {
        return _xList.back();
    }
    
    const Matrix& getZ(int i) const {
        return _zList[i];
    }
    
    const Matrix& getWeightChange(int i) const {
        return _weightChanges[i];
    }
    
    const Matrix& discreteWeightChange() const {
        return _disWeightChange;
    }
        
    virtual void forward(const std::vector<const Layer*>& layers, const std::vector<Matrix>& weights, const Matrix& disWeight, const Matrix& data) = 0;
    
    virtual void backward(const std::vector<const Layer*>& layers, const std::vector<Matrix>& weights, const Matrix& disWeight, const Matrix& y) = 0;
};

class SimplePropagator : public Propagator {
private:
    static Matrix discreteDot(const Matrix& disWeight, const MatrixI& x);
    static Matrix discreteDot(int maxRow, const MatrixI& x, const Matrix& delta);
public:
    virtual void forward(
            const std::vector<const Layer*>& layers,
            const std::vector<Matrix>& weights,
            const Matrix& disWeight,
            const Matrix& data);
    
    virtual void backward(const std::vector<const Layer*>& layers,
            const std::vector<Matrix>& weights, const Matrix& disWeight, const Matrix& y);
};

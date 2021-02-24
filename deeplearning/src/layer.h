#pragma once
#include "common.h"
#include "node.h"
#include "datautil.h"
#include "onehot.h"
#include "featurecolumn.h"

class Layer {
private:
    bool _training;
public:            
    virtual void setTraining(bool training);
    
    bool isTraining() const {
        return this->_training;
    }
    
    virtual int indexOfUnitNode() const = 0;

    virtual size_t numberOfOutNodes() const = 0;
        
    virtual size_t numberOfInNodes() const = 0;
    
    virtual Matrix eval(const Matrix& input) const = 0;
    
    virtual Array gDiff(const Matrix& z) const = 0;
};

class LayerWrapper : public Layer {
protected:
    Layer* _layer;
public:
    ~LayerWrapper() {
        if (_layer != NULL) {
            delete _layer;
        }
    }
    
    void setLayer(Layer* layer) {
        _layer = layer;
    }
    
    virtual size_t numberOfInNodes() const {
        return _layer->numberOfInNodes();
    }
    
    virtual size_t numberOfOutNodes() const {
        return _layer->numberOfOutNodes();
    }
    
    virtual int indexOfUnitNode() const {
        return _layer->indexOfUnitNode();
    }
};

class DropoutLayer : public LayerWrapper {
private:
    data_t _rate;
    Vector _dropVector;
public:
    DropoutLayer(data_t rate) : _rate(rate) {
    }
    
    virtual void setTraining(bool training);
    
    virtual Matrix eval(const Matrix& input) const;
    
    virtual Array gDiff(const Matrix& z) const;
};

class BaseLayer : public Layer {
    bool _hasUnitNode;
public:
    BaseLayer(bool addUnitNode) : _hasUnitNode(addUnitNode) {
    }
    
    bool hasUnitNode() const {
        return _hasUnitNode;
    }
    
    virtual int indexOfUnitNode() const {
        if (!hasUnitNode()) {
            return -1;
        } else {
            return numberOfOutNodes() - 1;
        }
    }
};

class SimpleLayer : public BaseLayer {
private:
    int _numberOfInNodes;
protected:
    Activation* _activation = NULL;
public:
    SimpleLayer(std::string activation, int numberOfInNodes, bool addUnitNode) : BaseLayer(addUnitNode) {
        _numberOfInNodes = numberOfInNodes;
        
        if (activation != "") {
            _activation = Activation::create(activation);
        }
    }
    
    ~SimpleLayer() {
        if (_activation != NULL) {
            delete _activation;
        }
    }
    
    const Activation* getActivation() const {
        return _activation;
    }
    
    virtual size_t numberOfInNodes() const {
        return _numberOfInNodes;
    }
    
    virtual size_t numberOfOutNodes() const {
        if (hasUnitNode()) {
            return numberOfInNodes() + 1;
        } else {
            return numberOfInNodes();
        }
    }    
};

class FirstLayer : public BaseLayer {
private:
    std::vector<NumericColumn*> _numerics;
    std::vector<DiscreteColumn*> _discretes;
public:
    FirstLayer(bool addUnitNode = true) : BaseLayer(addUnitNode) {
    }
    
    FirstLayer(const std::vector<int>& simpleNumericCols, bool addUnitNode = true) : BaseLayer(addUnitNode) {
        for (int c : simpleNumericCols) {
            _numerics.push_back(new SimpleNumericColumn(c));
        }
    }
    
    ~FirstLayer() {
        for (int i = 0; i < _numerics.size(); i++) {
            delete _numerics[i];
        }
        
        for (int i = 0; i < _discretes.size(); i++) {
            delete _discretes[i];
        }
    }
    
    virtual size_t numberOfInNodes() const {
        throw DPLException("FirstLayer::numberOfInNodes not implemented.");
    }
        
    virtual size_t numberOfOutNodes() const {
        int ret = _numerics.size();
        
        if (hasUnitNode()) {
            return ret + 1;
        } else {
            return ret;
        }
    }
    
    size_t numberOfDiscreteNodes() const {
        int ret = 0;
        for (int i = 0; i < _discretes.size(); i++) {
            ret += _discretes[i]->range();
        }
        
        return ret;
    }
    
    void addFeatureColumn(FeatureColumn* column) {
        if (column->type() == numeric) {
            _numerics.push_back(dynamic_cast<NumericColumn*>(column));
        } else {
            _discretes.push_back(dynamic_cast<DiscreteColumn*>(column));
        }
    }
    
    Onehot evalDiscrete(const Matrix& input) const;
    
    virtual Matrix eval(const Matrix& input) const;
    
    virtual Array gDiff(const Matrix& z) const {
        throw DPLException("should not call FirstLayer.gDiff function.");
    }    
};

class SimpleHiddenLayer : public SimpleLayer {
public:
    SimpleHiddenLayer(std::string activation, int numberOfNodes, bool addUnitNode = true)
        : SimpleLayer(activation, numberOfNodes, addUnitNode) {
    }
    
    
    virtual Matrix eval(const Matrix& input) const;
    
    virtual Array gDiff(const Matrix& z) const;
};

class OutputLayer : public SimpleLayer {
public:
    OutputLayer(std::string activation, int numberOfNodes) : SimpleLayer(activation, numberOfNodes, false) {
    }
    
    virtual Matrix eval(const Matrix& input) const;
    
    virtual Matrix delta(const Matrix& x, const Matrix& y) const = 0;
    
    virtual data_t loss(const Matrix& x, const Matrix& y) const = 0;
    
    virtual Array gDiff(const Matrix& z) const {
        throw DPLException("should not call OutputLayer.gDiff function.");
    }
};

class RegressionOutputLayer : public OutputLayer {
public:
    RegressionOutputLayer(int numberOfNodes = 1) : OutputLayer("", numberOfNodes) {
    }
    
    virtual Matrix delta(const Matrix& x, const Matrix& y) const;
    
    virtual data_t loss(const Matrix& x, const Matrix& y) const;
};

class ClassificationOutputLayer : public OutputLayer {
public:
    ClassificationOutputLayer() : OutputLayer("sigmoid", 1) {
    }
    
    virtual Matrix delta(const Matrix& x, const Matrix& y) const;
    
    virtual data_t loss(const Matrix& x, const Matrix& y) const;
};

class SoftmaxOutputLayer : public OutputLayer {
public:
    SoftmaxOutputLayer(int numberOfNodes) : OutputLayer("", numberOfNodes) {
    }
    
    virtual Matrix eval(const Matrix& input) const;
    
    virtual Matrix delta(const Matrix& x, const Matrix& y) const;
    
    virtual data_t loss(const Matrix& x, const Matrix& y) const;
};

/*
class CustomLayer {
private:
    std::vector<const BaseNode*> _nodes;
    
//    void applyActivationInPlace(CVector& input) const;
public:
    CustomLayer(bool isLastLayer = false);
    ~CustomLayer();
    
    virtual size_t size() const {
        return _nodes.size();
    }
    
    void addNode(const BaseNode* node);
    
    const BaseNode* getNode(int i) const;
    
    virtual Matrix eval(const Matrix& input) const;
        
    virtual Array gDiff(const Matrix& z) const;
};*/

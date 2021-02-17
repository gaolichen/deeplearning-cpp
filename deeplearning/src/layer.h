#pragma once
#include "common.h"
#include "node.h"

class Layer {
private:
    bool _addUnitNode;
public:
    Layer(bool addUnitNode) {
        _addUnitNode = addUnitNode;
    }
    
    bool hasUnitNode() const {
        return _addUnitNode;
    }

    size_t numberOfOutNodes() const {
        return _addUnitNode ? numberOfInNodes() + 1 : numberOfInNodes();
    }
    
    virtual size_t numberOfInNodes() const = 0;
    virtual Matrix eval(const Matrix& input) const = 0;
    virtual Array gDiff(const Matrix& z) const = 0;
};

class SimpleLayer : public Layer {
private:
    int _numberOfNodes;
public:
    SimpleLayer(int numberOfNodes, bool addUnitNode) : Layer(addUnitNode) {
        _numberOfNodes = numberOfNodes;
    }
    
    virtual size_t numberOfInNodes() const {
        return _numberOfNodes;
    }
};

class FirstLayer : public SimpleLayer {
public:
    FirstLayer(int numberOfNodes, bool addUnitNode = true) : SimpleLayer(numberOfNodes, addUnitNode) {
    }
    
    virtual Matrix eval(const Matrix& input) const;
    
    virtual Array gDiff(const Matrix& z) const {
        throw DPLException("should not call FirstLayer.gDiff function.");
    }
};

class SimpleHiddenLayer : public SimpleLayer {
private:
    Activation* _activation = NULL;
public:
    SimpleHiddenLayer(std::string activation, int numberOfNodes, bool addUnitNode = true);
    
    ~SimpleHiddenLayer() {
        if (_activation != NULL) {
            delete _activation;
        }
    }
    
    virtual Matrix eval(const Matrix& input) const;
    
    virtual Array gDiff(const Matrix& z) const;
};

class OutputLayer : public SimpleLayer {
private:
    Activation* _activation = NULL;
public:
    OutputLayer(std::string activation, int numberOfNodes) : SimpleLayer(numberOfNodes, false) {
        if (activation != "") {
            _activation = Activation::create(activation);
        }
    }
    
    ~OutputLayer() {
        if (_activation != NULL) {
            delete _activation;
        }
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

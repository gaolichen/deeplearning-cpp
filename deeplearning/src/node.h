#pragma once
#include "common.h"
#include "activation.h"

class BaseNode {
private:
    Activation* _activation;
public:
    BaseNode(std::string name = "");
    ~BaseNode();
    
    data_t eval(data_t input) const;
    data_t diff(data_t input) const;
    virtual void foo() {
    }
};

class ConstNode : public BaseNode {
private:
public:
    ConstNode(std::string name = "");
};

class FirstLayerNode : public BaseNode {
private:
public:
    FirstLayerNode(std::string name = "");
};

class HiddenNode : public BaseNode {
private:
public:
    HiddenNode(std::string name = "");
};

class OutputNode : public BaseNode {
private:
public:
    OutputNode(std::string name = "");
    
    virtual Vector delta(const RVector& res, const RVector& label) const = 0;
    
    static OutputNode* create(std::string name = "");
};

class RegressionOutputNode : public OutputNode {
public:
    RegressionOutputNode();
    
    virtual Vector delta(const RVector& res, const RVector& label) const;
};


class ClassificationOutputNode : public OutputNode {
public:
    ClassificationOutputNode();
    
    virtual Vector delta(const RVector& res, const RVector& label) const;
};

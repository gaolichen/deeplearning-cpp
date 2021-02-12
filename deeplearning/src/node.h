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
};

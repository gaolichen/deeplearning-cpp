#include "node.h"

BaseNode::BaseNode(std::string name) {
    _activation = Activation::create(name);
}

BaseNode::~BaseNode() {
    if (_activation != NULL) {
        delete _activation;
        _activation = NULL;
    }
}

data_t BaseNode::eval(data_t input) const {
    if (_activation != NULL) {
        return _activation->eval(input);
    } else {
        return input;
    }
}

data_t BaseNode::diff(data_t input) const {
    return _activation->diff(input);
}

ConstNode::ConstNode(std::string name) : BaseNode(name) {
}

FirstLayerNode::FirstLayerNode(std::string name) : BaseNode(name) {
}

HiddenNode::HiddenNode(std::string name) : BaseNode(name) {
}

OutputNode::OutputNode(std::string name) : BaseNode(name) {
}

RegressionOutputNode::RegressionOutputNode() : OutputNode("trivial") {
}

Vector RegressionOutputNode::delta(const RVector& res, const RVector& label) const {
    return (res - label).transpose() / res.size();
}

ClassificationOutputNode::ClassificationOutputNode() : OutputNode("sigmoid") {
}

Vector ClassificationOutputNode::delta(const RVector& res, const RVector& label) const {
    return (res - label).transpose() / res.size();
}

OutputNode* OutputNode::create(std::string name) {
    if (name == "regression") {
        return new RegressionOutputNode();
    } else if (name == "classification") {
        return new ClassificationOutputNode();
    } else {
        return NULL;
    }
}


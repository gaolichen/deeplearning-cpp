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

ConstNode::ConstNode(std::string name) : BaseNode(name) {
}

FirstLayerNode::FirstLayerNode(std::string name) : BaseNode(name) {
}

HiddenNode::HiddenNode(std::string name) : BaseNode(name) {
}

OutputNode::OutputNode(std::string name) : BaseNode(name) {
}


#include "layer.h"

Layer::Layer() {
    _nodes.push_back(new ConstNode());
}

Layer::~Layer() {
    for (int i = 0; i < _nodes.size(); i++) {
        delete _nodes[i];
    }
    _nodes.clear();
}

void Layer::addNode(const BaseNode* node) {
    _nodes.push_back(node);
}

Vector Layer::eval(const Vector& input) const {
    Vector ret(_nodes.size());
    for (int i = 0; i < _nodes.size(); i++) {
        ret(i) = _nodes[i]->eval(input[i]);
    }
    
    return ret;
}

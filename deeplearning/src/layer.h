#pragma once
#include "common.h"
#include "node.h"

class Layer {
private:
    std::vector<const BaseNode*> _nodes;
public:
    Layer();
    ~Layer();
    
    size_t size() const {
        return _nodes.size();
    }
    
    void addNode(const BaseNode* node);
    
    Vector eval(const Vector& input) const; 
};

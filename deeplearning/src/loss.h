#pragma once
#include "common.h"

class LossFunction {
private:
public:
    virtual data_t loss(const Vector& res, const Vector& expected) = 0;
    
    static LossFunction* create(std::string name);
};

class L2 : public LossFunction {
public:
    virtual data_t loss(const Vector& res, const Vector& expected);
};

class L1 : public LossFunction {
public:
    virtual data_t loss(const Vector& res, const Vector& expected);
};

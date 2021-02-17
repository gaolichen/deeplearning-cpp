#pragma once
#include "common.h"

class LossFunction {
private:
public:
    virtual data_t loss(const RVector& res, const RVector& expected) = 0;
    
    virtual data_t diff(const RVector& res, const RVector& expected, int i) = 0;
    
    static LossFunction* create(std::string name);
};

class SquareLoss : public LossFunction {
public:
    virtual data_t loss(const RVector& res, const RVector& expected);
    virtual data_t diff(const RVector& res, const RVector& expected, int i);
};

class LinearLoss : public LossFunction {
public:
    virtual data_t loss(const RVector& res, const RVector& expected);
    virtual data_t diff(const RVector& res, const RVector& expected, int i);
};

class LogLoss : public LossFunction {
public:
    virtual data_t loss(const RVector& res, const RVector& expected);
    virtual data_t diff(const RVector& res, const RVector& expected, int i);
};

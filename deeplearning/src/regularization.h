#pragma once
#include "common.h"

class Regularization {
public:
    virtual data_t complixity(const Matrix& weight) = 0;
    
    data_t complixity(const std::vector<Matrix>& weights);
    
    virtual Matrix diff(const Matrix& weight) = 0;
    
    static Regularization* create(std::string name);
};

class L2 : public Regularization {
public:
    L2() {
        std::cout << "L2 regularization is created." << std::endl;
    }
    virtual data_t complixity(const Matrix& weight);
    
    virtual Matrix diff(const Matrix& weight);
};

class L1 : public Regularization {
public:
    L1() {
        std::cout << "L1 regularization is created." << std::endl;
    }
    
    virtual data_t complixity(const Matrix& weight);
    
    virtual Matrix diff(const Matrix& weight);
};

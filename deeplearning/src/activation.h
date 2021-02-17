#pragma once
#include "common.h"

class Activation {
private:
public:
    virtual data_t eval(data_t input) = 0;
    virtual data_t diff(data_t input) = 0;
    
    static Activation* create(std::string name);
};

class TrivialActivation : public Activation {
private:
public:
    TrivialActivation() {
    }
    
    virtual data_t eval(data_t input) {
        return input;
    }
    
    virtual data_t diff(data_t input) {
        return 1.0;
    }
};

class Sigmoid : public Activation {
public:
    Sigmoid();
    
    virtual data_t eval(data_t input);
    
    virtual data_t diff(data_t input);
};

class ReLU : public Activation {
private:
public:
    ReLU();
    
    virtual data_t eval(data_t input);

    virtual data_t diff(data_t input);
};


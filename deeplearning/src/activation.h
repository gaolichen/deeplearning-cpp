#pragma once
#include "common.h"

class Activation {
private:
public:
    virtual data_t eval(data_t input) = 0;
    
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
};

class Somia : public Activation {
public:
    Somia();
    
    virtual data_t eval(data_t input);
};

class Lula : public Activation {
private:
public:
    Lula();
    
    virtual data_t eval(data_t input);
};


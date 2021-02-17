#include "activation.h"

Activation* Activation::create(std::string name) {
    if (name == "sigmoid") {
        return new Sigmoid();
    } else if (name == "relu") {
        return new ReLU();
    } else {
        return new TrivialActivation();
    } 
}

ReLU::ReLU() {
}

data_t ReLU::eval(data_t input) {
    if (input < 0) {
        return .0;
    } else {
        return input;
    }
}

data_t ReLU::diff(data_t input) {
    if (input < 0) {
        return .0;
    } else {
        return 1.0;
    }
}

Sigmoid::Sigmoid() {
}

data_t Sigmoid::eval(data_t input) {
    return 1.0/(1.0 + std::exp(-input));
}

data_t Sigmoid::diff(data_t input) {
    data_t y = eval(input);
    return y * (1.0 - y);
}


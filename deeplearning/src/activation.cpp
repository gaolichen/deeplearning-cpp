#include "activation.h"

Activation* Activation::create(std::string name) {
    if (name == "somia") {
        return new Somia();
    } else if (name == "lula") {
        return new Lula();
    } else {
        return new TrivialActivation();
    } 
}

Lula::Lula() {
}

data_t Lula::eval(data_t input) {
    if (input < 0) {
        return .0;
    } else {
        return input;
    }
}

Somia::Somia() {
}

data_t Somia::eval(data_t input) {
    return 1.0/(1.0 + std::exp(-input));
}

#include "loss.h"

LossFunction* LossFunction::create(std::string name) {
    if (name == "L1") {
        return new LinearLoss();
    } else if (name == "log") {
        return new LogLoss();
    } else if (name == "square") {
        return new SquareLoss();
    } else {
        return NULL;
    }
}

data_t SquareLoss::loss(const RVector& res, const RVector& expected) {
    data_t ret = .0;
    for (size_t i = 0; i < res.size(); i++) {
        ret += (res(i) - expected(i)) * (res(i) - expected(i));
    }
    
    return ret * 0.5 / res.size();
}

data_t SquareLoss::diff(const RVector& res, const RVector& expected, int i) {
    return (res(i) - expected(i)) / res.size();
}

data_t LinearLoss::loss(const RVector& res, const RVector& expected) {
    data_t ret = .0;
    for (size_t i = 0; i < res.size(); i++) {
        ret += std::abs(res(i) - expected(i));
    }
    
    return ret / res.size();
}

data_t LinearLoss::diff(const RVector& res, const RVector& expected, int i) {
    if (res(i) + EPS < expected(i)) {
        -1.0 / res.size();
    } else if (res(i) - EPS > expected(i)) {
        return 1.0 / res.size();
    } else {
        return 0.0;
    }
}

data_t LogLoss::loss(const RVector& res, const RVector& expected) {
    data_t ret = .0;
    for (size_t i = 0; i < res.size(); i++) {
        ret += -expected(i) * log(res(i)) - (1 - expected(i)) * log(1 - res(i));
    }
    
    return ret / res.size();
}

data_t LogLoss::diff(const RVector& res, const RVector& expected, int i) {
    return .0;
}

#include "loss.h"

LossFunction* LossFunction::create(std::string name) {
    if (name == "L1") {
        return new L1();
    } else {
        return new L2();
    }
}

data_t L2::loss(const Vector& res, const Vector& expected) {
    data_t ret = .0;
    for (size_t i = 0; i < res.size(); i++) {
        ret += (res(i) - expected(i)) * (res(i) - expected(i));
    }
    
    return ret / res.size();
}

data_t L1::loss(const Vector& res, const Vector& expected) {
    data_t ret = .0;
    for (size_t i = 0; i < res.size(); i++) {
        ret += std::abs(res(i) - expected(i));
    }
    
    return ret / res.size();
}

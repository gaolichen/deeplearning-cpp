#include "regularization.h"

Regularization* Regularization::create(std::string name) {
    if (name == "L1") {
        return new L1();
    } else if (name == "L2") {
        return new L2();
    } else {
        throw DPLException("regularization \"" + name + "\" does not exist.");
    }
}

data_t Regularization::complixity(const std::vector<Matrix>& weights) {
    data_t ret = .0;
    for (int i = 0; i < weights.size(); i++) {
        ret += complixity(weights[i]);
    }
    return ret;
}

data_t L2::complixity(const Matrix& weight) {
    return weight.squaredNorm();
}

Matrix L2::diff(const Matrix& weight) {
    return 2.0 * weight;
}

data_t L1::complixity(const Matrix& weight) {
    return weight.array().abs().sum();
}

Matrix L1::diff(const Matrix& weight) {
    Matrix ret(weight.rows(), weight.cols());
    for (int i = 0; i < weight.rows(); i++) {
        for (int j = 0; j < weight.cols(); j++) {
/*            if (weight(i, j) > EPS) {
                ret(i, j) = 1.0;
            } else if (weight(i, j) < -EPS){
                ret(i, j) = -1.0;
            } else {
                ret(i, j) = .0;
            }*/
            if (weight(i, j) > 0) {
                ret(i, j) = 1.0;
            } else {
                ret(i, j) = -1.0;
            }
        }
    }
    return ret;
}


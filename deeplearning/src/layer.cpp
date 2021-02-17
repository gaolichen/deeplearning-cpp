#include "layer.h"

Matrix FirstLayer::eval(const Matrix& input) const {
    if (input.cols() != numberOfInNodes()) {
        throw DPLException("FirstLayer::eval: invalid input matrix size.");
    }
    
    Matrix ret;
    if (!hasUnitNode()) {
        ret = input.transpose();
    } else {
        // add one row
        ret.resize(input.cols() + 1, input.rows());
        ret.block(0, 0, 1, input.rows()) = Matrix::Constant(1, input.rows(), 1.0);
        ret.block(1, 0, input.cols(), input.rows()) = input.transpose();
    }
    
    return ret;
}

SimpleHiddenLayer::SimpleHiddenLayer(std::string activation, int numberOfNodes, bool addUnitNode)
    : SimpleLayer(numberOfNodes, addUnitNode) {    
    if (activation != "") {
        this->_activation = Activation::create(activation);
    }
}

Matrix SimpleHiddenLayer::eval(const Matrix& input) const {
    if (input.rows() != this->numberOfInNodes()) {
        throw DPLException("SimpleHiddenLayer::eval: invalid input matrix size.");
    }
    
    Matrix ret;
    int br = 0;
    if (!hasUnitNode()) {
        ret.resize(input.rows(), input.cols());
    } else {
        ret.resize(input.rows() + 1, input.cols());
        ret.block(0, 0, 1, input.cols()) = Matrix::Constant(1, input.cols(), 1.0);
        br = 1;
    }
    if (_activation == NULL) {
        ret.block(br, 0, input.rows(), input.cols()) = input;
    } else {
        for (int i = 0; i < input.rows(); i++) {
            for (int j = 0; j < input.cols(); j++) {
                ret(br + i, j) = _activation->eval(input(i, j));
            }
        }
    }
    
    return ret;
}

Array SimpleHiddenLayer::gDiff(const Matrix& z) const {
    if (z.rows() != numberOfInNodes()) {
        throw DPLException("invalid matrix size of z argument.");
    }
    
    if (_activation == NULL) {
        return Matrix::Constant(z.cols(), z.rows(), 1.0);
    } else {
        Array ret(z.cols(), z.rows());
        for (int i = 0; i < z.cols(); i++) {
            for (int j = 0; j < z.rows(); j++) {
                ret(i, j) = _activation->diff(z(j, i));
            }
        }
            
        return ret;
    }
}

Matrix OutputLayer::eval(const Matrix& input) const {
    if (input.rows() != this->numberOfInNodes()) {
        throw DPLException("OutputLayer::eval: invalid input matrix size.");
    }
    
    if (_activation == NULL) {
        return input.transpose();
    } else {
        Matrix ret(input.cols(), input.rows());
        for (int i = 0; i < input.rows(); i++) {
            for (int j = 0; j < input.cols(); j++) {
                ret(j, i) = _activation->eval(input(i, j));
            }
        }
        return ret;
    }
}

Matrix RegressionOutputLayer::delta(const Matrix& x, const Matrix& y) const {
    if (x.rows() != y.rows() || x.cols() != y.cols()) {
        throw DPLException("RegressionOutputLayer::delta: matrix size of x and y not match.");
    }
    return (x - y) / y.rows();
}
    
data_t RegressionOutputLayer::loss(const Matrix& x, const Matrix& y) const {
    if (x.rows() != y.rows() || x.cols() != y.cols()) {
        throw DPLException("RegressionOutputLayer::loss: matrix size of x and y not match.");
    }
    return (x - y).squaredNorm() / (2 * y.rows());
}

Matrix ClassificationOutputLayer::delta(const Matrix& x, const Matrix& y) const {
    if (x.rows() != y.rows() || x.cols() != y.cols()) {
        throw DPLException("matrix size of x and y not match.");
    }
    return (x - y) / y.rows();
}
    
data_t ClassificationOutputLayer::loss(const Matrix& x, const Matrix& y) const {
    // TODO
    return 0.0;
//    return (x - y).squaredNorm() / y.rows();
}

Matrix SoftmaxOutputLayer::eval(const Matrix& input) const {
    Matrix ret(input.cols(), input.rows());
    for (int i = 0; i < input.cols(); i++) {
        Array exps = input.col(i).array().exp();
        data_t num = exps.sum();
        for (int j = 0; j < input.rows(); j++) {
            ret(i, j) = exps(j) / num;
        }
    }
    
    return ret;
}

Matrix SoftmaxOutputLayer::delta(const Matrix& x, const Matrix& y) const {
    if (x.rows() != y.rows() || x.cols() != y.cols()) {
        throw DPLException("matrix size of x and y not match.");
    }
    // TODO
    return (x - y) / y.rows();
}
    
data_t SoftmaxOutputLayer::loss(const Matrix& x, const Matrix& y) const {
    // TODO
    return 0.0;
}
/*
CustomLayer::CustomLayer(bool isLastLayer) {
    if (!isLastLayer) {
        _nodes.push_back(new ConstNode());
    }
}

CustomLayer::~CustomLayer() {
    for (int i = 0; i < _nodes.size(); i++) {
        delete _nodes[i];
    }
    _nodes.clear();
}

void CustomLayer::addNode(const BaseNode* node) {
    _nodes.push_back(node);
}

const BaseNode* CustomLayer::getNode(int i) const {
    return _nodes[i];
}

Matrix CustomLayer::eval(const Matrix& data) const {
    Matrix ret(data.rows(), data.cols());
    for (int i = 0; i < data.cols(); i++) {
        for (int j = 0; j < _nodes.size(); j++) {
            ret(j, i) = _nodes[j]->eval(data(j, i));
        }
//        this->applyActivationInPlace(data.col(i));
    }

    return ret;
}

Array Layer::gDiff(const Matrix& z) const {
    Array ret(z.cols(), z.rows());
    for (int i = 0; i < z.cols(); i++) {
        for (int j = 0; j < z.rows(); j++) {
            ret(i, j) = this->_nodes[j]->diff(z(j, i));
        }
    }
    return ret;
}*/

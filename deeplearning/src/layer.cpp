#include "layer.h"
#include "datautil.h"

Matrix FirstLayer::eval(const Matrix& input) const {
    if (input.cols() != numberOfInNodes()) {
        throw DPLException("FirstLayer::eval: invalid input matrix size.");
    }
    
    Matrix ret = input;
    if (hasUnitNode()) {
        DataUtil::appendColumnProduct(ret, std::vector<int>());
    }
    
    for (int i = 0; i < _transformers.size(); i++) {
        DataUtil::appendCustomColumn(ret, _transformers[i].column, _transformers[i].fun);
    }
    
    for (int i = 0; i < _crossFeatures.size(); i++) {
        DataUtil::appendColumnProduct(ret, _crossFeatures[i]);
    }
    
    return ret.transpose();
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
        throw DPLException("ClassificationOutputLayer::delta: matrix size of x and y not match.");
    }
    if (x.cols() != numberOfOutNodes()) {
        throw DPLException("ClassificationOutputLayer::delta: number of columns in x is incorrect.");
    }
    return (x - y) / y.rows();
}
    
data_t ClassificationOutputLayer::loss(const Matrix& x, const Matrix& y) const {
    if (x.rows() != y.rows() || x.cols() != y.cols()) {
        throw DPLException("ClassificationOutputLayer::loss: matrix size of x and y not match.");
    }
    if (x.cols() != numberOfOutNodes()) {
        throw DPLException("ClassificationOutputLayer::loss: number of columns in x is incorrect.");
    }
    
    data_t ret = .0;
    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            ret -= (1 - y(i, j)) * log(1 - x(i, j)) + y(i, j) * log(x(i, j));
        }
    }
    return ret / x.rows();
}

Matrix SoftmaxOutputLayer::eval(const Matrix& input) const {
    if (input.rows() != this->numberOfInNodes()) {
        throw DPLException("SoftmaxOutputLayer::eval: invalid input matrix size.");
    }
    Matrix ret(input.cols(), input.rows());
    for (int i = 0; i < input.cols(); i++) {
//        Array exps = input.col(i).array().exp();
//        data_t num = exps.sum();
//        for (int j = 0; j < input.rows(); j++) {
//            ret(i, j) = exps(j) / num;
//        }
        Array tmp = (input.col(i).array() - input.col(i).maxCoeff()).exp();
        for (int j = 0; j < input.rows(); j++) {
            ret(i, j) = tmp(j) / tmp.sum();
        }
    }
    
    return ret;
}

Matrix SoftmaxOutputLayer::delta(const Matrix& x, const Matrix& y) const {
    if (x.rows() != y.rows() || x.cols() != y.cols()) {
        throw DPLException("SoftmaxOutputLayer::delta: matrix size of x and y not match.");
    }
    if (x.cols() != numberOfOutNodes()) {
        throw DPLException("SoftmaxOutputLayer::delta: number of columns in x is incorrect.");
    }
    
    Array arr(x.rows(), x.cols());
    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            if (std::abs(y(i, j) - 1) < EPS) {
                arr(i, j) = 1.0;
            } else if (std::abs(y(i, j)) < EPS) {
                arr(i, j) = -x(i, j) / (1 - x(i, j));
            } else {
                throw DPLException("SoftmaxOutputLayer::delta: the value labels can only be 0 or 1.");
            }
        }
    }
    
    Matrix rSum = arr.rowwise().sum().matrix().asDiagonal();
    return (rSum * x - arr.matrix())/x.rows();
}
    
data_t SoftmaxOutputLayer::loss(const Matrix& x, const Matrix& y) const {
    if (x.rows() != y.rows() || x.cols() != y.cols()) {
        std::cout << x.rows() << ' ' << x.cols() << std::endl;
        std::cout << y.rows() << ' ' << y.cols() << std::endl;
        throw DPLException("SoftmaxOutputLayer::loss: matrix size of x and y not match.");
    }
    if (x.cols() != numberOfOutNodes()) {
        throw DPLException("SoftmaxOutputLayer::loss: number of columns in x is incorrect.");
    }
    data_t ret = .0;
    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            if (std::abs(y(i, j)) < EPS) {
                ret -= (1 - y(i, j)) * log(1 - x(i, j));
            } else {
                ret -= y(i, j) * log(x(i, j)); 
            }
        }
    }
    return ret / x.rows();
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

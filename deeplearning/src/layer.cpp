#include "layer.h"
#include "datautil.h"

void Layer::setTraining(bool training) {
    this->_training = training;
}


void DropoutLayer::setTraining(bool training) {
    Layer::setTraining(training);
    if (!training) {
        _dropVector.resize(0);
    } else {
        _dropVector = DataUtil::randomDropoutVector(_layer->numberOfOutNodes(), _rate, {_layer->indexOfUnitNode()});
    }
}

Matrix DropoutLayer::eval(const Matrix& input) const {
    if (!isTraining()) {
        return _layer->eval(input);
    } else {
        return _dropVector.asDiagonal() * _layer->eval(input);
    }
}

Array DropoutLayer::gDiff(const Matrix& z) const {
    if (!isTraining()) {
        return _layer->gDiff(z);
    } else {
        if (_layer->indexOfUnitNode() >= 0) {
            // we assume the bias unit node is always the last node.
            return (_layer->gDiff(z).matrix() * _dropVector.block(0, 0, _layer->numberOfInNodes(), 1).asDiagonal()).array();
        } else {
            return (_layer->gDiff(z).matrix() * _dropVector.asDiagonal()).array();
        }
    }
}
    
Matrix FirstLayer::eval(const Matrix& input) const {
    int rows = numberOfOutNodes();
    Matrix ret(rows, input.rows());
    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < _numerics.size(); j++) {
            ret(j, i) = _numerics[j]->evalNumeric(input.row(i));
        }
        if (hasUnitNode()) {
            ret(_numerics.size(), i) = 1.0;
        }
    }
    return ret;    
}

Onehot FirstLayer::evalDiscrete(const Matrix& input) const {
    Onehot ret(_discretes.size(), input.rows(), false);
    if (_discretes.size() == 0) {
        return ret;
    }
    for (int j = 0; j < _discretes.size(); j++) {
        ret.range(j) = _discretes[j]->range();
    }
//    std::cout << "FirstLayer::evalDiscrete input.row(0)=" << input.row(0) << std::endl;
    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < _discretes.size(); j++) {
            ret(j, i) = _discretes[j]->evalDiscrete(input.row(i));
        }
    }
//    std::cout << "ret.data() = " << std::endl << ret.data() << std::endl;
    return ret;
}

Matrix SimpleHiddenLayer::eval(const Matrix& input) const {
    if (input.rows() != this->numberOfInNodes()) {
        throw DPLException("SimpleHiddenLayer::eval: invalid input matrix size.");
    }
    
    Matrix ret;
    if (!hasUnitNode()) {
        ret.resize(input.rows(), input.cols());
    } else {
        ret.resize(input.rows() + 1, input.cols());
        ret.block(indexOfUnitNode(), 0, 1, input.cols()) = Matrix::Constant(1, input.cols(), 1.0);
    }
    if (_activation == NULL) {
        ret.block(0, 0, input.rows(), input.cols()).noalias() = input;
    } else {
        for (int i = 0; i < input.rows(); i++) {
            for (int j = 0; j < input.cols(); j++) {
                ret(i, j) = _activation->eval(input(i, j));
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
        return Array::Constant(z.cols(), z.rows(), 1.0);
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

data_t ClassificationOutputLayer::accuracy(const Matrix& x, const Matrix& y) const {
    int accuracy = 0;
    for (int i = 0; i < x.rows(); i++) {
        if (x(i, 0) > _threshold && y(i) > 1 - EPS) {
            accuracy++;
        } else if (x(i, 0) < _threshold && y(i) < EPS) {
            accuracy++;
        }
    }
    return accuracy / (data_t)x.rows();
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
    if (x.rows() != y.rows() || y.cols() != 1) {
        std::cout << x.rows() << ' ' << x.cols() << std::endl;
        std::cout << y.rows() << ' ' << y.cols() << std::endl;
        throw DPLException("SoftmaxOutputLayer::delta: matrix size of x and y are not correct.");
    }
/*    if (x.rows() != y.rows() || x.cols() != y.cols()) {
        throw DPLException("SoftmaxOutputLayer::delta: matrix size of x and y not match.");
    }*/
    if (x.cols() != numberOfOutNodes()) {
        throw DPLException("SoftmaxOutputLayer::delta: number of columns in x is incorrect.");
    }
    
    Array arr(x.rows(), x.cols());
    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            if (std::abs(y(i, 0) - j) < EPS) {
                arr(i, j) = 1.0;
            } else {
                arr(i, j) = -x(i, j) / (1 - x(i, j));
            }
/*            if (std::abs(y(i, j) - 1) < EPS) {
                arr(i, j) = 1.0;
            } else if (std::abs(y(i, j)) < EPS) {
                arr(i, j) = -x(i, j) / (1 - x(i, j));
            } else {
                throw DPLException("SoftmaxOutputLayer::delta: the value labels can only be 0 or 1.");
            }*/
        }
    }
    
    Matrix rSum = arr.rowwise().sum().matrix().asDiagonal();
    return (rSum * x - arr.matrix())/x.rows();
}

data_t SoftmaxOutputLayer::loss(const Matrix& x, const Matrix& y) const {
/*    if (x.rows() != y.rows() || x.cols() != y.cols()) {
        std::cout << x.rows() << ' ' << x.cols() << std::endl;
        std::cout << y.rows() << ' ' << y.cols() << std::endl;
        throw DPLException("SoftmaxOutputLayer::loss: matrix size of x and y not match.");
    }*/
    if (x.rows() != y.rows() || y.cols() != 1) {
        std::cout << x.rows() << ' ' << x.cols() << std::endl;
        std::cout << y.rows() << ' ' << y.cols() << std::endl;
        throw DPLException("SoftmaxOutputLayer::loss: matrix size of x and y are not correct.");
    }
    if (x.cols() != numberOfOutNodes()) {
        throw DPLException("SoftmaxOutputLayer::loss: number of columns in x is incorrect.");
    }
    data_t ret = .0;
    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            if (abs(y(i, 0) - j) < EPS) {
                ret -= log(x(i, j));
            } else {
                ret -= log(1 - x(i, j));
            }
/*            if (std::abs(y(i, j)) < EPS) {
                ret -= (1 - y(i, j)) * log(1 - x(i, j));
            } else {
                ret -= y(i, j) * log(x(i, j)); 
            }*/
        }
    }
    return ret / x.rows();
}

data_t SoftmaxOutputLayer::accuracy(const Matrix& x, const Matrix& y) const {
    int accuracy = 0;
    for (int i = 0; i < x.rows(); i++) {
        Matrix::Index index;
        x.row(i).maxCoeff(&index);
        if (index == y(i)) {
            accuracy++;
        }
    }
    return accuracy / (data_t)x.rows();
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

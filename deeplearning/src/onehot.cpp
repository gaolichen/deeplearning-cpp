#include "onehot.h"


Onehot::Onehot(bool rowMajor) : _rowMajor(rowMajor) {
}

Onehot::Onehot(int rows, int cols, bool rowMajor) : _rowMajor(rowMajor) {
    this->resize(rows, cols);
}

const Matrix Onehot::expand() const {
    Matrix ret;
    if (rowMajor()) {
        ret = Matrix::Zero(_data.rows(), totalRange());
        for (int i = 0; i < _data.rows(); i++) {
            int base = 0;
            for (int j = 0; j < _data.cols(); j++) {
                ret(i, base + _data(i, j)) = 1.0;
                base += range(j);
            }
        }
    } else {
        ret = Matrix::Zero(totalRange(), _data.cols());
        int base = 0;
        for (int i = 0; i < _data.rows(); i++) {
            for (int j = 0; j < _data.cols(); j++) {
                ret(base + _data(i, j), j) = 1.0;
                
            }
            base += range(i);
        }
    }
    
    return ret;
}

// onehot * matrix.
Matrix Onehot::operator * (const Matrix& mat) const {
    if (!this->_rowMajor) {
        if (_data.cols() != mat.rows()) {
            std::cout << "_data.cols()=" << _data.cols() << " mat.rows()=" << mat.rows() << std::endl;
            throw DPLException("onehot right multiplication 1: matrix size not match.");
        }
        
        // column major means a column is a onehot vector.
        int range = _ranges.sum();
        Matrix ret = Matrix::Zero(range, mat.cols());
        int base = 0;
        for (int i = 0; i < _data.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
                for (int k = 0; k < _data.cols(); k++) {
                    ret(_data(i, k) + base, j) += mat(k, j);
                }
            }
            base += _ranges[i];
        }
        return ret;        
    } else {
        if (totalRange() != mat.rows()) {
            throw DPLException("onehot right multiplication 2: matrix size not match.");
        }
        // row major means a row is a onehot vector.
        Matrix ret = Matrix::Zero(_data.rows(), mat.cols());

        for (int i = 0; i < _data.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
                int base = 0;
                for (int k = 0; k < _data.cols(); k++) {
                    ret(i, j) += mat(base + _data(i, k), j);
                    base += _ranges[k];
                }
            }
        }
        
        return ret;
    }
}

// matrix * onehot
Matrix operator * (const Matrix& mat, const Onehot& onehot) {
    if (onehot.rowMajor()) {
        // row major.
        if (mat.cols() != onehot.rows()) {
            throw DPLException("onehot left multiplication 1: matrix size not match.");
        }
        // ret(i, j) = \sum_{k} mat(i, k) * onehot(k, j);
        // ret(i, onehot(k, j) + base) = \sum_{k} mat(i, k) 
        int range = onehot.totalRange();
        Matrix ret = Matrix::Zero(mat.rows(), range);
        
        for (int i = 0; i < mat.rows(); i++) {
            int base = 0;
            for (int j = 0; j < onehot.cols(); j++) {
                for (int k = 0; k < onehot.rows(); k++) {
                    ret(i, base + onehot(k, j)) += mat(i, k);
                }
                base += onehot.range(j);
            }
        }
        return ret;
    } else {
        // column major.
        if (mat.cols() != onehot.totalRange()) {
            throw DPLException("onehot left multiplication 2: matrix size not match.");
        }
        Matrix ret = Matrix::Zero(mat.rows(), onehot.cols());

        for (int i = 0; i < mat.rows(); i++) {
            for (int j = 0; j < onehot.cols(); j++) {
                int base = 0;
                for (int k = 0; k < onehot.rows(); k++) {
//                    std::cout << "i=" << i << ", base=" << base << ", onehot(k,j)=" << onehot(k, j) << std::endl;
                    ret(i, j) += mat(i, base + onehot(k, j));
                    base += onehot.range(k);
                }
            }
        }
        
        return ret;
    }
}


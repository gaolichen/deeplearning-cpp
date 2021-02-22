#pragma once
#include "common.h"

class Onehot {
private:
    bool _rowMajor;
    Eigen::Array<int, Eigen::Dynamic, 1> _ranges;
    MatrixI _data;
public:
    Onehot(bool rowMajor = true);
    
    Onehot(int rows, int cols, bool rowMajor = true);
    
    void resize(int rows, int cols) {
        _data.resize(rows, cols);
        if (_rowMajor) {
            _ranges.resize(cols);
        } else {
            _ranges.resize(rows);
        }
    }
    
    int operator() (int rows, int cols) const {
        return _data(rows, cols);
    }
    
    int& operator() (int rows, int cols) {
        return _data(rows, cols);
    }
    
    void transposeInPlace() {
        _rowMajor = !_rowMajor;
        _data.transposeInPlace();
    }
    
    size_t rows() const {
        return _data.rows();
    }
    
    size_t cols() const {
        return _data.cols();
    }
    
    int range(int i) const {
        return _ranges(i);
    }
    
    int& range(int i) {
        return _ranges(i);
    }
        
    int totalRange() const {
        return _ranges.sum();
    }
    
    size_t numberOfFlags() const {
        if (_rowMajor) {
            return _data.cols();
        } else {
            return _data.rows();
        }
    }
    
    bool rowMajor() const {
        return _rowMajor;
    }
    
    const MatrixI& data() const {
        return _data;
    }
    
    const Matrix expand() const;
    
    Matrix operator * (const Matrix& mat) const;
};

Matrix operator * (const Matrix& mat, const Onehot& onehot);

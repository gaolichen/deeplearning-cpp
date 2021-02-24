#pragma once
#include "common.h"


enum ColumnType {
    numeric = 0,
    discrete = 1,
};

class FeatureColumn {
public:
    virtual ColumnType type() const = 0;
    virtual data_t evalNumeric(const RVector& row) const = 0;
    virtual int evalDiscrete(const RVector& row) const = 0;
};

class ISimpleColumn {
public:
    virtual int columnIndex() const = 0;
};

class NumericColumn : public FeatureColumn {
public:
    virtual ColumnType type() const {
        return numeric;
    }
        
    virtual int evalDiscrete(const RVector& row) const {
        throw new DPLException("NumericColumn::evalDiscrete: Should not be called.");
    };
};

class SimpleNumericColumn : public NumericColumn, ISimpleColumn {
private:
    int _columIndex;
public:
    SimpleNumericColumn(int columnIndex) : _columIndex(columnIndex) {
    }

    virtual int columnIndex() const {
        return _columIndex;
    }
    
    virtual data_t evalNumeric(const RVector& row) const {
        if (row.size() <= this->columnIndex()) {
            throw DPLException("SimpleNumericColumn::evalNumeric the size of row is too small.");
        }
        
        return row(this->columnIndex());
    }
};

class NumericCrossColumn : public NumericColumn {
private:
    std::vector<int> _cols;
public:
    NumericCrossColumn(const std::vector<int>& cols) : _cols(cols) {
    }
        
    virtual data_t evalNumeric(const RVector& row) const {
        data_t ret = 1.0;
        for (int i = 0; i < _cols.size(); i++) {
            ret *= row(_cols[i]);
        }
        return ret;
    }
};

class DiscreteColumn : public FeatureColumn {
public:
    virtual ColumnType type() const {
        return discrete;
    }
    
    virtual int range() const = 0;
    
    virtual data_t evalNumeric(const RVector& row) const {
        throw DPLException("DiscreteColumn::evalNumeric should not be called.");
    }
};

class SimpleDiscreteColumn : public DiscreteColumn, ISimpleColumn {
private:
    int _range;
    int _columnIndex;
public:
    SimpleDiscreteColumn(int columnIndex, int range) : _columnIndex(columnIndex), _range(range) {
    }
        
    virtual int evalDiscrete(const RVector& row) const {
        data_t v = row(this->columnIndex());
        if (v < 0 || v > _range - 1 || abs(floor(v + 0.1) - v) > EPS) {
            throw DPLException("DiscreteColumn::evaluate v is not integer");
        }
        return (int)v;
    }
    
    virtual int columnIndex() const {
        return _columnIndex;
    }
    
    int range() const {
        return _range;
    }
};

class BucketedColumn : public SimpleDiscreteColumn {
private:
    data_t _minValue;
    data_t _maxValue;
    data_t _resolution;
public:
    BucketedColumn(int columnIndex, data_t minValue, data_t maxValue, data_t resolution)
        : SimpleDiscreteColumn(columnIndex, ceil((maxValue - minValue)/resolution)), _minValue(minValue), _maxValue(maxValue), _resolution(resolution) {
    }
    
    virtual int evalDiscrete(const RVector& row) const {
        data_t v = row(this->columnIndex());
        if (v < _minValue || v > _maxValue) {
            std::cout << "BucketedColumn::evalDiscrete: v is out of range. ";
            std::cout << "v=" << v << " minValue=" << _minValue << " maxValue=" << _maxValue << std::endl;
//            std::cout << "row=" << row << std::endl;
//            throw DPLException("BucketedColumn::evaluate v is out of range.");
            return v < _minValue ? 0 : range() - 1;
        }
        return (int)floor((v - _minValue) / _resolution);
    }
    
    data_t resolution() {
        return this->_resolution;
    }
};


class DiscreteCrossColumn : public DiscreteColumn {
private:
    const std::vector<DiscreteColumn*> _cols;
    int _range;
public:
    DiscreteCrossColumn(const std::vector<DiscreteColumn*> cols) : _cols(cols) {
        _range = 1;
        for (int i = 0; i < _cols.size(); i++) {
            _range *= _cols[i]->range();
        }
    }
    
    ~DiscreteCrossColumn() {
        for (int i = 0; i < _cols.size(); i++) {
            delete _cols[i];
        }
    }
    
    virtual int range() const {
        return _range;
    }
        
    virtual int evalDiscrete(const RVector& row) const {
        int ret = 0;
        int base = 1;
        for (int i = 0; i < _cols.size(); i++) {
            ret += _cols[i]->evalDiscrete(row) * base;
            base *= _cols[i]->range();
        }
        
        return ret;
    }
};

#pragma once
#include "common.h"

typedef data_t (*dataTransformFun)(data_t);

class DataUtil {
public:
    static void appendColumnProduct(Matrix& mat, std::vector<int> cols);
    
    static void appendCustomColumn(Matrix& mat, int column, dataTransformFun);
};

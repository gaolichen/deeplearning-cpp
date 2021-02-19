#include "datautil.h"

void DataUtil::appendColumnProduct(Matrix& mat, std::vector<int> cols) {
    mat.conservativeResize(mat.rows(), mat.cols() + 1);
    if (cols.size() == 0) {
        mat.col(mat.cols() - 1) = Vector::Constant(mat.rows(), 1.0);
        return;
    }
    Array col = mat.col(cols[0]);
    for (int i = 1; i < cols.size(); i++) {
        col = col * mat.col(cols[i]).array();
    }
    
    mat.col(mat.cols() - 1) = col.matrix();
}

void DataUtil::appendCustomColumn(Matrix& mat, int column, dataTransformFun fun) {
    mat.conservativeResize(mat.rows(), mat.cols() + 1);
    int lastCol = mat.cols() - 1;
    for (int i = 0; i < mat.rows(); i++) {
        mat(i, lastCol) = fun(mat(i, column));
    }
}

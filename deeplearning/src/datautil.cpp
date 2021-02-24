#include <fstream>
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

std::vector<data_t> DataUtil::smooth(const std::vector<data_t>& data, int range) {
    std::vector<data_t> ret(data.size(), 0.0);
    for (int i = 0; i < data.size(); i++) {
        int cnt = 0;
        for (int j = -range; j <= 0; j++) {
//        for (int j = -range; j <= 0; j++) {
            if (j + i < 0 || j + i >= data.size()) continue;
            ret[i] += data[i + j];
            cnt++;
        }
        ret[i] /= cnt;
    }
    return ret;
}

std::vector<int> DataUtil::randomIndex(int range) {
    std::vector<int> ret(range);
    for (int i = 0; i < ret.size(); i++) {
        ret[i] = i;
    }
    
    std::random_shuffle(ret.begin(), ret.end());    
    return ret;
}

PermutationMatrix DataUtil::randomPermutation(int size) {
    PermutationMatrix perm(size);
    perm.setIdentity();
    std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
    return perm;
}

Matrix DataUtil::randomRowShuffle(const Matrix& data) {
    PermutationMatrix perm = randomPermutation(data.rows());
    return perm * data;
}

IndexType DataUtil::randomSubIndex(int range, int setSize) {
    std::vector<int> indies = pickRandomIndex(range, setSize);
    return Eigen::Map<Eigen::ArrayXi>(&indies[0], indies.size());
}

std::vector<data_t> DataUtil::eigenVectorToSTL(const Vector& vec, bool smooth) {
    if (smooth) {
        return DataUtil::smooth(std::vector<data_t>(vec.data(), vec.data() + vec.size()), 5);
    } else {
        return std::vector<data_t>(vec.data(), vec.data() + vec.size());
    }
}

std::vector<data_t> DataUtil::eigenArrayToSTL(const Array& array, bool smooth) {
    if (smooth) {
        return DataUtil::smooth(std::vector<data_t>(array.data(), array.data() + array.size()), 5);
    } else {
        return std::vector<data_t>(array.data(), array.data() + array.size());
    }
}

Array DataUtil::colWiseStd(const Matrix& mat) {
    Matrix diff = mat.rowwise() - mat.colwise().mean().matrix();
    return (diff.colwise().squaredNorm().array() / diff.rows()).sqrt();
}

Matrix DataUtil::zScoreNormalize(const Matrix& mat) {
    Matrix diff = mat.rowwise() - mat.colwise().mean().matrix();
    RVector std = (diff.colwise().squaredNorm().array() / diff.rows()).sqrt().matrix();
    for (int i = 0; i < diff.cols(); i++) {
        diff.col(i) /= std(i);
    }
    return diff;
}

Vector DataUtil::randomDropoutVector(int size, data_t dropRate, const std::vector<int>& noDropPositions) {
    Vector ret(size);
    for (int i = 0; i < size; i++) {
        if (std::find(noDropPositions.begin(), noDropPositions.end(), i) != noDropPositions.end()) {
            ret(i) = 1;
        } else {
            if (random(0, 1) < dropRate) {
                ret(i) = 0;
            } else {
                ret(i) = 1/(1 - dropRate);
            }
        }
    }
    
    return ret;
}

void CSVData::read(std::string path, bool hasHeader) {
    std::ifstream fin;
    fin.open(path);
    
    std::string line;
    if (hasHeader) {
        std::getline(fin, line);
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            _headers.push_back(cell);
        }
//        std::cout << "line=" << line << std::endl;
    }
    
    int rows = 0;
    while (std::getline(fin, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            _innerData.push_back(std::stod(cell));
        }
        ++rows;
    }
    fin.close();
//    std::cout << "_innerData.size() = " << _innerData.size() << std::endl;
    
    _data = Eigen::Map<Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(_innerData.data(), rows, _innerData.size()/rows);
}

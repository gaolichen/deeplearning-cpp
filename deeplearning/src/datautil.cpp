#include <fstream>
#include <iomanip>
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

#define BYTE2INT(buf) (((unsigned char)buf[0] << 24) + ((unsigned char)buf[1] << 16) + ((unsigned char)buf[2] << 8) + (unsigned char)buf[3])

void DataUtil::readMNISTimage(std::string path, Matrix& data, bool normalize) {
    std::ifstream fin (path.c_str(), std::ios::in | std::ios::binary);
    char *buf = new char[28 * 28 + 1];
    fin.read(buf, 4);
    fin.read(buf, 4);
    int count = BYTE2INT(buf);
    
    fin.read(buf, 4);
    int r = BYTE2INT(buf);
    fin.read(buf, 4);
    int c = BYTE2INT(buf);
    data.resize(count, r * c);
    for (int i = 0; i < count; i++) {
        fin.read(buf, r * c);
        for (int j = 0; j < r * c; j++) {
            if (normalize) {
                data(i, j) = (unsigned char)buf[j] / 255.0;
            } else {
                data(i, j) = (unsigned char)buf[j];
            }
        }
    }
    
    delete[] buf;
    fin.close();
}

void DataUtil::readMNISTlabel(std::string path, Vector& data) {
    std::ifstream fin (path.c_str(), std::ios::in | std::ios::binary);
    char *buf = new char[10];
    fin.read(buf, 4);
    fin.read(buf, 4);
    int count = BYTE2INT(buf);
    data.resize(count);
    for (int i = 0; i < count; i++) {
        fin.read(buf, 1);
        data(i) = (unsigned char)buf[0];
    }
    
    delete[] buf;
    fin.close();
}

void DataUtil::showMNISTimage(const RVector& row) {
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            std::cout << std::setw(3) << row(i * 28 + j) << ' ';
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl;
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

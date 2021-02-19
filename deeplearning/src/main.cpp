#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <sstream>
#include "common.h"
using namespace std;

void TestMatrix() {
    Matrix v3 = Matrix::Random(10, 1);
    std::cout << "v3=" << v3.transpose() << std::endl;
    v3.resize(5, 1);
    std::cout << "after resize v3=" << v3 << std::endl;
    
    Matrix mat = Matrix::Random(3, 2);
    std::cout << "mat=" << std::endl << mat << std::endl;
    for (int i = 0; i < mat.rows(); i++) {
        std::cout << "row(" << i << ").norm()=" << mat.row(i).norm() << std::endl;
    }
}

void TestArray() {
    Array a = Array::Random(2, 3);
    std::cout << "a=" << a << std::endl;
    std::cout << "1-a=" << 1.0 - a << std::endl;
    std::cout <<"a/(1-a) = " << a/(1-a) << std::endl;
    std::cout << "a.rowwise().sum()=" << a.rowwise().sum().matrix() << std::endl;
    Matrix mat = a.rowwise().sum().matrix().asDiagonal();
    std::cout << "mat=" << mat << std::endl;
}

void TestPickRandomIndex() {
//    std::cout << pickRandomIndex(1000, 500) << std::endl;
}

void TestIndexing() {
    Matrix mat = Matrix::Random(10, 3);
    Eigen::ArrayXi rows(6);
    rows << 1, 3, 4, 6, 7, 8;
    Matrix mat2 = indexing(mat, rows);
    std::cout << "mat=" << mat << std::endl;
    std::cout << "mat2=" << mat2 << std::endl;
    
    std::vector<int> v = pickRandomIndex(10, 6);
    Eigen::Map<Eigen::ArrayXi> av(&v[0], v.size());
    std::cout << "av=" << av << std::endl;
    Matrix mat3 = indexing(mat, av);
    
    std::cout << "mat3=" << mat3 << std::endl;
}

int main(int argc, char* argv[])
{
    std::cout << "deep learning..." << std::endl;
    TestArray();
//    TestIndexing();
	return 0;
}


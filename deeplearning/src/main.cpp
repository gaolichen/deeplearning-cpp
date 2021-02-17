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
    Vector v3 = Vector::Constant(3, 1);
    Vector v2 = Vector::Constant(2, 1);
    Matrix mat = Matrix::Random(3, 2);
//    Matrix mat2 = Matrix::Random(
    std::cout << "mat=" << mat << std::endl;
    
    std::cout << "v3.transpose() * mat=" << v3.transpose() * mat << std::endl << std::endl;
    
    std::cout << "mat * v3 =" << mat * v3 << std::endl << std::endl;
    
    std::cout << "mat * v2=" << mat * v2 << std::endl << std::endl;
    std::cout << "v2 * mat=" << v2 * mat << std::endl << std::endl;
    std::cout << "v2.transpose() = " << v2.transpose() << std::endl;
}

int main(int argc, char* argv[])
{
    std::cout << "deep learning..." << std::endl;
    TestMatrix();
	return 0;
}


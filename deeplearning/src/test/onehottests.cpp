//Link to Boost
#define BOOST_TEST_DYN_LINK

//VERY IMPORTANT - include this last
//#include <boost/test/included/unit_test.hpp>
#include <boost/test/unit_test.hpp>

#include "test.h"
#include "onehot.h"

// test suite
BOOST_FIXTURE_TEST_SUITE(Onehot_suite, SimpleTestFixture, * utf::label("OnehotTests"))

BOOST_DATA_TEST_CASE(expand_test1, bdata::random(50, 100) ^ bdata::random(3, 10) ^ bdata::xrange(20), rows, cols, index)
{
    Onehot oh(rows, cols, true);
    for (int i = 0; i < cols; i++) {
        oh.range(i) = random64(10) + 10;
    }
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            oh(i, j) = random64(oh.range(j) - 1);
        }
    }
    
    Matrix mat = oh.expand();
    BOOST_TEST_REQUIRE(mat.sum() == rows * cols);
    
    for (int i = 0; i < mat.rows(); i++) {
        int base = 0;
        int k = 0;
        for (int j = 0; j < mat.cols(); j++) {
            if (mat(i, j) > EPS) {
                BOOST_TEST_REQUIRE(mat(i, j) == 1.0);
                BOOST_TEST_REQUIRE(oh(i, k) == j - base);
                base += oh.range(k);
                k++;
            }
        }
    }
}

BOOST_DATA_TEST_CASE(expand_test2, bdata::random(3, 10) ^ bdata::random(50, 100) ^ bdata::xrange(20), rows, cols, index)
{
    Onehot oh(rows, cols, false);
    for (int i = 0; i < rows; i++) {
        oh.range(i) = random64(10) + 10;
    }
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            oh(i, j) = random64(oh.range(i) - 1);
        }
    }
    
    Matrix mat = oh.expand();
    BOOST_TEST_REQUIRE(mat.sum() == rows * cols);
    
    for (int i = 0; i < mat.cols(); i++) {
        int base = 0;
        int k = 0;
        for (int j = 0; j < mat.rows(); j++) {
            if (mat(j, i) > EPS) {
                BOOST_TEST_REQUIRE(mat(j, i) == 1.0);
                BOOST_TEST_REQUIRE(oh(k, i) == j - base);
                base += oh.range(k);
                k++;
            }
        }
    }
}

BOOST_DATA_TEST_CASE(transpose_test, bdata::random(3, 10) ^ bdata::random(50, 100) ^ bdata::xrange(20), rows, cols, index)
{
    Onehot oh(rows, cols, false);
    for (int i = 0; i < rows; i++) {
        oh.range(i) = random64(10) + 10;
    }
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            oh(i, j) = random64(oh.range(i) - 1);
        }
    }
    
    Matrix mat1 = oh.expand();
    oh.transposeInPlace();
    Matrix mat2 = oh.expand();
    BOOST_TEST_REQUIRE((mat2 - mat1.transpose()).norm() < EPS);    
}

BOOST_DATA_TEST_CASE(multiplication_test1, bdata::random(50, 100) ^ bdata::random(3, 10) ^ bdata::xrange(20), rows, cols, index)
{
    Onehot oh(rows, cols, true);
    for (int i = 0; i < cols; i++) {
        oh.range(i) = random64(10) + 10;
    }
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            oh(i, j) = random64(oh.range(j) - 1);
        }
    }
    
    Matrix mat1 = Matrix::Random(oh.totalRange(), 10);
    Matrix mat2 = Matrix::Random(10, rows);
    Matrix expandOh = oh.expand();
    
    BOOST_TEST_REQUIRE((oh * mat1 - expandOh * mat1).norm() < EPS);
    BOOST_TEST_REQUIRE((mat2 * oh - mat2 * expandOh).norm() < EPS);
}

BOOST_DATA_TEST_CASE(multiplication_test2, bdata::random(50, 100) ^ bdata::random(3, 10) ^ bdata::xrange(20), rows, cols, index)
{
    Onehot oh(rows, cols, false);
    for (int i = 0; i < rows; i++) {
        oh.range(i) = random64(10) + 10;
    }
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            oh(i, j) = random64(oh.range(i) - 1);
        }
    }
    
    Matrix mat1 = Matrix::Random(cols, 10);
    Matrix mat2 = Matrix::Random(10, oh.totalRange());
    Matrix expandOh = oh.expand();
    
    BOOST_TEST_REQUIRE((oh * mat1 - expandOh * mat1).norm() < EPS);
    BOOST_TEST_REQUIRE((mat2 * oh - mat2 * expandOh).norm() < EPS);
}

BOOST_AUTO_TEST_SUITE_END()

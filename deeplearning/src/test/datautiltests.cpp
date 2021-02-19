//Link to Boost
#define BOOST_TEST_DYN_LINK

//VERY IMPORTANT - include this last
//#include <boost/test/included/unit_test.hpp>
#include <boost/test/unit_test.hpp>

#include "test.h"
#include "../datautil.h"

// test suite
BOOST_FIXTURE_TEST_SUITE(DataUtil_suite, SimpleTestFixture, * utf::label("DataUtilSeries"))

BOOST_DATA_TEST_CASE(appendColumnProduct_test, bdata::random(2, 10) ^ bdata::random(2, 10) ^ bdata::xrange(20), rows, cols, index)
{
    Matrix mat = Matrix::Random(rows, cols);
    
    int a = random64(cols - 1);
    int b = random64(cols - 1);
    DataUtil::appendColumnProduct(mat, std::vector<int>{a, b});
    BOOST_TEST(mat.cols() == cols + 1);
    for (int i = 0; i < mat.rows(); i++) {
        BOOST_TEST_REQUIRE(abs(mat(i, cols) - mat(i, a) * mat(i, b)) < EPS);
    }
}

BOOST_DATA_TEST_CASE(appendColumnProduct_emptycolumn_test, bdata::random(2, 10) ^ bdata::random(2, 10) ^ bdata::xrange(20), rows, cols, index)
{
    Matrix mat = Matrix::Random(rows, cols);
    DataUtil::appendColumnProduct(mat, std::vector<int>());
    BOOST_TEST(mat.cols() == cols + 1);
    for (int i = 0; i < mat.rows(); i++) {
        BOOST_TEST_REQUIRE(abs(mat(i, cols) - 1.0) < EPS);
    }
}

BOOST_AUTO_TEST_SUITE_END()

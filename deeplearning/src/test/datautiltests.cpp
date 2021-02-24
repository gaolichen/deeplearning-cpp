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

BOOST_DATA_TEST_CASE(randomIndex_test, bdata::random(15, 16) ^ bdata::xrange(20), range, index)
{
    std::vector<int> v1 = DataUtil::randomIndex(range);
    std::vector<int> v2 = DataUtil::randomIndex(range);
    BOOST_TEST_REQUIRE(v1.size() == range);
    BOOST_TEST_REQUIRE(v2.size() == range);
    BOOST_TEST_REQUIRE(CompareVector(v1, v2) != 0);
    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());
    BOOST_TEST_REQUIRE(CompareVector(v1, v2) == 0);
    for (int i = 0; i < v1.size(); i++) {
        BOOST_TEST_REQUIRE(v1[i] == i);
    }
}

BOOST_DATA_TEST_CASE(randomRowShuffle, bdata::random(15, 16) ^ bdata::random(2, 7) ^ bdata::xrange(20), rows, cols, index)
{
    Matrix mat = Matrix::Random(rows, cols);
    Matrix mat2 = DataUtil::randomRowShuffle(mat);
    BOOST_TEST_REQUIRE(mat.rows() == mat2.rows());
    BOOST_TEST_REQUIRE(mat.cols() == mat2.cols());
    
    for (int i = 0; i < mat.rows(); i++) {
        bool found = false;
        for (int j = 0; j < mat2.rows(); j++) {
            if (mat2.row(j) == mat.row(i)) {
                found = true;
                break;
            }
        }
        BOOST_TEST_INFO("i=" << i);
        BOOST_TEST_REQUIRE(found);
    }
}

BOOST_DATA_TEST_CASE(randomDropoutVector, bdata::random(20, 50) ^ bdata::random(1,5) ^ bdata::xrange(10), size, noDropSize, index)
{
    std::vector<int> noDrops = pickRandomIndex(size, noDropSize);    
    data_t rate = random(0, 1);
    
    int loops = 500;
    data_t sum = .0;
    for (int i = 0; i < loops; i++) {
        Vector res = DataUtil::randomDropoutVector(size, rate, noDrops);
        for (int j = 0; j < noDrops.size(); j++) {
            BOOST_TEST_REQUIRE(res(noDrops[j]) == 1);
        }

        for (int j = 0; j < size; j++) {
            if (std::find(noDrops.begin(), noDrops.end(), j) == noDrops.end()) {
                BOOST_TEST_REQUIRE(abs(res(j) * (res(j) - 1.0/(1 - rate))) < EPS);
            }
        }
        sum += res.sum();
    }
    
    data_t expected = loops * size;
    
    BOOST_TEST_INFO("sum=" << sum << ", expected=" << expected << ", dropRate=" << rate);
    BOOST_TEST_REQUIRE(abs((sum - expected)/expected) < 0.1);
}


BOOST_DATA_TEST_CASE(colWiseStd_test, bdata::random(2, 7) ^ bdata::random(2, 7) ^ bdata::xrange(1), rows, cols, index)
{
    Matrix mat(3, 2);
    mat << 1, 2, 3, 4, 5, 6;
    std::cout << "mat=" << mat << std::endl;
    std::cout << "std=" << DataUtil::colWiseStd(mat) << std::endl;
    
    std::cout << "z-norm=" << DataUtil::zScoreNormalize(mat) << std::endl;
}

BOOST_DATA_TEST_CASE(randomSubIndex, bdata::random(15, 16) ^ bdata::random(2, 7) ^ bdata::xrange(2), range, setSize, index)
{
//    IndexType it(NULL, 0);
//    std::cout << it.size() << std::endl;
    std::cout << range << ' ' << setSize << std::endl;
    std::vector<int> indices = pickRandomIndex(range, setSize);
    IndexType it(&indices[0], indices.size());
//    std::cout << "it=" << it << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()


// test suite
BOOST_FIXTURE_TEST_SUITE(CSVData_suite, SimpleTestFixture, * utf::disabled())

BOOST_AUTO_TEST_CASE(read_test)
{
    CSVData csv;
    csv.read("/home/gaolichen/gitroot/prototype/deeplearning/california_housing_test.csv", true);
    std::cout << csv.headers() << std::endl;
    BOOST_TEST_REQUIRE(csv.headers().size() == csv.data().cols());
    BOOST_TEST_REQUIRE(csv.data().rows() > 0);
}

BOOST_AUTO_TEST_SUITE_END()

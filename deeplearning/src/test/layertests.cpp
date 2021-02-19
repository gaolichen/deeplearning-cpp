//Link to Boost
#define BOOST_TEST_DYN_LINK

//VERY IMPORTANT - include this last
//#include <boost/test/included/unit_test.hpp>
#include <boost/test/unit_test.hpp>

#include "test.h"
#include "layer.h"

// test suite
BOOST_FIXTURE_TEST_SUITE(Layer_suite, SimpleTestFixture, * utf::label("LayerTests"))

BOOST_DATA_TEST_CASE(FirstLayer_addCustomFeature_test, bdata::random(2, 10) ^ bdata::random(2, 10) ^ bdata::random(2, 10) ^ bdata::xrange(20), rows, cols, column, index)
{
    Matrix mat = Matrix::Random(rows, cols);
    int c = column % cols;
    FirstLayer layer(cols, true);
    layer.addCustomFeature(c, [](data_t v) { return 1.0/v; });
    Matrix ret = layer.eval(mat);
    BOOST_TEST(ret.rows() == cols + 2);
    for (int i = 0; i < ret.cols(); i++) {
        BOOST_TEST_REQUIRE(abs(ret(cols + 1, i) - 1/ret(c, i)) < EPS);
    }
}

BOOST_DATA_TEST_CASE(FirstLayer_addCrossFeature_test, bdata::random(2, 10) ^ bdata::random(2, 10) ^ bdata::xrange(20), rows, cols, index)
{
    Matrix mat = Matrix::Random(rows, cols);
    FirstLayer layer(cols, true);
    int a = random64(cols - 1);
    int b = random64(cols - 1);
    layer.addCrossFeature(std::vector<int> {a, b});
    Matrix ret = layer.eval(mat);
    BOOST_TEST(ret.rows() == cols + 2);
    for (int i = 0; i < ret.cols(); i++) {
        BOOST_TEST_REQUIRE(abs(ret(cols + 1, i) - ret(a, i) * ret(b, i)) < EPS);
    }
}

BOOST_DATA_TEST_CASE(FirstLayer_eval_test, bdata::random(2, 10) ^ bdata::random(2, 10) ^ bdata::xrange(20), rows, cols, index)
{
    Matrix mat = Matrix::Random(rows, cols);
    FirstLayer layer(cols, false);
    Matrix ret = layer.eval(mat);
    BOOST_TEST_REQUIRE(ret.rows() == cols);
    BOOST_TEST_REQUIRE(ret.cols() == rows);
    BOOST_TEST(abs((mat - ret.transpose()).norm()) < EPS);
}

BOOST_DATA_TEST_CASE(FirstLayer_eval_test2, bdata::random(2, 10) ^ bdata::random(2, 10) ^ bdata::xrange(20), rows, cols, index)
{
    Matrix mat = Matrix::Random(rows, cols);
    FirstLayer layer(cols, true);
    Matrix ret = layer.eval(mat);
    BOOST_TEST_REQUIRE(ret.rows() == cols + 1);
    BOOST_TEST_REQUIRE(ret.cols() == rows);
    for (int i = 0; i < ret.rows(); i++) {
        for (int j = 0; j < ret.cols(); j++) {
            if (i == ret.rows() - 1) {
                BOOST_TEST_REQUIRE(abs(ret(i, j) - 1) < EPS);
            } else {
                BOOST_TEST_REQUIRE(abs(ret(i, j) - mat(j, i)) < EPS);
            }
        }
    }    
}

BOOST_DATA_TEST_CASE(RegressionOutputLayer_eval_test, bdata::random(10, 50) ^ bdata::random(2, 10) ^ bdata::xrange(20), testcases, units, index)
{
    Matrix mat = Matrix::Random(units, testcases);
    RegressionOutputLayer layer(units);
    Matrix ret = layer.eval(mat);
    BOOST_TEST_REQUIRE(ret.rows() == testcases);
    BOOST_TEST_REQUIRE(ret.cols() == units);
    BOOST_TEST_REQUIRE(layer.numberOfInNodes() == units);
    BOOST_TEST_REQUIRE(layer.numberOfOutNodes() == units);
    
    BOOST_TEST_REQUIRE((ret - mat.transpose()).norm() < EPS);
}

BOOST_DATA_TEST_CASE(RegressionOutputLayer_delta_test, bdata::random(20, 30) ^ bdata::random(2, 10) ^ bdata::xrange(20), testcases, units, index)
{
    Matrix y = Matrix::Random(testcases, units) * 5;

    RegressionOutputLayer layer(units);
    Matrix z = Matrix::Random(units, testcases);
    Matrix x = layer.eval(z);
    data_t loss1 = layer.loss(x, y);
    BOOST_TEST_REQUIRE(loss1 > EPS);
    
    data_t learningRate = 1e-3;
    Matrix delta = layer.delta(x, y);
    for (int i = 0; i < z.rows(); i++) {
        for (int j = 0; j < z.cols(); j++) {
            z(i, j) -= delta(j, i) * learningRate;
            x = layer.eval(z);
            data_t loss2 = layer.loss(x, y);
//            BOOST_TEST_INFO(loss1 - loss2 << " " << learningRate * delta(j, i) * delta(j, i));
//            BOOST_TEST(loss2 < loss1 - EPS);
            BOOST_TEST_REQUIRE(abs((loss1 - loss2)/(learningRate * delta(j, i) * delta(j, i)) - 1.0) < 1e-3);
            z(i, j) += delta(j, i) * learningRate;
        }
    }
}

BOOST_DATA_TEST_CASE(RegressionOutputLayer_delta_test2, bdata::random(30, 40) ^ bdata::random(3, 10) ^ bdata::xrange(20), testcases, units, index)
{
    Matrix y = Matrix::Random(testcases, units) * 5;
    
    RegressionOutputLayer layer(units);
    Matrix z = Matrix::Random(units, testcases);
    Matrix x = layer.eval(z);
 
    data_t learningRate = 1e-1;
    int steps = 1000;
    std::vector<data_t> losses;
    losses.push_back(layer.loss(x, y));
    for (int i = 0; i < steps; i++) {
        Matrix delta = layer.delta(x, y);
        z -= learningRate * delta.transpose();
        x = layer.eval(z);
        losses.push_back(layer.loss(x, y));
        BOOST_TEST_REQUIRE(losses[i + 1] < losses[i]);
    }
}

BOOST_DATA_TEST_CASE(ClassificationOutputLayer_eval_test, bdata::random(2, 10) ^ bdata::xrange(20), testcases, index)
{
    Matrix mat = Matrix::Random(1, testcases);
    ClassificationOutputLayer layer;
    Matrix ret = layer.eval(mat);
    BOOST_TEST_REQUIRE(ret.rows() == testcases);
    BOOST_TEST_REQUIRE(ret.cols() == 1);
    BOOST_TEST_REQUIRE(layer.numberOfInNodes() == 1);
    BOOST_TEST_REQUIRE(layer.numberOfOutNodes() == 1);
    
    Sigmoid sd;
    for (int i = 0; i < ret.rows(); i++) {
        BOOST_TEST_REQUIRE(ret(i, 0) < 1);
        BOOST_TEST_REQUIRE(ret(i, 0) > 0);
        BOOST_TEST_REQUIRE(abs(ret(i, 0) - sd.eval(mat(0, i))) < EPS);
    }
}

BOOST_DATA_TEST_CASE(ClassificationOutputLayer_delta_test, bdata::random(50, 100) ^ bdata::xrange(20), testcases, index)
{
    Matrix y = Matrix::Zero(testcases, 1);
    
    for (int i = 0; i < testcases; i++) {
        y(i, 0) = random64(1);
    }
    
    ClassificationOutputLayer layer;    
    Matrix z = Matrix::Random(1, testcases);
    Matrix x = layer.eval(z);
    data_t loss1 = layer.loss(x, y);
    BOOST_TEST_REQUIRE(loss1 > EPS);
    
    data_t learningRate = 1e-3;
    Matrix delta = layer.delta(x, y);
    for (int i = 0; i < z.rows(); i++) {
        for (int j = 0; j < z.cols(); j++) {
            z(i, j) -= delta(j, i) * learningRate;
            x = layer.eval(z);
            data_t loss2 = layer.loss(x, y);
            BOOST_TEST(loss2 < loss1 - EPS);
            BOOST_TEST_REQUIRE(abs((loss1 - loss2)/(learningRate * delta(j, i) * delta(j, i)) - 1.0) < 1e-3);
            z(i, j) += delta(j, i) * learningRate;
        }
    }
}

BOOST_DATA_TEST_CASE(ClassificationOutputLayer_delta_test2, bdata::random(50, 100) ^ bdata::xrange(20), testcases, index)
{
    Matrix y = Matrix::Zero(testcases, 1);
    for (int i = 0; i < testcases; i++) {
        y(i, 0) = random64(1);
    }
    
    ClassificationOutputLayer layer;
    Matrix z = Matrix::Random(1, testcases);
    Matrix x = layer.eval(z);
 
    data_t learningRate = 1e-1;
    int steps = 1000;
    std::vector<data_t> losses;
    losses.push_back(layer.loss(x, y));
    for (int i = 0; i < steps; i++) {
        Matrix delta = layer.delta(x, y);
        z -= learningRate * delta.transpose();
        x = layer.eval(z);
        losses.push_back(layer.loss(x, y));
        BOOST_TEST_REQUIRE(losses[i + 1] < losses[i]);
    }
}

BOOST_DATA_TEST_CASE(SoftmaxOutputLayer_eval_test, bdata::random(2, 10) ^ bdata::random(2, 10) ^ bdata::xrange(20), testcases, units, index)
{
    Matrix mat = Matrix::Random(units, testcases);
    SoftmaxOutputLayer layer(units);
    Matrix ret = layer.eval(mat);
    BOOST_TEST_REQUIRE(ret.rows() == testcases);
    BOOST_TEST_REQUIRE(ret.cols() == units);
    BOOST_TEST_REQUIRE(layer.numberOfInNodes() == units);
    BOOST_TEST_REQUIRE(layer.numberOfOutNodes() == units);
    
    Matrix sums = ret.rowwise().sum();
    for (int i = 0; i < ret.rows(); i++) {
        BOOST_TEST_REQUIRE(abs(sums(i) - 1.0) < EPS);
        for (int j = 0; j < ret.cols(); j++) {
            for (int k = j + 1; k < ret.cols(); k++) {
                BOOST_TEST_REQUIRE(abs(ret(i, j) / ret(i, k) - exp(mat(j, i) - mat(k, i))) < EPS);
            }
        }
    }    
}

BOOST_DATA_TEST_CASE(SoftmaxOutputLayer_loss_test, bdata::random(2, 10) ^ bdata::random(2, 10) ^ bdata::xrange(20), testcases, units, index)
{
    Matrix x = Matrix::Zero(testcases, units);
    Matrix y = Matrix::Zero(testcases, units);
    
    for (int i = 0; i < testcases; i++) {
        int r = random64(units - 1);
        x(i, r) = y(i, r) = 1.0;
    }
    
    SoftmaxOutputLayer layer(units);
    data_t loss = layer.loss(x, y);
    BOOST_TEST_REQUIRE(loss < EPS);
    
    Matrix z = Matrix::Random(units, testcases);
    x = layer.eval(z);
    loss = layer.loss(x, y);
    BOOST_TEST_REQUIRE(loss > EPS);
}

BOOST_DATA_TEST_CASE(SoftmaxOutputLayer_delta_test, bdata::random(2, 10) ^ bdata::random(2, 10) ^ bdata::xrange(20), testcases, units, index)
{
    Matrix y = Matrix::Zero(testcases, units);
    
    for (int i = 0; i < testcases; i++) {
        int r = random64(units - 1);
        y(i, r) = 1.0;
    }
    
    SoftmaxOutputLayer layer(units);    
    Matrix z = Matrix::Random(units, testcases);
    Matrix x = layer.eval(z);
    data_t loss1 = layer.loss(x, y);
    BOOST_TEST_REQUIRE(loss1 > EPS);
    
    data_t learningRate = 1e-3;
    Matrix delta = layer.delta(x, y);
    for (int i = 0; i < z.rows(); i++) {
        for (int j = 0; j < z.cols(); j++) {
            z(i, j) -= delta(j, i) * learningRate;
            x = layer.eval(z);
            data_t loss2 = layer.loss(x, y);
            BOOST_TEST(loss2 < loss1 - EPS);
            BOOST_TEST_REQUIRE(abs((loss1 - loss2)/(learningRate * delta(j, i) * delta(j, i)) - 1.0) < 1e-3);
            z(i, j) += delta(j, i) * learningRate;
        }
    }
}

BOOST_DATA_TEST_CASE(SoftmaxOutputLayer_delta_test2, bdata::random(50, 100) ^ bdata::random(3, 10) ^ bdata::xrange(20), testcases, units, index)
{
    Matrix y = Matrix::Zero(testcases, units);
    for (int i = 0; i < testcases; i++) {
        int r = random64(units - 1);
        y(i, r) = 1.0;
    }
    
    SoftmaxOutputLayer layer(units);
    Matrix z = Matrix::Random(units, testcases);
    Matrix x = layer.eval(z);
 
    data_t learningRate = 1e-1;
    int steps = 1000;
    std::vector<data_t> losses;
    losses.push_back(layer.loss(x, y));
    for (int i = 0; i < steps; i++) {
        Matrix delta = layer.delta(x, y);
        z -= learningRate * delta.transpose();
        x = layer.eval(z);
        losses.push_back(layer.loss(x, y));
        BOOST_TEST_REQUIRE(losses[i + 1] < losses[i]);
    }
}

BOOST_AUTO_TEST_SUITE_END()

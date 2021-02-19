//Link to Boost
#define BOOST_TEST_DYN_LINK

//Define our Module name (prints at testing)
#define BOOST_TEST_MODULE DPL UnitTests

//VERY IMPORTANT - include this last
#include <boost/test/included/unit_test.hpp>
//#include <boost/test/unit_test.hpp>

#include "test.h"
#include "model.h"
#include "datautil.h"

// test suite
BOOST_FIXTURE_TEST_SUITE(Demo_suite, SimpleTestFixture, * utf::label("UnityStartSystem"))

BOOST_AUTO_TEST_CASE(demo_random_input)
{
    std::cout << "runnng demo_random_input." << std::endl;
    Model model;
    size_t dataSize = 100;
    size_t featureSize = 5;
    size_t hiddenLayerSize = 3;
    model.addLayer(new FirstLayer(featureSize));

    for (int i = 0; i < hiddenLayerSize; i++) {
        model.addLayer(new SimpleHiddenLayer("relu", featureSize));
//        model.addLayer(new SimpleHiddenLayer("", featureSize));
    }
    model.addLayer(new RegressionOutputLayer());
    model.prepare();
    
    Matrix data = Matrix::Random(dataSize, featureSize);
    Matrix y = Matrix::Random(dataSize, 1);
    
//    Vector loss = model.train(data, y, epic, learningRate, lambda);
    HyperParameter params = {
        .epochs = 100,
        .batch = -1, 
        .learningRate = 0.01,
        .lambda = 0.001,
    };

    Vector loss = model.train(data, y, params);
    std::cout << "training loss=" << loss.transpose() << std::endl;
        
    RVector input = RVector::Random(featureSize);
    std::cout << "input=" << input << std::endl;
    std::cout << "output=" << model.eval(input) << std::endl;
    std::cout << "demo end." << std::endl;
}

BOOST_AUTO_TEST_CASE(demo_line)
{
    std::cout << "runnng demo_line." << std::endl;
    Model model;
    size_t dataSize = 10000;
    size_t featureSize = 2;
    size_t hiddenLayerSize = 0;
    std::string regularization = "L1";
    model.addLayer(new FirstLayer(featureSize));

    for (int i = 0; i < hiddenLayerSize; i++) {
        model.addLayer(new SimpleHiddenLayer("relu", featureSize + 1));
    }
    model.addLayer(new ClassificationOutputLayer());
    model.prepare(regularization);
    
    Matrix data = Matrix::Random(dataSize, featureSize);
    Matrix y(dataSize, 1);
    for (int i = 0; i < data.rows(); i++) {
        if (data(i, 0) > data(i, 1)) {
            y(i) = 1.0;
        } else {
            y(i) = 0.0;
        }
    }
    
    HyperParameter params = {
        .epochs = 3000,
        .batch = 100, 
        .learningRate = 0.05,
        .lambda = 0.01,
    };
    
    Stopwatch watch;
    Vector loss = model.train(data, y, params);
    double sec = watch.Elapsed();
    std::cout << "training loss=" << loss.transpose() << std::endl;
    for (int i = 0; i < model.weights().size(); i++) {
        std::cout << "weight " << i << std::endl;
        std::cout << model.weights()[i] << std::endl;
    }
    
    int testcases = 20;
    Matrix input = Matrix::Random(testcases, 2);
    Matrix output = model.eval(input);

    for (int i = 0; i < testcases; i++) {
        std::cout << "test " << i << std::endl;
        std::cout << "input=" << input.row(i) << " output=" << output.row(i);
        if ((output(i, 0) > 0.5) == (input(i, 0) > input(i, 1))) {
            std::cout << " passed" << std::endl;
        } else {
            std::cout << " failed" << std::endl;
        }
    }
    
    std::cout << "takes " << sec << " seconds to train the model." << std::endl;
}

BOOST_AUTO_TEST_CASE(demo_circle)
{
    std::cout << "runnng demo_circle." << std::endl;
    size_t dataSize = 10000;
    size_t featureSize = 2;
    size_t hiddenLayerSize = 0;
        
    Matrix data = Matrix::Random(dataSize, featureSize);
    Matrix y(dataSize, 1);
    data_t r = 0.8;
    for (int i = 0; i < data.rows(); i++) {
        if (data.row(i).norm() < r) {
            y(i) = 1.0;
        } else {
            y(i) = 0.0;
        }
    }
    
    std::cout << "y.size()=" << y.size() << ", y.sum()=" << y.sum() << std::endl;
    
    Model model;
    FirstLayer* firstLayer = new FirstLayer(2);
    firstLayer->addCrossFeature(std::vector<int>{0, 0});
    firstLayer->addCrossFeature(std::vector<int>{1, 1});
    model.addLayer(firstLayer);

    for (int i = 0; i < hiddenLayerSize; i++) {
        model.addLayer(new SimpleHiddenLayer("sigmoid", featureSize + 1));
    }
    model.addLayer(new ClassificationOutputLayer());
    model.prepare();

    HyperParameter params = {
        .epochs = 2000,
        .batch = 100, 
        .learningRate = 0.03,
        .lambda = 0.001,
    };

    Vector loss = model.train(data, y, params);    
    std::cout << "training loss=" << loss.transpose() << std::endl;
    
    int testcases = 20;
    Matrix input = Matrix::Random(testcases, 2);
    Matrix output = model.eval(input);
    
    for (int i = 0; i < testcases; i++) {
        std::cout << "test " << i << std::endl; 
        std::cout << "input=" << input.row(i) << " output=" << output.row(i);
        if ((output(i, 0) > 0.5) == (input.row(i).norm() < r)) {
            std::cout << " passed" << std::endl;
        } else {
            std::cout << " failed" << std::endl;
        }
    }
}

RVector quadrant(data_t x, data_t y) {
    RVector ret(4);
    if (x > 0) {
        if (y > 0) {
            ret << 1, 0, 0, 0;
        } else {
            ret << 0, 0, 0, 1;
        }
    } else {
        if (y > 0) {
            ret << 0, 1, 0, 0;
        } else {
            ret << 0, 0, 1, 0;
        }
    }
    return ret;
}

BOOST_AUTO_TEST_CASE(demo_softmax)
{
    std::cout << "runnng demo_softmax." << std::endl;
    Model model;
    size_t dataSize = 1000;
    size_t featureSize = 2;
    size_t hiddenLayerSize = 0;
    std::string regularization = "L1";
    
    FirstLayer* firstLayer = new FirstLayer(featureSize);
    //firstLayer->addCrossFeature(std::vector<int>{0, 1});
/*    firstLayer->addCustomFeature(0, [](data_t v) {
        return 1.0/v; 
    });
    firstLayer->addCustomFeature(1, [](data_t v) {
        return 1.0/v; 
    });*/
    model.addLayer(firstLayer);

    for (int i = 0; i < hiddenLayerSize; i++) {
        model.addLayer(new SimpleHiddenLayer("sigmoid", 3));
    }
    model.addLayer(new SoftmaxOutputLayer(4));
    model.prepare(regularization);
    
    Matrix data = Matrix::Random(dataSize, featureSize);
    Matrix y(dataSize, 4);
    for (int i = 0; i < data.rows(); i++) {
        y.row(i) = quadrant(data(i, 0), data(i, 1));
    }
    
//    std::cout << "data=" << std::endl << data << std::endl << std::endl;
//    std::cout << "y=" << std::endl << y << std::endl << std::endl;
    
    HyperParameter params = {
        .epochs = 500,
        .batch = 100, 
        .learningRate = 0.05,
        .lambda = 0.001,
    };
    
    Stopwatch watch;
    Vector loss = model.train(data, y, params);
    double sec = watch.Elapsed();
    std::cout << "training loss=" << loss.transpose() << std::endl;
    for (int i = 0; i < model.weights().size(); i++) {
        std::cout << "weight " << i << std::endl;
        std::cout << model.weights()[i] << std::endl;
    }
    
    Matrix input = Matrix::Random(20, 2);
    Matrix expected(20, 4);
    for (int i = 0; i < input.rows(); i++) {
        expected.row(i) = quadrant(input(i, 0), input(i, 1));
    }
    
    Matrix res = model.eval(input);

    for (int i = 0; i < 20; i++) {
        std::cout << "test " << i << std::endl;
        std::cout << "res = " << res.row(i) << ", expected=" << expected.row(i) << std::endl;
    }
    
    std::cout << "takes " << sec << " seconds to train the model." << std::endl;
}

BOOST_AUTO_TEST_CASE(demo_softmax2)
{
    std::cout << "runnng demo_softmax2." << std::endl;
    Model model;
    size_t dataSize = 100;
    size_t featureSize = 2;
    size_t hiddenLayerSize = 0;
    std::string regularization = "L2";
    
    FirstLayer* firstLayer = new FirstLayer(featureSize);
    model.addLayer(firstLayer);

    for (int i = 0; i < hiddenLayerSize; i++) {
        model.addLayer(new SimpleHiddenLayer("relu", 2));
    }
    model.addLayer(new SoftmaxOutputLayer(2));
    model.prepare(regularization);
    
    Matrix data = Matrix::Random(dataSize, featureSize);
    Matrix y(dataSize, 2);
    for (int i = 0; i < data.rows(); i++) {
        if (data(i, 0) > data(i, 1)) {
            y(i, 0) = 1;
            y(i, 1) = 0;
        } else {
            y(i, 1) = 1;
            y(i, 0) = 0;
        }
    }
    
//    std::cout << "data=" << std::endl << data << std::endl << std::endl;
//    std::cout << "y=" << std::endl << y << std::endl << std::endl;
    
    HyperParameter params = {
        .epochs = 5000,
        .batch = 100, 
        .learningRate = 0.005,
        .lambda = 0.001,
    };
    
    Stopwatch watch;
    Vector loss = model.train(data, y, params);
    double sec = watch.Elapsed();
    std::cout << "training loss=" << loss.transpose() << std::endl;
    for (int i = 0; i < model.weights().size(); i++) {
        std::cout << "weight " << i << std::endl;
        std::cout << model.weights()[i] << std::endl;
    }
    
    Matrix input = Matrix::Random(20, 2);
    Matrix expected(20, 2);
    for (int i = 0; i < input.rows(); i++) {
        if (input(i, 0) > input(i, 1)) {
            expected(i, 0) = 1;
            expected(i, 1) = 0;
        } else {
            expected(i, 1) = 1;
            expected(i, 0) = 0;
        }
    }
    
    Matrix res = model.eval(input);

    for (int i = 0; i < 20; i++) {
        std::cout << "test " << i << std::endl;
        std::cout << "res = " << res.row(i) << ", expected=" << expected.row(i) << std::endl;
    }
    
    std::cout << "takes " << sec << " seconds to train the model." << std::endl;
}



BOOST_AUTO_TEST_SUITE_END()

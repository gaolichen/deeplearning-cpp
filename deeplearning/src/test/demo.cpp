//Link to Boost
#define BOOST_TEST_DYN_LINK

//Define our Module name (prints at testing)
#define BOOST_TEST_MODULE BetheSolver UnitTests

//VERY IMPORTANT - include this last
#include <boost/test/included/unit_test.hpp>
//#include <boost/test/unit_test.hpp>

#include "test.h"
#include "model.h"

// test suite
BOOST_FIXTURE_TEST_SUITE(Demo_suite, SimpleTestFixture, * utf::label("UnityStartSystem"))

BOOST_AUTO_TEST_CASE(Demo)
{
    std::cout << "runnng demo." << std::endl;
    Model model;
    size_t dataSize = 100;
    size_t featureSize = 5;
    size_t hiddenLayerSize = 3;
    int epic = 100;
    data_t learningRate = 0.01;
    model.addLayer(new FirstLayer(featureSize));

    for (int i = 0; i < hiddenLayerSize; i++) {
        model.addLayer(new SimpleHiddenLayer("relu", featureSize));
//        model.addLayer(new SimpleHiddenLayer("", featureSize));
    }
    model.addLayer(new RegressionOutputLayer());
    model.prepare();
    
    Matrix data = Matrix::Random(dataSize, featureSize);
    Matrix y = Matrix::Random(dataSize, 1);
    
    Vector loss = model.train(data, y, epic, learningRate);
    std::cout << "training loss=" << loss.transpose() << std::endl;
        
    RVector input = RVector::Random(featureSize);
    std::cout << "input=" << input << std::endl;
    std::cout << "output=" << model.eval(input) << std::endl;
    std::cout << "demo end." << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()

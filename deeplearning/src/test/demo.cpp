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
    size_t featureSize = 10;
    size_t layerSize = 3;
    int batchSize = 10;
    int epic = 20;
    for (int i = 0; i < layerSize; i++) {
        Layer* layer = new Layer();
        for (int j = 0; j < featureSize; j++) {
            if (i == 0) {
                layer->addNode(new FirstLayerNode());
            } else {
                layer->addNode(new HiddenNode("lula"));
            }
        }
        model.addLayer(layer);
    }
    model.prepare();
    
    Matrix data = Matrix::Random(dataSize, featureSize);
    Vector output = Vector::Random(dataSize);
    
    Vector loss = model.train(data, output, batchSize, epic);
    Vector input = Vector::Random(featureSize);
    std::cout << "input=" << input << std::endl;
    std::cout << "loss=" << loss << std::endl;
    std::cout << model.eval(input) << std::endl;
    std::cout << "demo end." << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()

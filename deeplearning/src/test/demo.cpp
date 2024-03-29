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
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

// test suite
BOOST_FIXTURE_TEST_SUITE(Demo_suite, SimpleTestFixture, * utf::disabled())

BOOST_AUTO_TEST_CASE(demo_random_input)
{
    std::cout << "runnng demo_random_input." << std::endl;
    Model model;
    size_t dataSize = 100;
    size_t featureSize = 5;
    size_t hiddenLayerSize = 3;
    model.addLayer(new FirstLayer({0, 1, 2, 3, 4}));

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
        .epochs = 5,
        .batchSize = 32, 
        .learningRate = 0.01,
        .lambda = 0.001,
    };

    model.train(data, y, params);
    
    RVector input = RVector::Random(featureSize);
    std::cout << "predict input=" << input << std::endl;
    std::cout << "predict output=" << model.predict(input) << std::endl;
}

BOOST_AUTO_TEST_CASE(demo_line)
{
    std::cout << "runnng demo_line." << std::endl;
    Model model;
    size_t dataSize = 10000;
    size_t featureSize = 2;
    size_t hiddenLayerSize = 0;
    std::string regularization = "L1";
    model.addLayer(new FirstLayer({0, 1}));

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
        .epochs = 100,
        .batchSize = 100, 
        .learningRate = 0.05,
        .lambda = 0.01,
        .validation_split = 0.1,
    };
    
    Stopwatch watch;
    model.train(data, y, params);
    
    double sec = watch.Elapsed();
    
    int testcases = 20;
    int failure = 0;
    Matrix input = Matrix::Random(testcases, 2);
    Matrix output = model.predict(input);

    for (int i = 0; i < testcases; i++) {
        if ((output(i, 0) > 0.5) == (input(i, 0) > input(i, 1))) {
            // do nothing for passed.
        } else {
            std::cout << "test " << i << std::endl;
            std::cout << "input=" << input.row(i) << " output=" << output.row(i);
            std::cout << " failed" << std::endl;
        }
    }
    
    std::cout << failure << " test cases failed." << std::endl;    
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
        
    Model model;
    FirstLayer* firstLayer = new FirstLayer();
    firstLayer->addFeatureColumn(new NumericCrossColumn({0, 0}));
    firstLayer->addFeatureColumn(new NumericCrossColumn({1, 1}));
    model.addLayer(firstLayer);
    
    for (int i = 0; i < hiddenLayerSize; i++) {
        model.addLayer(new SimpleHiddenLayer("sigmoid", featureSize + 1));
    }
    model.addLayer(new ClassificationOutputLayer());
    model.prepare();

    HyperParameter params = {
        .epochs = 20,
        .batchSize = 100, 
        .learningRate = 0.03,
        .lambda = 0.001,
        .validation_split = 0.1,
    };

    model.train(data, y, params);
    
    int testcases = 20;
    int failures = 0;
    Matrix input = Matrix::Random(testcases, 2);
    Matrix output = model.predict(input);
    
    for (int i = 0; i < testcases; i++) {
        if ((output(i, 0) > 0.5) == (input.row(i).norm() < r)) {
            // do nothing
        } else {
            std::cout << "test " << i << std::endl; 
            std::cout << "input=" << input.row(i) << " output=" << output.row(i);
            std::cout << " failed" << std::endl;
            failures ++;
        }
    }
    std::cout << failures << " out of " << testcases << " test cases failed." << std::endl;
}

int quadrant(data_t x, data_t y) {
//    RVector ret(4);
    if (x > 0) {
        if (y > 0) {
//            ret << 1, 0, 0, 0;
            return 0;
        } else {
//            ret << 0, 0, 0, 1;
            return 3;
        }
    } else {
        if (y > 0) {
//            ret << 0, 1, 0, 0;
            return 1;
        } else {
//            ret << 0, 0, 1, 0;
            return 2;
        }
    }
//    return ret;
}

BOOST_AUTO_TEST_CASE(demo_softmax)
{
    std::cout << "runnng demo_softmax." << std::endl;
    Model model;
    size_t dataSize = 500;
    size_t featureSize = 2;
    size_t hiddenLayerSize = 0;
    std::string regularization = "L1";    
    model.addLayer(new FirstLayer({0, 1}));

    for (int i = 0; i < hiddenLayerSize; i++) {
        model.addLayer(new SimpleHiddenLayer("sigmoid", 3));
    }
    model.addLayer(new SoftmaxOutputLayer(4));
    model.prepare(regularization);
    
    Matrix data = Matrix::Random(dataSize, featureSize);
//    Matrix y(dataSize, 4);
    Vector y(dataSize);
    for (int i = 0; i < data.rows(); i++) {
//        y.row(i) = quadrant(data(i, 0), data(i, 1));
        y(i) = quadrant(data(i, 0), data(i, 1));
    }
        
    HyperParameter params = {
        .epochs = 100,
        .batchSize = 32, 
        .learningRate = 0.05,
        .lambda = 0.001,
        .validation_split = 0.2,
    };
    
    Stopwatch watch;
    model.train(data, y, params);
    double sec = watch.Elapsed();
        
    model.plotLoss();
    
    Matrix input = Matrix::Random(20, 2);
    Vector expected(20);
    for (int i = 0; i < input.rows(); i++) {
        expected(i) = quadrant(input(i, 0), input(i, 1));
    }
    
    data_t accuracy = model.evaluate(input, expected);
    std::cout << "test loss = " << accuracy << std::endl;
    std::cout << "takes " << sec << " seconds to train the model." << std::endl;
}

BOOST_AUTO_TEST_CASE(demo_softmax2)
{
    std::cout << "runnng demo_softmax2." << std::endl;
    Model model;
    size_t dataSize = 1000;
    size_t featureSize = 2;
    size_t hiddenLayerSize = 0;
    std::string regularization = "L2";
    
    model.addLayer(new FirstLayer({0, 1}));

    for (int i = 0; i < hiddenLayerSize; i++) {
        model.addLayer(new SimpleHiddenLayer("relu", 2));
    }
    model.addLayer(new SoftmaxOutputLayer(2));
    model.prepare(regularization);
    
    Matrix data = Matrix::Random(dataSize, featureSize);
    Vector y(dataSize);
    for (int i = 0; i < data.rows(); i++) {
        if (data(i, 0) > data(i, 1)) {
            y(i) = 0;
        } else {
            y(i) = 1;
        }
    }
        
    HyperParameter params = {
        .epochs = 50,
        .batchSize = 32, 
        .learningRate = 0.005,
        .lambda = 0.001,
    };
    
    Stopwatch watch;
    model.train(data, y, params);
    double sec = watch.Elapsed();
    Matrix input = Matrix::Random(20, 2);
    Vector expected(20);
    for (int i = 0; i < input.rows(); i++) {
        if (input(i, 0) > input(i, 1)) {
            expected(i) = 0;
        } else {
            expected(i) = 1;
        }
    }
    
    Matrix res = model.predict(input);
    int failures = 0;

    for (int i = 0; i < 20; i++) {
        if ((res(i, 0) >= 0.5) == (expected(i) < 1)) {
            // succeed
            continue;
        }
        
        std::cout << "test " << i << std::endl;
        std::cout << "res = " << res.row(i) << ", expected=" << expected(i) << std::endl;
        failures++;
    }
    std::cout << failures << "/20 test cases failed." << std::endl; 
    std::cout << "takes " << sec << " seconds to train the model." << std::endl;
}

BOOST_AUTO_TEST_CASE(ca_house)
{
    CSVData csvTrain, csvTest;
    csvTrain.read("./testdata/california_housing_train.csv", true);
    csvTest.read("./testdata/california_housing_test.csv", true);
    
    std::cout << csvTrain.headers() << std::endl;
    std::cout << csvTest.headers() << std::endl;
    
    Matrix trainData = DataUtil::randomRowShuffle(csvTrain.data());
    Matrix testData = DataUtil::randomRowShuffle(csvTest.data());
    
    int featureCol = csvTrain.headerIndex("\"median_income\"");
    int labelCol = csvTrain.headerIndex("\"median_house_value\"");
    
    std::cout << "featureCol=" << featureCol << ", labelCol=" << labelCol << std::endl;
    
    data_t scaleFactor = 1000.0;
    trainData.col(labelCol) /= scaleFactor;
    testData.col(labelCol) /= scaleFactor;
    
    Model model;
    std::string regularization = "L2";
    
    model.addLayer(new FirstLayer({featureCol}, true));

    int hiddenLayerSize = 0;
    for (int i = 0; i < hiddenLayerSize; i++) {
        model.addLayer(new SimpleHiddenLayer("relu", 2));
    }
    model.addLayer(new RegressionOutputLayer(1));
    model.prepare(regularization);
    
    HyperParameter params = {
        .epochs = 70,
        .batchSize = 32, 
        .learningRate = 0.08    ,
        .lambda = 0.0,
        .validation_split = 0.2,
    };
    
    model.train(trainData, trainData.col(labelCol), params);
    std::cout << "test data loss rms= " << sqrt(model.evaluate(testData, testData.col(labelCol))) << std::endl;
    model.plotLoss(true);
}

// test discrete cross column
BOOST_AUTO_TEST_CASE(ca_house2)
{
    CSVData csvTrain, csvTest;
    csvTrain.read("./testdata/california_housing_train.csv", true);
    csvTest.read("./testdata/california_housing_test.csv", true);
    
    Matrix trainData = DataUtil::randomRowShuffle(csvTrain.data());
    Matrix testData = DataUtil::randomRowShuffle(csvTest.data());
    
    int featureCol = csvTrain.headerIndex("\"median_income\"");
    int labelCol = csvTrain.headerIndex("\"median_house_value\"");
    int latitudeCol = csvTrain.headerIndex("\"latitude\"");
    int longitudeCol = csvTrain.headerIndex("\"longitude\"");
        
    data_t scaleFactor = 1000.0;
    trainData.col(labelCol) /= scaleFactor;
    testData.col(labelCol) /= scaleFactor;
    Model model;
    std::string regularization = "L2";
    
    FirstLayer* firstLayer = new FirstLayer();        
    firstLayer->addFeatureColumn(new SimpleNumericColumn(featureCol));
    
    data_t resolution = .4;
    firstLayer->addFeatureColumn(
        new DiscreteCrossColumn({
                new BucketedColumn(latitudeCol,
                trainData.col(latitudeCol).minCoeff(),
                trainData.col(latitudeCol).maxCoeff(), 
                resolution), 
                new BucketedColumn(longitudeCol,
                trainData.col(longitudeCol).minCoeff(),
                trainData.col(longitudeCol).maxCoeff(), 
                resolution)
            }
        )
    );
    
    model.addLayer(firstLayer);

    int hiddenLayerSize = 0;
    for (int i = 0; i < hiddenLayerSize; i++) {
        model.addLayer(new SimpleHiddenLayer("relu", 2));
    }
    model.addLayer(new RegressionOutputLayer(1));
    model.prepare(regularization);
    
    HyperParameter params = {
        .epochs = 35,
        .batchSize = 100, 
        .learningRate = 0.04    ,
        .lambda = 0.001,
        .validation_split = 0.2,
    };
    
    model.train(trainData, trainData.col(labelCol), params);
    std::cout << "test data loss rms= " << sqrt(model.evaluate(testData, testData.col(labelCol))) << std::endl;
    model.plotLoss(true);
}

BOOST_AUTO_TEST_CASE(ca_house3)
{
    CSVData csvTrain, csvTest;
    csvTrain.read("./testdata/california_housing_train.csv", true);
    csvTest.read("./testdata/california_housing_test.csv", true);
    
    Matrix trainData = DataUtil::randomRowShuffle(csvTrain.data());
    Matrix testData = DataUtil::randomRowShuffle(csvTest.data());
    
    trainData = DataUtil::zScoreNormalize(trainData);
    testData = DataUtil::zScoreNormalize(testData);
    int labelCol = csvTrain.headerIndex("\"median_house_value\"");
    int medianIncomeCol = csvTrain.headerIndex("\"median_income\"");
    int latitudeCol = csvTrain.headerIndex("\"latitude\"");
    int longitudeCol = csvTrain.headerIndex("\"longitude\"");
    int polulationCol = csvTrain.headerIndex("\"population\"");
    
    Model model;
    std::string regularization = "L2";
    
    FirstLayer* firstLayer = new FirstLayer({medianIncomeCol, polulationCol});
    
    std::cout <<"latitude range=" << std::vector<data_t>{trainData.col(latitudeCol).minCoeff(),
                trainData.col(latitudeCol).maxCoeff()} << std::endl;
    std::cout <<"longitude range=" << std::vector<data_t>{trainData.col(longitudeCol).minCoeff(),
                trainData.col(longitudeCol).maxCoeff()} << std::endl;
    data_t resolution = .3;
    firstLayer->addFeatureColumn(
        new DiscreteCrossColumn({
                new BucketedColumn(latitudeCol,
                trainData.col(latitudeCol).minCoeff(),
                trainData.col(latitudeCol).maxCoeff(), 
                resolution), 
                new BucketedColumn(longitudeCol,
                trainData.col(longitudeCol).minCoeff(),
                trainData.col(longitudeCol).maxCoeff(), 
                resolution)
            }
        )
    );
    
    model.addLayer(firstLayer);

    int hiddenLayerSize = 0;
    for (int i = 0; i < hiddenLayerSize; i++) {
        model.addLayer(new SimpleHiddenLayer("relu", 2));
    }
    model.addLayer(new RegressionOutputLayer(1));
    model.prepare(regularization);
    
    HyperParameter params = {
        .epochs = 100,
        .batchSize = 1000, 
        .learningRate = 0.05,
        .lambda = 0.00,
        .validation_split = 0.2,
    };
    
    model.train(trainData, trainData.col(labelCol), params);
    data_t testLoss = model.evaluate(testData, testData.col(labelCol));
    std::cout << "test data loss= " << testLoss << std::endl;
    model.plotLoss(false);
}

BOOST_AUTO_TEST_CASE(ca_house4)
{
    CSVData csvTrain, csvTest;
    csvTrain.read("./testdata/california_housing_train.csv", true);
    csvTest.read("./testdata/california_housing_test.csv", true);
    
    Matrix trainData = DataUtil::randomRowShuffle(csvTrain.data());
    Matrix testData = DataUtil::randomRowShuffle(csvTest.data());
    
    trainData = DataUtil::zScoreNormalize(trainData);
    testData = DataUtil::zScoreNormalize(testData);
    int labelCol = csvTrain.headerIndex("\"median_house_value\"");
    int medianIncomeCol = csvTrain.headerIndex("\"median_income\"");
    int latitudeCol = csvTrain.headerIndex("\"latitude\"");
    int longitudeCol = csvTrain.headerIndex("\"longitude\"");
    int polulationCol = csvTrain.headerIndex("\"population\"");
    
    Model model;
    std::string regularization = "L2";
    
    FirstLayer* firstLayer = new FirstLayer({medianIncomeCol, polulationCol});
    
    std::cout <<"latitude range=" << std::vector<data_t>{trainData.col(latitudeCol).minCoeff(),
                trainData.col(latitudeCol).maxCoeff()} << std::endl;
    std::cout <<"longitude range=" << std::vector<data_t>{trainData.col(longitudeCol).minCoeff(),
                trainData.col(longitudeCol).maxCoeff()} << std::endl;
    data_t resolution = .3;
    firstLayer->addFeatureColumn(
        new DiscreteCrossColumn({
                new BucketedColumn(latitudeCol,
                trainData.col(latitudeCol).minCoeff(),
                trainData.col(latitudeCol).maxCoeff(), 
                resolution), 
                new BucketedColumn(longitudeCol,
                trainData.col(longitudeCol).minCoeff(),
                trainData.col(longitudeCol).maxCoeff(), 
                resolution)
            }
        )
    );
    
    model.addLayer(firstLayer);
    model.addLayer(new SimpleHiddenLayer("relu", 20));
    model.addLayer(new SimpleHiddenLayer("relu", 12));
    model.addLayer(new RegressionOutputLayer(1));
    model.prepare(regularization);
    
    HyperParameter params = {
        .epochs = 100,
        .batchSize = 100,
        .learningRate = 0.05,
        .lambda = 0.00,
        .validation_split = 0.2,
    };
    
    model.train(trainData, trainData.col(labelCol), params);
    data_t testLoss = model.evaluate(testData, testData.col(labelCol));
    std::cout << "test data loss= " << testLoss << std::endl;
    model.plotLoss(false);
}

BOOST_AUTO_TEST_CASE(ca_house5)
{
    CSVData csvTrain, csvTest;
    csvTrain.read("./testdata/california_housing_train.csv", true);
    csvTest.read("./testdata/california_housing_test.csv", true);
    
    Matrix trainData = DataUtil::randomRowShuffle(csvTrain.data());
    Matrix testData = DataUtil::randomRowShuffle(csvTest.data());
    trainData = DataUtil::zScoreNormalize(trainData);
    
    testData = DataUtil::zScoreNormalize(testData);
    int labelCol = csvTrain.headerIndex("\"median_house_value\"");
    int medianIncomeCol = csvTrain.headerIndex("\"median_income\"");
    int latitudeCol = csvTrain.headerIndex("\"latitude\"");
    int longitudeCol = csvTrain.headerIndex("\"longitude\"");
    int polulationCol = csvTrain.headerIndex("\"population\"");
    
    Model model;
    std::string regularization = "L2";
    
    FirstLayer* firstLayer = new FirstLayer({medianIncomeCol, polulationCol});
    
    std::cout <<"latitude range=" << std::vector<data_t>{trainData.col(latitudeCol).minCoeff(),
                trainData.col(latitudeCol).maxCoeff()} << std::endl;
    std::cout <<"longitude range=" << std::vector<data_t>{trainData.col(longitudeCol).minCoeff(),
                trainData.col(longitudeCol).maxCoeff()} << std::endl;
    data_t resolution = .3;
    firstLayer->addFeatureColumn(
        new DiscreteCrossColumn({
                new BucketedColumn(latitudeCol,
                trainData.col(latitudeCol).minCoeff(),
                trainData.col(latitudeCol).maxCoeff(), 
                resolution), 
                new BucketedColumn(longitudeCol,
                trainData.col(longitudeCol).minCoeff(),
                trainData.col(longitudeCol).maxCoeff(), 
                resolution)
            }
        )
    );
    
    model.addLayer(firstLayer);
    model.addLayer(new SimpleHiddenLayer("relu", 20));
    model.addLayer(new DropoutLayer(0.2));
    model.addLayer(new SimpleHiddenLayer("relu", 12));
    model.addLayer(new RegressionOutputLayer(1));
    model.prepare(regularization);
    
    HyperParameter params = {
        .epochs = 50,
        .batchSize = 4000, 
        .learningRate = 0.028,
        .lambda = 0.01,
        .validation_split = 0.2,
    };
    
    model.train(trainData, trainData.col(labelCol), params);
    data_t testLoss = model.evaluate(testData, testData.col(labelCol));
    std::cout << "test data loss= " << testLoss << std::endl;
    model.plotLoss(false);
}

BOOST_AUTO_TEST_CASE(demo_MNIST)
{
    //Eigen::setNbThreads(4);
    std::cout << "number of threads: " << Eigen::nbThreads() << std::endl;
    Matrix xTrain, xTest;
    Vector yTrain, yTest;
    std::string xTrainPath = "./testdata/train-images-idx3-ubyte";
    std::string xTestPath = "./testdata/t10k-images-idx3-ubyte";
    std::string yTrainPath = "./testdata/train-labels-idx1-ubyte";
    std::string yTestPath = "./testdata/t10k-labels-idx1-ubyte";
    
    DataUtil::readMNISTimage(xTrainPath, xTrain, true);
    DataUtil::readMNISTimage(xTestPath, xTest, true);
    DataUtil::readMNISTlabel(yTrainPath, yTrain);
    DataUtil::readMNISTlabel(yTestPath, yTest);
    
    Model model;
    std::string regularization = "L2";
    std::vector<int> featureCols;
    for (int i = 0; i < 28 * 28; i++) {
        featureCols.push_back(i);
    }
    FirstLayer* firstLayer = new FirstLayer(featureCols);
        
    model.addLayer(firstLayer);
    model.addLayer(new SimpleHiddenLayer("relu", 32));
    //model.addLayer(new DropoutLayer(0.2));
    model.addLayer(new SoftmaxOutputLayer(10));
    
    model.prepare(regularization);
    
    HyperParameter params = {
        .epochs = 20,
        .batchSize = 256, 
        .learningRate = 0.012,
        .lambda = 0.01,
        .validation_split = 0.2,
    };
    
    Stopwatch watch;
    model.train(xTrain, yTrain, params);
    data_t secs = watch.Elapsed();
    data_t testLoss = model.evaluate(xTest, yTest);
    std::cout << "test data loss= " << testLoss << std::endl;
    std::cout << "takes " << secs << " seconds to train." << std::endl;
//    model.plotLoss(false);
}

BOOST_AUTO_TEST_SUITE_END()

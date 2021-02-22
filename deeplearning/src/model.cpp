#include "model.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#define SUBSET(mat, idx) indexing((mat), IndexType(&(idx)[0], (idx).size()))

std::ostream& operator<< (std::ostream& out, const HyperParameter& params) {
  out << "{ epochs=" << params.epochs << ", batch=" << params.batch;
  out << ", learningRate=" << params.learningRate << ", lambda=" << params.lambda;
  out << ", validation_split=" << params.validation_split << " }";  
  return out;
}


Model::Model() {
}

Model::~Model() {
    if (_regular != NULL) {
        delete _regular;
    }
    
    if (_propagator != NULL) {
        delete _propagator;
    }
    
    for (size_t i = 0; i < _layers.size(); i++) {
        delete _layers[i];
    }
    _layers.clear();
}

void Model::addLayer(const Layer* layer) {
    _layers.push_back(layer);
}

void Model::prepare(std::string regularization) {
    std::cout << "preparing model..." << std::endl;
    
    if (_layers.size() < 2) {
        throw DPLException("The neunon network should have at least two layers.");
    }
    const FirstLayer *firstLayer = dynamic_cast<const FirstLayer*>(_layers[0]);
    if (firstLayer == NULL) {
        throw DPLException("The first layer must be of type FirstLayer.");
    }
    
    srand(time(NULL));
    for (int i = 0; i < _layers.size() - 1; i++) {
        _weights.push_back(Matrix::Random(_layers[i + 1]->numberOfInNodes(), _layers[i]->numberOfOutNodes()));
    }
    
    _discreteWeight = Matrix::Random(_layers[1]->numberOfInNodes(), firstLayer->numberOfDiscreteNodes());

    _regular = Regularization::create(regularization);
    _propagator = new SimplePropagator();
}

void Model::train(const Matrix& data, const Matrix& y, HyperParameter params) {
    std::cout << "begin training" << std::endl;
    std::cout << "hyperparameters=" << params << std::endl;
    
    int range = floor(((1 - params.validation_split) * data.rows() + 0.5));
    const Matrix& training = data.block(0, 0, range, data.cols());
    const Matrix& trainingY = y.block(0, 0, range, y.cols());
    
    const Matrix& validation = data.block(range, 0, data.rows() - range, data.cols());
    const Matrix& validationY = y.block(range, 0, y.rows() - range, y.cols());

/*    std::cout << "trainig=" << training << std::endl << std::endl;
    std::cout << "trainigY=" << trainingY << std::endl << std::endl;
    
    std::cout << "validation=" << validation << std::endl << std::endl;
    std::cout << "validationY=" << validationY << std::endl << std::endl;*/
    
    _trainingLoss.resize(params.epochs);
    _validationLoss.resize(params.epochs);
    for (int i = 0; i < params.epochs; i++) {        
        std::vector<int> v;
        if (params.batch > 0 && params.batch < training.rows()) {
            v = pickRandomIndex(training.rows(), params.batch);
        }
        
        const Matrix& features = v.size() ? SUBSET(training, v) : training;
        const Matrix& labels = v.size() ? SUBSET(trainingY, v) : trainingY;
        
        // do forward and backward propagators
        _propagator->forward(_layers, _weights, _discreteWeight, features);
        _propagator->backward(_layers, _weights, _discreteWeight, labels);
        
        for (int j = 0; j < _layers.size() - 1; j++) {
            if (_weights[j].rows() != _propagator->getWeightChange(j).rows() ||
                _weights[j].cols() != _propagator->getWeightChange(j).cols()) {
                throw DPLException("Model::train: getWeightChange matrix size not match.");
            }
            _weights[j] -= params.learningRate * (_propagator->getWeightChange(j)
                                    + params.lambda * _regular->diff(_weights[j]));
        }
        
        _discreteWeight -= params.learningRate * (_propagator->discreteWeightChange() 
                + params.lambda * _regular->diff(_discreteWeight));
        
        if (validation.rows() > 0) {
            _validationLoss[i] = evaluate(validation, validationY);
        }
        _trainingLoss[i] = evaluate(training, trainingY);
    }
}

data_t Model::evaluate(const Matrix& testX, const Matrix& testY) {
    SimplePropagator prop;
    prop.forward(_layers, _weights, _discreteWeight, testX);
    return getOutputLayer()->loss(prop.getResult(), testY);
}

Matrix Model::predict(const Matrix& input) {
    SimplePropagator prop;
    prop.forward(_layers, _weights, _discreteWeight, input);
    return prop.getResult();    
}

void Model::plotLoss(bool rms) const {
    plt::figure_size(1200, 780);
    
    std::vector<data_t> tLoss;
    std::vector<data_t> vLoss;
    bool smoothData = false;
    if (rms) {
        tLoss = DataUtil::eigenArrayToSTL(_trainingLoss.array().sqrt(), smoothData);
        vLoss = DataUtil::eigenArrayToSTL(_validationLoss.array().sqrt(), smoothData);
    } else {
        tLoss = DataUtil::eigenVectorToSTL(_trainingLoss, smoothData);
        vLoss = DataUtil::eigenVectorToSTL(_validationLoss, smoothData);
    }
    // Plot a line whose name will show up as "training loss" in the legend.
    if (tLoss.size() > 0) {
        plt::named_plot("Training loss", tLoss);
    }
    if (vLoss.size() > 0) {
        plt::named_plot("Validation loss", vLoss);
    }
    
    plt::xlabel("Epoch");
    plt::ylabel("Root Mean Square Error");
    
    // Add graph title
    plt::title("Loss");

    // Enable legend.
    plt::legend();
    plt::show();
}

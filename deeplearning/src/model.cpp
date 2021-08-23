#include "model.h"
#include "progressbar.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#define SUBSET(mat, idx) indexing((mat), IndexType(&(idx)[0], (idx).size()))

std::ostream& operator<< (std::ostream& out, const HyperParameter& params) {
  out << "{ epochs=" << params.epochs << ", batch=" << params.batchSize;
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

void Model::addLayer(Layer* layer) {
    LayerWrapper* wrapper = dynamic_cast<LayerWrapper*>(layer);
    if (wrapper == NULL) {
        _layers.push_back(layer);
    } else {
        if (_layers.size() == 0) {
            throw DPLException("There is no layer to wrap.");
        }
        wrapper->setLayer(_layers.back());
        _layers[_layers.size() - 1] = wrapper;
    }
}

void Model::setTraining(bool isTraining) {
    for (int i = 0; i < _layers.size(); i++) {
        _layers[i]->setTraining(isTraining);
    }
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
    const OutputLayer *lastLayer = dynamic_cast<const OutputLayer*>(_layers.back());
    if (lastLayer == NULL) {
        throw DPLException("The last layer must be of type OutputLayer.");
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
    if (validation.rows() > 0) {
        _validationLoss.resize(params.epochs);
    } else {
        _validationLoss.resize(0);
    }
    
    for (int i = 0; i < params.epochs; i++) {
//        std::vector<int> v;
//        if (params.batchSize > 0 && params.batchSize < training.rows()) {
//            v = pickRandomIndex(training.rows(), params.batchSize);
//        }

        // set isTraining to true.
        setTraining(true);
        std::vector<int> idx = DataUtil::randomIndex(training.rows());
        
        const Matrix& x_new = indexing(training, IndexType(&(idx)[0], (idx).size()));
        const Matrix& y_new = indexing(trainingY, IndexType(&(idx)[0], (idx).size()));
        
        data_t currentLoss = .0;

        std::string header = "Epoch " + ToString(i+1) + "/" + ToString(params.epochs);
        int steps = (training.rows() + params.batchSize - 1) / params.batchSize;
        ProgressBar bar(header, steps);

        for (int k = 0; k < steps; k++) {
            SHOW_PROGRESS(bar, k, "loss=" << currentLoss / std::max(k, 1));
        
            int pos = k * params.batchSize;
            int end = std::min(pos + params.batchSize, (int)training.rows());
            const Matrix& features = x_new.block(pos, 0, end - pos, x_new.cols());
            const Matrix& labels = y_new.block(pos, 0, end - pos, y_new.cols());

            // do forward and backward propagators
            _propagator->forward(_layers, _weights, _discreteWeight, features);
            
            currentLoss += getOutputLayer()->loss(_propagator->getResult(), labels);
            _propagator->backward(_layers, _weights, _discreteWeight, labels);
            
            for (int j = 0; j < _layers.size() - 1; j++) {
                if (_weights[j].rows() != _propagator->getWeightChange(j).rows() ||
                    _weights[j].cols() != _propagator->getWeightChange(j).cols()) {
                    std::cout << std::vector<Eigen::Index>{_weights[j].rows(), _weights[j].cols(), _propagator->getWeightChange(j).rows(), _propagator->getWeightChange(j).cols()} << " j=" << j << std::endl;
                    std::cout << "_weights[j]=" << _weights[j] << std::endl;
                    std::cout << "_propagator->getWeightChange(j)=" << _propagator->getWeightChange(j) << std::endl;
                    throw DPLException("Model::train: getWeightChange matrix size not match.");
                }
                _weights[j] -= params.learningRate * (_propagator->getWeightChange(j)
                                        + params.lambda * _regular->diff(_weights[j]));
            }
            
            _discreteWeight -= params.learningRate * (_propagator->discreteWeightChange() 
                    + params.lambda * _regular->diff(_discreteWeight));
            
        }
        _trainingLoss[i] = currentLoss / steps;
        
        // set isTraining to false.
        setTraining(false);
        if (validation.rows() > 0) {
            _validationLoss[i] = evaluate(validation, validationY);
            PROGRESS_DONE(bar, "loss=" << _trainingLoss[i] << " val_loss=" << _validationLoss[i]);
        } else {
            PROGRESS_DONE(bar, "loss=" << _trainingLoss[i]);
        }
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
    int firstIndex = 1; // skip the first loss.
    if (rms) {
        tLoss = DataUtil::eigenArrayToSTL(_trainingLoss.array().sqrt(), smoothData);
        vLoss = DataUtil::eigenArrayToSTL(_validationLoss.array().sqrt(), smoothData);
    } else {
        tLoss = DataUtil::eigenVectorToSTL(_trainingLoss, smoothData);
        vLoss = DataUtil::eigenVectorToSTL(_validationLoss, smoothData);
    }
    // Plot a line whose name will show up as "training loss" in the legend.
    if (tLoss.size() > firstIndex) {
        plt::named_plot("Training loss", std::vector<data_t>(tLoss.begin() + firstIndex, tLoss.end()));
    }
    if (vLoss.size() > firstIndex) {
        plt::named_plot("Validation loss", std::vector<data_t>(vLoss.begin() + firstIndex, vLoss.end()));
    }
    
    plt::xlabel("Epoch");
    plt::ylabel("Root Mean Square Error");
    
    // Add graph title
    plt::title("Loss");

    // Enable legend.
    plt::legend();
    plt::show();
}

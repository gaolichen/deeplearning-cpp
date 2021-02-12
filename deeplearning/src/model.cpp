#include "model.h"

Model::Model(std::string loss) {
    std::cout << "model created.." << std::endl;
    _output = new OutputNode();
    _loss = LossFunction::create(loss);
}

Model::~Model() {
    delete _output;
    delete _loss;
    for (size_t i = 0; i < _layers.size(); i++) {
        delete _layers[i];
    }
    _layers.clear();
}

void Model::addLayer(const Layer* layer) {
    std::cout << "add a layer" << std::endl;
    _layers.push_back(layer);
}

void Model::prepare() {
    srand(time(NULL));
    std::cout << "preparing model..." << std::endl;
    std::cout << "_layers.size()=" << _layers.size() << std::endl;
    for (size_t i = 0; i  + 1 < _layers.size(); i++) {
        _weights.push_back(Matrix::Random(_layers[i]->size(), _layers[i + 1]->size()));
    }
    _weights.push_back(Matrix::Random(_layers.back()->size(), 1));
    std::cout << "model prepared" << std::endl;
}

Vector Model::train(const Matrix& data, const Vector& res, size_t batchSize, size_t epic) {
    std::cout << "begin training" << std::endl;
    return Vector::Random(epic);
}

data_t Model::eval(const Vector& input) {
    std::cout << "evaluating input..." << std::endl;
    
    Vector vec1(input.size() + 1);
    vec1(0) = 1.0;
    vec1.block(1, 0, input.size(), 1) = input;
    Vector vec2;
    for (int i = 0; i < _layers.size(); i++) {
        vec2 = _layers[i]->eval(vec1) * _weights[i];
        std::swap(vec1, vec2);
    }
    
    return _output->eval(vec1(0));
}

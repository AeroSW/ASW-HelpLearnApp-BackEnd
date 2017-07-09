#include "Neuron.h"
#include "NetworkException.h"
#include "FunctionFactory.h"
#include <vector>
#include <iostream>


ASW::Neuron::Neuron(double * weights, uint32_t nweights, const std::string &fkey, double bias) :
m_weights(weights,nweights), m_bias(bias), m_function(FunctionFactory::createFunction(fkey)){
//	this->m_weights = new double[num_weights]; // Allocate resources on the heap for storing of the weights.
//	std::copy(weights, weights + nweights, m_weights);
}
ASW::Neuron::Neuron(uint32_t nweights, const std::string &fkey, double bias) :
m_weights(nweights), m_bias(bias), m_function(FunctionFactory::createFunction(fkey)) {
//	m_weights = new double[num_weights];
}
ASW::Neuron::Neuron(const Neuron &n) :
m_weights(n.m_weights), m_bias(n.m_bias), m_function(n.m_function){
//	this->m_weights = new double[num_weights];
//	std::copy(n.m_weights, n.m_weights + n.num_weights, m_weights);
}
ASW::Neuron::~Neuron() {
//	if (m_weights != nullptr) {
//		delete m_weights;
//		num_weights = 0;
//	}
}

double ASW::Neuron::feed(std::valarray<double> inputs) {
	NEURAL_ASSERT(inputs.size() == m_weights.size(), "Size mismatch between input and number of weights.");
//	double summ = 0.0;
//	for (uint32_t cx = 0; cx < num_weights; cx++) {
//		summ += (inputs[cx] * m_weights[cx]);
//	}
	std::valarray<double> prod = m_weights * inputs;
	double summ = prod.sum() + m_bias;
	m_value = m_function->activationFunction(summ);
	return m_value;
}
double ASW::Neuron::train(double input, std::valarray<double> inputs) {
	NEURAL_ASSERT(inputs.size() == m_weights.size(), "Size mismatch between input and number of weights.");
//	double summ = 0.0;
//	for (uint32_t cx = 0; cx < num_weights; cx++) {
//		summ += (inputs[cx] * m_weights[cx]);
//	}
	std::valarray<double> prod = m_weights * inputs;
	double summ = m_bias + prod.sum();
	m_training_value = m_function->trainingFunction(summ) * input;
	return m_training_value;
}

double ASW::Neuron::getBias() {
	return m_bias;
}
double ASW::Neuron::getTrainingValue() {
	return m_training_value;
}
double ASW::Neuron::getValue() {
	return m_value;
}
double ASW::Neuron::getWeight(uint32_t index) {
	return m_weights[index];
}
std::valarray<double> ASW::Neuron::getWeights() {
//	std::vector<double> weights_vector(num_weights);
//	for (uint32_t cx = 0; cx < num_weights; cx++) {
//		weights_vector[cx] = m_weights[cx];
//	}
	return m_weights;
}

void ASW::Neuron::setBias(double bias) {
	m_bias = bias;
}
void ASW::Neuron::setTrainingValue(double tv) {
	m_training_value = tv;
}
void ASW::Neuron::setValue(double v) {
	m_value = v;
}
void ASW::Neuron::setWeight(double weight, uint32_t index) {
	std::string index_str = std::to_string(index);
	NEURAL_ASSERT(0 < index && index < m_weights.size(), index_str + " is not within the bounds of the array of weights.");
	m_weights[index] = weight;
}
void ASW::Neuron::setWeights(std::valarray<double> n_weights) {
	NEURAL_ASSERT(n_weights.size() == m_weights.size(),"Size mismatch between input and number of weights.");
	m_weights = n_weights;
}
uint32_t ASW::Neuron::numWeights() {
	return m_weights.size();
}
ASW::Neuron& ASW::Neuron::operator=(const ASW::Neuron& n) {
	if (this == &n) return *this;
//	if (m_weights != nullptr) {
//		delete[] m_weights;
//	}
	if (m_weights.size != n.m_weights.size()) {
		m_weights.resize(n.m_weights.size());
	}
	m_weights = n.m_weights;
	m_function = n.m_function;
	m_bias = n.m_bias;
	m_training_value = n.m_training_value;
	m_value = n.m_value;
//	num_weights = n.num_weights;
//	m_weights = new double[num_weights];
//	for (uint32_t cx = 0; cx < num_weights; cx++) {
//		m_weights[cx] = n.m_weights[cx];
//	}
	return *this;
}

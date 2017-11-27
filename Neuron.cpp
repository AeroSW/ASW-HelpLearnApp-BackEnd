#include "Neuron.h"
#include "NetworkException.h"
#include "FunctionFactory.h"
#include <vector>
#include <iostream>


ASW::Neuron::Neuron(double * weights, unsigned int nweights, const std::string &fkey, double bias) :
m_weights(weights,nweights), m_bias(bias), m_function(FunctionFactory::createFunction(fkey)){}

ASW::Neuron::Neuron(unsigned int nweights, const std::string &fkey, double bias) :
m_weights(nweights), m_bias(bias), m_function(FunctionFactory::createFunction(fkey)) {}

ASW::Neuron::Neuron(const Neuron &n) :
m_weights(n.m_weights), m_bias(n.m_bias), m_function(n.m_function){}

ASW::Neuron::~Neuron() {}

double ASW::Neuron::feed(std::valarray<double> inputs) {
	NEURAL_ASSERT(inputs.size() == m_weights.size(), "Size mismatch between input and number of weights.");
	std::valarray<double> prod = m_weights * inputs;
	double summ = prod.sum() + m_bias;
	m_value = m_function->activationFunction(summ);
	return m_value;
}
double ASW::Neuron::train(double input, std::valarray<double> inputs) {
	NEURAL_ASSERT(inputs.size() == m_weights.size(), "Size mismatch between input and number of weights.");
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
double ASW::Neuron::getWeight(unsigned int index) {
	return m_weights[index];
}
std::valarray<double> ASW::Neuron::getWeights() {
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
void ASW::Neuron::setWeight(double weight, unsigned int index) {
	std::string index_str = std::to_string(index);
	NEURAL_ASSERT((0 < index && index < m_weights.size()), index_str + " is not within the bounds of the array of weights.");
	m_weights[index] = weight;
}
void ASW::Neuron::setWeights(std::valarray<double> n_weights) {
	NEURAL_ASSERT(n_weights.size() == m_weights.size(),"Size mismatch between input and number of weights.");
	m_weights = n_weights;
}
unsigned int ASW::Neuron::numWeights() {
	return m_weights.size();
}
ASW::Neuron& ASW::Neuron::operator=(const ASW::Neuron& n) {
	if (this == &n) return *this;
	if (m_weights.size() != n.m_weights.size()) {
		m_weights.resize(n.m_weights.size());
	}
	m_weights = n.m_weights;
	m_function = n.m_function;
	m_bias = n.m_bias;
	m_training_value = n.m_training_value;
	m_value = n.m_value;
	return *this;
}

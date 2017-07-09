#include "Neuron.h"
#include "FunctionFactory.h"
#include <vector>

#include <iostream>


ASW::Neuron::Neuron(double * weights, uint32_t nweights, const std::string &fkey, double bias) :
num_weights(nweights), m_bias(bias), m_function(FunctionFactory::createFunction(fkey)){
	this->m_weights = new double[num_weights]; // Allocate resources on the heap for storing of the weights.
	for (uint32_t cx = 0; cx < num_weights; cx++) {
		m_weights[cx] = weights[cx]; // Copy elements over from the weights array one at a time.
	}
}
ASW::Neuron::Neuron(uint32_t nweights, const std::string &fkey, double bias) :
num_weights(nweights), m_bias(bias), m_function(FunctionFactory::createFunction(fkey)) {
	m_weights = new double[num_weights];
}
ASW::Neuron::Neuron(const Neuron &n) :
num_weights(n.num_weights), m_bias(n.m_bias), m_function(n.m_function){
	this->m_weights = new double[num_weights];
	for (uint32_t cx = 0; cx < num_weights; cx++) {
		m_weights[cx] = n.m_weights[cx];
	}
}
ASW::Neuron::~Neuron() {
	if (m_weights != nullptr) {
		delete m_weights;
		num_weights = 0;
	}
}

double ASW::Neuron::feed(double * inputs) {
	double summ = 0.0;
	for (uint32_t cx = 0; cx < num_weights; cx++) {
		summ += (inputs[cx] * m_weights[cx]);
	}
	summ += m_bias;
	m_value = m_function->activationFunction(summ);
	return m_value;
}
double ASW::Neuron::train(double input, double * inputs) {
	double summ = 0.0;
	for (uint32_t cx = 0; cx < num_weights; cx++) {
		summ += (inputs[cx] * m_weights[cx]);
	}
	summ += m_bias;
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
	if (index < 0 || index >= num_weights) throw -1;
	return m_weights[index];
}
std::vector<double> ASW::Neuron::getWeights() {
	std::vector<double> weights_vector(num_weights);
	for (uint32_t cx = 0; cx < num_weights; cx++) {
		weights_vector[cx] = m_weights[cx];
	}
	return weights_vector;
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
	if (index < 0 || index >= num_weights) {
		throw -1;
	}
	m_weights[index] = weight;
}
void ASW::Neuron::setWeights(double * weights) {
	for (uint32_t cx = 0; cx < num_weights; cx++) {
		m_weights[cx] = weights[cx];
	}
}
uint32_t ASW::Neuron::numWeights() {
	return num_weights;
}
ASW::Neuron& ASW::Neuron::operator=(const ASW::Neuron& n) {
	if (this == &n) return *this;
	if (m_weights != nullptr) {
		delete[] m_weights;
	}
	m_function = n.m_function;
	m_bias = n.m_bias;
	m_training_value = n.m_training_value;
	m_value = n.m_value;
	num_weights = n.num_weights;
	m_weights = new double[num_weights];
	for (uint32_t cx = 0; cx < num_weights; cx++) {
		m_weights[cx] = n.m_weights[cx];
	}
	return *this;
}

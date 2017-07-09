#include "NeuralNetwork.h"
#include "NetworkException.h"
#include <iostream>

ASW::NeuralNetwork::NeuralNetwork(std::vector<std::string>& layer_functions, std::vector<uint32_t>& blueprint) :
m_layers(blueprint),m_starting_indices(blueprint.size()),m_ending_indices(blueprint.size()){
	uint32_t num_layers = m_layers.size(); 
	uint32_t total_num_neurons = 0; 
	for (uint32_t cx = 0; cx < num_layers; cx++) total_num_neurons += m_layers[cx]; 
	m_network = new Neuron*[total_num_neurons]; 
	for (uint32_t cx = 0; cx < m_layers[0]; cx++) {
		m_network[cx] = new Neuron(0l, "linear", 0.0f);
	}
	for (uint32_t cx = 1; cx < num_layers; cx++) {
		uint32_t s_neuron = 0;
		for (uint32_t dx = 0; dx < cx; dx++) s_neuron += m_layers[dx];
		uint32_t e_neuron = s_neuron + m_layers[cx];
		for (uint32_t dx = s_neuron; dx < e_neuron; dx++) {
			m_network[dx] = new Neuron(m_layers[cx - 1], layer_functions[cx - 1], 0.0);
		}
	}
	m_starting_indices[0] = 0;
	m_ending_indices[0] = m_layers[0];
	
	for (uint32_t cx = 1; cx < m_layers.size(); cx++) {
		uint32_t summ = 0;
		for (uint32_t dx = 0; dx < cx; dx++) {
			summ += m_layers[dx];
		}
		m_starting_indices[cx] = summ;
		m_ending_indices[cx] = summ + m_layers[cx];
	}
}
ASW::NeuralNetwork::NeuralNetwork(const NeuralNetwork& nn) :
m_layers(nn.m_layers),m_starting_indices(nn.m_starting_indices),m_ending_indices(nn.m_ending_indices){
	uint32_t total_num_neurons = 0;
	for (uint32_t cx = 0; cx < m_layers.size(); cx++) {
		total_num_neurons += m_layers[cx];
	}
	m_network = new Neuron*[total_num_neurons];
	for (uint32_t cx = 0; cx < total_num_neurons; cx++) {
		m_network[cx] = new Neuron(*(nn.m_network[cx]));
	}
}
ASW::NeuralNetwork::~NeuralNetwork() {
	if (m_network != nullptr) {
		uint32_t total_num_neurons = 0;
		for (uint32_t cx = 0; cx < m_layers.size(); cx++) {
			total_num_neurons += m_layers[cx];
		}
		for (uint32_t cx = 0; cx < total_num_neurons; cx++) {
			delete m_network[cx];
		}
		delete[] m_network;
		m_layers.clear();
		m_starting_indices.clear();
		m_ending_indices.clear();
	}
}
uint32_t ASW::NeuralNetwork::numLayers() {
	return m_layers.size();
}
uint32_t ASW::NeuralNetwork::numNodes(uint32_t layer) {
	if (layer < 0 || layer >= m_layers.size()) {
		NETWORK_THROW("Layer index is out of bounds.");
	}
	return m_layers[layer];
}
uint32_t ASW::NeuralNetwork::getEndingIndex(uint32_t layer) {
	if (layer < 0 || layer >= m_layers.size()) {
		NETWORK_THROW("Layer index is out of bounds.");
	}
	return m_ending_indices[layer];
}
uint32_t ASW::NeuralNetwork::getStartingIndex(uint32_t layer) {
	if (layer < 0 || layer >= m_layers.size()) {
		NETWORK_THROW("Layer index is out of bounds.");
	}
	return m_starting_indices[layer];
}
uint32_t ASW::NeuralNetwork::getNumWeights(uint32_t layer, uint32_t neuron) {
	if (layer < 0 || layer >= m_layers.size()) {
		NETWORK_THROW("Layer index is out of bounds.");
	}
	if (neuron < 0 || neuron >= m_layers[layer]) {
		NETWORK_THROW("Neuron index is out of bounds.")
	}
	uint32_t n = m_starting_indices[layer] + neuron;
	return m_network[n]->numWeights();
}
std::vector<double> ASW::NeuralNetwork::feed(double * inputs) {
	for (uint32_t cx = 0; cx < m_layers[0]; cx++) {
		m_network[cx]->setValue(inputs[cx]);
	}
	for (uint32_t cx = 1; cx < m_layers.size(); cx++) {
		uint32_t input_size = m_layers[cx - 1];
		uint32_t input_start = m_starting_indices[cx - 1];
		uint32_t input_end = m_ending_indices[cx - 1];
		uint32_t curr_start = m_starting_indices[cx];
		uint32_t curr_end = m_ending_indices[cx];

		double * neuron_inputs = new double[input_size];
		for (uint32_t dx = input_start; dx < input_end; dx++) {
			neuron_inputs[dx - input_start] = m_network[dx]->getValue();
		}
		for (uint32_t dx = curr_start; dx < curr_end; dx++) {
			m_network[dx]->feed(neuron_inputs);
		}
		delete[] neuron_inputs;
	}
	std::vector<double> outputs;
	for (uint32_t cx = m_starting_indices[m_layers.size() - 1]; cx < m_ending_indices[m_layers.size() - 1]; cx++) {
		outputs.push_back(m_network[cx]->getValue());
	}
	return outputs;
}
std::vector<double> ASW::NeuralNetwork::getWeights(uint32_t layer, uint32_t neuron) {
	if (layer < 0 || layer >= m_layers.size()) {
		NETWORK_THROW("Layer index is out of bounds.");
	}
	if (neuron < 0 || neuron >= m_layers[layer]) {
		NETWORK_THROW("Neuron index is out of bounds.")
	}
	uint32_t n = m_starting_indices[layer] + neuron;
	return m_network[n]->getWeights();
}
double ASW::NeuralNetwork::getBias(uint32_t layer, uint32_t neuron) {
	if (layer < 0 || layer >= m_layers.size()) {
		NETWORK_THROW("Layer index is out of bounds.");
	}
	if (neuron < 0 || neuron >= m_layers[layer]) {
		NETWORK_THROW("Neuron index is out of bounds.")
	}
	uint32_t n = m_starting_indices[layer] + neuron;
	return m_network[n]->getBias();
}
double ASW::NeuralNetwork::getTraining(uint32_t layer, uint32_t neuron) {
	if (layer < 0 || layer >= m_layers.size()) {
		NETWORK_THROW("Layer index is out of bounds.");
	}
	if (neuron < 0 || neuron >= m_layers[layer]) {
		NETWORK_THROW("Neuron index is out of bounds.")
	}
	uint32_t n = m_starting_indices[layer] + neuron;
	return m_network[n]->getTrainingValue();
}
double ASW::NeuralNetwork::getValue(uint32_t layer, uint32_t neuron) {
	if (layer < 0 || layer >= m_layers.size()) {
		NETWORK_THROW("Layer index is out of bounds.");
	}
	if (neuron < 0 || neuron >= m_layers[layer]) {
		NETWORK_THROW("Neuron index is out of bounds.")
	}
	uint32_t n = m_starting_indices[layer] + neuron;
	return m_network[n]->getValue();
}
double ASW::NeuralNetwork::getWeight(uint32_t layer, uint32_t neuron, uint32_t weight) {
	if (layer < 0 || layer >= m_layers.size()) {
		NETWORK_THROW("Layer index is out of bounds.");
	}
	if (neuron < 0 || neuron >= m_layers[layer]) {
		NETWORK_THROW("Neuron index is out of bounds.")
	}
	uint32_t n = m_starting_indices[layer] + neuron;


	return m_network[n]->getWeight(weight);
}
double ASW::NeuralNetwork::trainNeuron(uint32_t layer, uint32_t neuron, double input) {
	if (layer < 0 || layer >= m_layers.size()) {
		NETWORK_THROW("Layer index is out of bounds.");
	}
	if (neuron < 0 || neuron >= m_layers[layer]) {
		NETWORK_THROW("Neuron index is out of bounds.")
	}
	uint32_t index = m_starting_indices[layer] + neuron;
	double * inputs;
	if (layer != 0) {
		inputs = new double[m_layers[layer - 1]];
		uint32_t start = m_starting_indices[layer - 1];
		uint32_t end = m_ending_indices[layer - 1];
		for (uint32_t i = start; i < end; i++) {
			inputs[i - start] = m_network[i]->getValue();
		}
		double value = m_network[index]->train(input, inputs);
		delete[] inputs;
		return value;
	}
	return 0.0;
}
void ASW::NeuralNetwork::setBias(uint32_t layer, uint32_t neuron, double bias) {
	if (layer < 0 || layer >= m_layers.size()) {
		NETWORK_THROW("Layer index is out of bounds.");
	}
	if (neuron < 0 || neuron >= m_layers[layer]) {
		NETWORK_THROW("Neuron index is out of bounds.")
	}
	uint32_t n = m_starting_indices[layer] + neuron;
	m_network[n]->setBias(bias);
}
void ASW::NeuralNetwork::setTraining(uint32_t layer, uint32_t neuron, double training) {
	if (layer < 0 || layer >= m_layers.size()) {
		NETWORK_THROW("Layer index is out of bounds.");
	}
	if (neuron < 0 || neuron >= m_layers[layer]) {
		NETWORK_THROW("Neuron index is out of bounds.")
	}
	uint32_t n = m_starting_indices[layer] + neuron;
	m_network[n]->setTrainingValue(training);
}
void ASW::NeuralNetwork::setValue(uint32_t layer, uint32_t neuron, double value) {
	if (layer < 0 || layer >= m_layers.size()) {
		NETWORK_THROW("Layer index is out of bounds.");
	}
	if (neuron < 0 || neuron >= m_layers[layer]) {
		NETWORK_THROW("Neuron index is out of bounds.")
	}
	uint32_t n = m_starting_indices[layer] + neuron;
	m_network[n]->setValue(value);
}
void ASW::NeuralNetwork::setWeight(uint32_t layer, uint32_t neuron, uint32_t weight, double weight_value) {
	if (layer < 0 || layer >= m_layers.size()) {
		NETWORK_THROW("Layer index is out of bounds.");
	}
	if (neuron < 0 || neuron >= m_layers[layer]) {
		NETWORK_THROW("Neuron index is out of bounds.")
	}
	uint32_t n = m_starting_indices[layer] + neuron;
	m_network[n]->setWeight(weight_value, weight);
}
void ASW::NeuralNetwork::setWeights(uint32_t layer, uint32_t neuron, double * weights) {
	if (layer < 0 || layer >= m_layers.size()) {
		NETWORK_THROW("Layer index is out of bounds.");
	}
	if (neuron < 0 || neuron >= m_layers[layer]) {
		NETWORK_THROW("Neuron index is out of bounds.")
	}
	uint32_t n = m_starting_indices[layer] + neuron;
	m_network[n]->setWeights(weights);
}
ASW::NeuralNetwork& ASW::NeuralNetwork::operator=(const ASW::NeuralNetwork& nn) {
	if (this == &nn) return *this;
	if (m_network != nullptr) {
		uint32_t total_num_neurons = 0;
		for (uint32_t cx = 0; cx < m_layers.size(); cx++) {
			total_num_neurons += m_layers[cx];
		}
		for (uint32_t cx = 0; cx < total_num_neurons; cx++) {
			delete m_network[cx];
		}
		delete[] m_network;
	}
	m_layers.clear();
	m_starting_indices.clear();
	m_ending_indices.clear();
	m_layers = nn.m_layers;
	m_starting_indices = nn.m_starting_indices;
	m_ending_indices = nn.m_ending_indices;
	uint32_t nn_num_neurons = 0;
	for (uint32_t cx = 0; cx < nn.m_layers.size(); cx++) {
		nn_num_neurons += nn.m_layers[cx];
	}
	m_network = new Neuron*[nn_num_neurons];
	for (uint32_t cx = 0; cx < nn_num_neurons; cx++) {
		m_network[cx] = new Neuron(*(nn.m_network[cx]));
	}
	return *this;
}
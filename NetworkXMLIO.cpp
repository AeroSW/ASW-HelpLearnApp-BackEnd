#include "NetworkXMLIO.h"
#include <map>

ASW::NetworkXMLIO::NetworkXMLIO(const std::string &file_name, const IOType &type):
m_type(type), m_file_name(file_name), doc_opened(false){
	m_document = new tinyxml2::XMLDocument();
	if(m_type == IOType::READ) {
		if(!this->isDocumentOpen()) {
			if(!this->openDocument()) {
				// TODO - STDERROR document opening
			}
		}
	}
}
ASW::NetworkXMLIO::~NetworkXMLIO() {
	if(isDocumentOpen()) {
		if(m_type == IOType::WRITE) {
			this->saveDocument();
		}
		this->closeDocument();
	}
}
bool ASW::NetworkXMLIO::isDocumentOpen() {
	return !this->m_document->Error();
}
bool ASW::NetworkXMLIO::openDocument() {
	tinyxml2::XMLError result;
	if(this->isDocumentOpen()) {
		this->closeDocument();
	}

	result = this->m_document->LoadFile(this->m_file_name.c_str);
	if(result == tinyxml2::XMLError::XML_SUCCESS) {
		this->doc_opened = true;
		return true;
	}
	// TODO - LOG ERROR
	return false;
}
void ASW::NetworkXMLIO::closeDocument(){
	if(this->isDocumentOpen()) {
		this->m_document->Clear();
		this->doc_opened = false;
	}
}
bool ASW::NetworkXMLIO::saveDocument() {
	tinyxml2::XMLError result = this->m_document->SaveFile(this->m_file_name.c_str, false);
	if(result == tinyxml2::XMLError::XML_SUCCESS) {
		return true;
	}
	// TODO - Log Error
	return false;
}
ASW::NeuralNetwork * ASW::NetworkXMLIO::scanNetwork() {
	if(!this->isDocumentOpen()) {
		this->openDocument();
	}
	std::vector<layer_t*> layers;
	std::vector<std::string> layer_functions;
	tinyxml2::XMLElement * network_element = this->m_document->FirstChildElement("network");
	unsigned int layer_count = 0;
	unsigned int node_count = 0;
	for(tinyxml2::XMLElement * layer_element = network_element->FirstChildElement("layer"); layer_element != nullptr; layer_element = layer_element->NextSiblingElement("layer")) {
		layer_count++;
		const char * function_attribute = layer_element->Attribute("function");
		if(function_attribute == nullptr) {
			function_attribute = "linear";
		}
		std::string function_str(function_attribute);
		unsigned long long layer_node_count = 0;
		layers.push_back(scanLayer(layer_element));
		layer_functions.push_back(function_str);
	}
	NeuralNetwork * nn = buildNetwork(layers, layer_functions);
	for(size_t layer_counter = 0; layer_counter < layers.size(); layer_counter++) {
		delete layers[layer_counter];
	}
	return nn;
}
ASW::NetworkIOInterface::layer_t * ASW::NetworkXMLIO::scanLayer(tinyxml2::XMLElement * layer_element) {
	layer_t * layer = new layer_t();
	unsigned long long num_nodes = 0;
	for(tinyxml2::XMLElement * neuron_element = layer_element->FirstChildElement("neuron"); neuron_element != nullptr; neuron_element = neuron_element->NextSiblingElement("neuron")) {
		const char * bias_attribute = neuron_element->Attribute("bias");
		if(bias_attribute == nullptr) {
			bias_attribute = "0";
		}
		std::string bias_str(bias_attribute);
		long double bias = stold(bias_str);
		std::valarray<double> weights(scanNeuron(neuron_element));

		layer->weights.push_back(weights);
		layer->biases.push_back(bias);
		num_nodes++;
	}
	layer->num_nodes = num_nodes;
	return layer;
}
std::valarray<double> ASW::NetworkXMLIO::scanNeuron(tinyxml2::XMLElement * neuron_elem) {
	std::vector<long double> temp_weight_vector;
	for(tinyxml2::XMLElement * weight_elem = neuron_elem->FirstChildElement("weight"); weight_elem != nullptr; weight_elem = weight_elem->NextSiblingElement("weight")) {
		const char * val_attribute = weight_elem->Attribute("value");
		if(val_attribute == nullptr) {
			val_attribute = "0";
		}
		std::string val_str(val_attribute);
		long double value = stold(val_str);
		temp_weight_vector.push_back(value);
	}
	std::valarray<double> weights(temp_weight_vector.size());
	for(size_t counter = 0; counter < temp_weight_vector.size(); counter++) {
		weights[counter] = temp_weight_vector[counter];
	}
	return weights;
}

ASW::NeuralNetwork * ASW::NetworkXMLIO::buildNetwork(std::vector<layer_t*> layers, std::vector<std::string> functions) {
	std::vector<unsigned int> blueprint;
	for(size_t layer_count = 0; layer_count < layers.size(); layer_count++) {
		blueprint.push_back(layers[layer_count]->num_nodes);
	}
	NeuralNetwork * nn = new NeuralNetwork(functions, blueprint);
	for(size_t layer_count = 0; layer_count < layers.size(); layer_count++) {
		if(layers[layer_count]->biases.size() != layers[layer_count]->weights.size()) {
			// TODO - Error
			continue;
		}
		for(size_t node_counter = 0; node_counter < layers[layer_count]->num_nodes; node_counter++) {
			nn->setBias(layer_count, node_counter, layers[layer_count]->biases[node_counter]);
			nn->setWeights(layer_count, node_counter, layers[layer_count]->weights[node_counter]);
		}
	}
	return nn;
}

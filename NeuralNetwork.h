#ifndef ASW_HEADER_NEURAL_NETWORK_H
#define ASW_HEADER_NEURAL_NETWORK_H
#include "Neuron.h"
#include <vector>

namespace ASW {
	class NeuralNetwork {
		private:
			Neuron** m_network;
			std::vector<uint32_t> m_layers;
			std::vector<uint32_t> m_starting_indices;
			std::vector<uint32_t> m_ending_indices;

		public:
			NeuralNetwork(std::vector<std::string>&, std::vector<uint32_t>&);
			NeuralNetwork(const NeuralNetwork&);
			virtual ~NeuralNetwork();
			virtual std::valarray<double> feed(std::valarray<double>);
			virtual std::valarray<double> getWeights(uint32_t, uint32_t); // layer, node
			virtual uint32_t numLayers(); // Returns number of layers in neural net.
			virtual uint32_t numNodes(uint32_t); // Returns number of nodes in a given layer.
			virtual uint32_t getStartingIndex(uint32_t);
			virtual uint32_t getEndingIndex(uint32_t);
			virtual uint32_t getNumWeights(uint32_t, uint32_t);
			virtual double getBias(uint32_t, uint32_t); // layer, node
			virtual double getTraining(uint32_t, uint32_t);
			virtual double getValue(uint32_t, uint32_t); // (layer index, node index)
			virtual double getWeight(uint32_t, uint32_t, uint32_t); // layer, node, weight index
			virtual double trainNeuron(uint32_t, uint32_t, double);
			virtual void setBias(uint32_t, uint32_t, double);
			virtual void setTraining(uint32_t, uint32_t, double);
			virtual void setValue(uint32_t, uint32_t, double);
			virtual void setWeight(uint32_t, uint32_t, uint32_t, double);
			virtual void setWeights(uint32_t, uint32_t, std::valarray<double>);
			virtual NeuralNetwork& operator=(const NeuralNetwork&);
	};
}
#endif

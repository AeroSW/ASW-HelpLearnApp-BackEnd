#ifndef ASW_HEADER_NEURAL_NETWORK_H
#define ASW_HEADER_NEURAL_NETWORK_H
#include "Neuron.h"
#include <vector>

namespace ASW {
	class NeuralNetwork {
		private:
			Neuron** m_network;
			std::vector<unsigned int> m_layers;
			std::vector<unsigned int> m_starting_indices;
			std::vector<unsigned int> m_ending_indices;

		public:
			NeuralNetwork(std::vector<std::string>&, std::vector<unsigned int>&);
			NeuralNetwork(const NeuralNetwork&);
			virtual ~NeuralNetwork();
			virtual std::valarray<double> feed(std::valarray<double>);
			virtual std::valarray<double> getWeights(unsigned int, unsigned int); // layer, node
			virtual unsigned int numLayers(); // Returns number of layers in neural net.
			virtual unsigned int numNodes(unsigned int); // Returns number of nodes in a given layer.
			virtual unsigned int getStartingIndex(unsigned int);
			virtual unsigned int getEndingIndex(unsigned int);
			virtual unsigned int getNumWeights(unsigned int, unsigned int);
			virtual double getBias(unsigned int, unsigned int); // layer, node
			virtual double getTraining(unsigned int, unsigned int);
			virtual double getValue(unsigned int, unsigned int); // (layer index, node index)
			virtual double getWeight(unsigned int, unsigned int, unsigned int); // layer, node, weight index
			virtual double trainNeuron(unsigned int, unsigned int, double);
			virtual void setBias(unsigned int, unsigned int, double);
			virtual void setTraining(unsigned int, unsigned int, double);
			virtual void setValue(unsigned int, unsigned int, double);
			virtual void setWeight(unsigned int, unsigned int, unsigned int, double);
			virtual void setWeights(unsigned int, unsigned int, std::valarray<double>);
			virtual NeuralNetwork& operator=(const NeuralNetwork&);
	};
}
#endif

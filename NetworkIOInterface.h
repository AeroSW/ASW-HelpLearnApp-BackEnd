#pragma once
#include "IOInterface.h"
#include "NeuralNetwork.h"
namespace ASW {
	class NetworkIOInterface : public IOInterface {
		protected:
			struct layer_t {
				std::vector<std::valarray<double>> weights;
				std::vector<double> biases;
				unsigned long long num_nodes;
			};
		public:
			virtual ~NetworkIOInterface() {}
			virtual bool writeNetwork(NeuralNetwork * network) = 0;
			virtual NeuralNetwork * readNetwork() = 0;
			virtual bool changeFile(const std::string &file_name) = 0;
			virtual bool changeType(const IOType &type) = 0;
	};
}

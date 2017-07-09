#pragma once
#include "NeuralNetwork.h"
namespace ASW {
	class TrainingInterface{
		public:
			virtual ~TrainingInterface() {}
			virtual NeuralNetwork* train(NeuralNetwork * network, std::vector<double *> inputs, std::vector<double *> outputs) = 0;
	};
}

#pragma once
#include "NeuralNetwork.h"
namespace ASW {
	class TrainingInterface{
		public:
			virtual ~TrainingInterface() {}
			virtual NeuralNetwork* train(NeuralNetwork * network, std::vector<std::valarray<double>> inputs, std::vector<std::valarray<double>> outputs) = 0;
	};
}

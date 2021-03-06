#pragma once
#include "TrainingInterface.h"
#include "TrainingFactory.h"

namespace ASW {
	class BackPropagationTraining_Tol : public TrainingInterface {
		private:
			double learning_rate;
			double tolerance;
		public:
			BackPropagationTraining_Tol(double, double);
			BackPropagationTraining_Tol(const BackPropagationTraining_Tol&);
			virtual ~BackPropagationTraining_Tol();
			virtual NeuralNetwork * train(NeuralNetwork * network, std::vector<std::valarray<double>> inputs, std::vector<std::valarray<double>> outputs);
			virtual TrainingInterface * copy();
	};
	REGISTER_TRAINING(back_prop_tolerance, BackPropagationTraining_Tol, double, double);
}

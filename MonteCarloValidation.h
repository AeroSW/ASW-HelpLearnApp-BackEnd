#pragma once
// Validation Interface
#include "ValidationInterface.h"
#include "ValidationFactory.h"

/*
	TO-DOs
		design and implement neuralnetwork creating function.
		implement makeBlueprints
		design and implement calculateErrors function
*/

namespace ASW {
	class MonteCarloValidation : public ValidationInterface {
		private:
			unsigned int m_num_bins; // The number of bins to use for validation.
			unsigned int m_max_layers; // The maximum number of layers in a neural network allowed.
			unsigned int m_max_nodes; // The maximum number of nodes in a layer for a neural network allowed.
			std::string m_output_location; // Where we are writing the neural network.
			std::string m_error_calculator;

			std::vector<std::vector<unsigned int>> makeBlueprints(unsigned int num_inputs, unsigned int num_outputs);
		
		public:
			MonteCarloValidation(unsigned int num_bins, unsigned int max_num_layers, unsigned int max_num_nodes, const std::string &output_path, const std::string &error_method = "percent_avg");
			MonteCarloValidation(const MonteCarloValidation &mcv);
			virtual ~MonteCarloValidation();
			virtual void findValidArchitecture(std::vector<std::string> &func_list, std::vector<std::valarray<double>> &inputs, std::vector<std::valarray<double>> &outputs, TrainingInterface* trainer);
	};
} // namespace ASW end

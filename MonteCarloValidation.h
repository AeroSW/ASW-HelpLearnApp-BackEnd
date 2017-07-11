#pragma once
#include "ValidationInterface.h"
#include "TrainingFactory.h"
#include "TrainingInterface.h"
#include "NeuralNetwork.h"

#include "ValidationException.h"
#include "TrainingException.h"
#include "NetworkException.h"

namespace ASW {
	class MonteCarloValidation : public ValidationInterface {
		private:
			uint32_t m_num_bins;
			uint32_t m_max_layers;
			uint32_t m_max_nodes;

			std::vector<std::vector<uint32_t>> makeBlueprints(uint32_t num_inputs, uint32_t num_outputs);
		
		public:
			MonteCarloValidation(uint32_t num_bins, uint32_t max_num_layers, uint32_t max_num_nodes, const std::string &output_path);
			MonteCarloValidation(const MonteCarloValidation &mcv);
			virtual ~MonteCarloValidation();
			template<typename... A>
			virtual void findValidArchitecture(std::vector<std::string> &func_list, std::vector<std::valarray<double>> &inputs, std::vector<std::valarray<double>> &outputs, const std::string &training_method, A... training_args);
	};

	template<typename... A>
	void MonteCarloValidation::findValidArchitecture(std::vector<std::string> &func_list, std::vector<std::valarray<double>> &inputs, 
													 std::vector<std::valarray<double>> &outputs, const std::string &training_method, A... training_args){
		// Check if there are inputs and outputs and they are of equal size.
		VALIDATION_ASSERT((inputs.size() > 0 && outputs.size() > 0 && outputs.size() == inputs.size()), "Invalid number of inputs and outputs.");
		
		TrainingInterface * m_trainer = TrainingFactory::createTraining<A...>(training_method, training_args);
		uint32_t num_inputs = inputs[0].size();
		uint32_t num_outputs = outputs[0].size();
		std::vector<std::vector<uint32_t>> blueprints = makeBlueprints();
	}
}

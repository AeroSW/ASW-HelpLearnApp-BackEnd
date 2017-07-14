#pragma once
#include "ValidationInterface.h"
#include "TrainingFactory.h"
#include "TrainingInterface.h"
#include "NeuralNetwork.h"

#include "ValidationException.h"
#include "TrainingException.h"
#include "NetworkException.h"

#include <map>
#include <random>
#include <cmath>

#include "Debug.h"

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
		
	//	TrainingInterface * m_trainer = TrainingFactory::createTraining<A...>(training_method, training_args);
		uint32_t num_inputs = inputs[0].size();
		uint32_t num_outputs = outputs[0].size();
		std::vector<std::vector<uint32_t>> blueprints = makeBlueprints(num_inputs, num_outputs);
		
		std::vector<std::map<std::string, std::vector<std::valarray<double>>>> m_bins(m_num_bins);
		std::vector<std::valarray<double>> test_inputs;
		std::vector<std::valarray<double>> test_outputs;

		#pragma omp parallel for schedule(dynamic)
		for (uint32_t cx = 0; cx < m_num_bins; cx++) {
			m_bins[cx]["inputs"] = std::vector<std::valarray<double>>();
			m_bins[cx]["outputs"] = std::vector<std::valarray<double>>();
		}

		std::random_device bin_rand;
		std::mt19937_64 bin_gen(bin_rand);
		std::uniform_int_distribution<uint32_t> bin_dist(0, m_num_bins); // 0-based

		uint32_t max_size = static_cast<uint32_t>(ceil(inputs.size() + 1 / (m_num_bins + 1))); // size + 1 to include test set.
		// Evenly and randomly divide inputs and outputs into the bins.
		for (uint32_t cx = 0; cx < inputs.size(); cx++) {
			uint32_t index = bin_dist(bin_gen);
			// while the current bin is max size, reroll the selected bin.
			while (m_bins[index]["inputs"].size() >= max_size) {
				index = bin_dist(bin_gen);
			}
			if (index != m_num_bins) {
				m_bins[index]["inputs"].push_back(inputs[cx]);
				m_bins[index]["outputs"].push_back(outputs[cx]);
			}
			else {
				test_inputs.push_back(inputs[cx]);
				test_outputs.push_back(outputs[cx]);
			}
		}
	
		if (DEBUG) {
			for (uint32_t cx = 0; cx < m_num_bins + 1; cx++) {
				if (cx != m_num_bins) {
					std::string index_num = std::to_string(cx);
					DEBUG_PRINT(index_num + "'s size", m_bins[index]["inputs"].size(), "Inputs for " + index_num);
				}
				else {
					DEBUG_PRINT("Test's size", test_inputs.size(), "Test input's size");
				}
			}
		}

		struct CurrentBestArchetecture {
			NeuralNetwork * m_network;
			double error;
		} cba;
		cba.m_network = nullptr;
		cba.error = 9999999.99;

		#pragma omp parallel shared(cba, m_bins, test_inputs, test_outputs, blueprints)
		{
			TrainingInterface * thread_trainer = TrainingFactory::createTraining<A...>(training_method, training_args...);
			TRAINING_ASSERT((thread_trainer != nullptr), "Training method, " + training_method + ", with specified arguments does not exist.");
			std::random_device thread_rd;
			std::mt19937_64 thread_gen(bin_rand);
			std::uniform_int_distribution<uint32_t> thread_dist(0, m_num_bins - 1); // 0-based
			
			#pragma omp for schedule(dynamic)
			for (uint32_t cx = 0; cx < blueprints.size(); cx++) {
				NeuralNetwork * thread_net = new NeuralNetwork(func_list, blueprints[cx]);
				TRAINING_ASSERT((thread_net != nullptr), "Null neural network was created.");
				double thread_error = 0.0;

				for (uint32_t dx = 0; dx < m_num_bins; dx++) {
					uint32_t non_index = thread_dist(thread_gen);
					std::vector<std::valarray<double>> curr_inputs;
					std::vector<std::valarray<double>> curr_outputs;

					for (uint32_t bx = 0; bx < m_num_bins; bx++) {
						if (bx == non_index) continue;
						curr_inputs.insert(curr_inputs.end(), std::make_move_iterator(m_bins[bx]["inputs"].begin()), std::make_move_iterator(m_bins[bx]["inputs"].end()));
						curr_outputs.insert(curr_outputs.end(), std::make_move_iterator(m_bins[bx]["outputs"].begin()), std::make_move_iterator(m_bins[bx]["outputs"].end()));
					}

					NeuralNetwork * thread_trained_network = thread_trainer->train(thread_net, curr_inputs, curr_outputs);
					TRAINING_ASSERT((thread_trained_network != nullptr), "A nullptr was returned for the training network.");
					//////////////////////////// Calculate error function /////////////////////////////////////////////////////////
					double t_error = 0.0;
					for (uint32_t bx = 0; bx < m_bins[non_index]["inputs"].size(); bx++) {
						std::valarray<double> thread_trained_results = thread_trained_network->feed(m_bins[non_index]["inputs"][bx]);
						t_error += calculateError(thread_trained_results, m_bins[non_index]["outputs"][bx]);
					}
					t_error /= m_bins[non_index]["inputs"].size(); // Average the errors.
					thread_error += t_error;
					///////////////////////////////////////////////////////////////////////////////////////////////////////////////
					delete thread_trained_network;
				}
				thread_error /= m_num_bins;
			//	NeuralNetwork * thread_trained_net = thread_trainer->train(thread_net, );
				#pragma omp critical
				{
					if (cba.error > thread_error) {
						if (cba.m_network != nullptr) delete cba.m_network;
						cba.m_network = thread_net;
						cba.error = thread_error;
					}
					else {
						delete thread_net;
					}
				}
			}

			delete thread_trainer;
		}
		finalTests(cba.m_network, cba.error, test_inputs, test_outputs);
		TrainingInterface * final_trainer = nullptr;
		NeuralNetwork * final_network = nullptr;
		final_trainer = TrainingFactory::createTraining<A...>(training_method, training_args...); // new Trainer
		TRAINING_ASSERT((final_trainer != nullptr), "Training method, " + training_method + ", with specified arguments does not exist.");
		final_network = final_trainer->train(cba.m_network, inputs, outputs); // new Network
		TRAINING_ASSERT((final_network != nullptr), "Final network references null.");
		writeNetwork(final_network);
		
		// Don't need to check final_trainer or final_network... Asserts are raised if they are null...
		delete final_trainer; // delete new Trainer
		delete final_network; // delete new Network
		if(cba.m_network != nullptr)
			delete cba.m_network; // if cba is not null
	} // findValidArchitecture(...) end
} // namespace ASW end

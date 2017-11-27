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
		//	virtual void findValidArchitecture(std::vector<std::string> &func_list, std::vector<std::valarray<double>> &inputs, std::vector<std::valarray<double>> &outputs, const std::string &training_method, A... training_args);
			virtual void findValidArchitecture(std::vector<std::string> &func_list, std::vector<std::valarray<double>> &inputs, std::vector<std::valarray<double>> &outputs, TrainingInterface* trainer);
	};
	/*template<typename... A>
	void MonteCarloValidation<A...>::findValidArchitecture(std::vector<std::string> &func_list, std::vector<std::valarray<double>> &inputs, 
													 std::vector<std::valarray<double>> &outputs, const std::string &training_method, 
													 A... training_args){
		// Check if there are inputs and outputs and they are of equal size.
		VALIDATION_ASSERT((inputs.size() > 0 && outputs.size() > 0 && outputs.size() == inputs.size()), "Invalid number of inputs and outputs.");
		
	//	TrainingInterface * m_trainer = TrainingFactory::createTraining<A...>(training_method, training_args);
		unsigned int num_inputs = inputs[0].size();
		unsigned int num_outputs = outputs[0].size();
		std::vector<std::vector<unsigned int>> blueprints = makeBlueprints(num_inputs, num_outputs);
		
		std::vector<std::map<std::string, std::vector<std::valarray<double>>>> m_bins(m_num_bins);
		std::vector<std::valarray<double>> test_inputs;
		std::vector<std::valarray<double>> test_outputs;

		#pragma omp parallel for schedule(dynamic)
		for (unsigned int cx = 0; cx < m_num_bins; cx++) {
			m_bins[cx]["inputs"] = std::vector<std::valarray<double>>();
			m_bins[cx]["outputs"] = std::vector<std::valarray<double>>();
		} // end for cx

		std::random_device bin_rand;
		std::mt19937_64 bin_gen(bin_rand);
		std::uniform_int_distribution<unsigned int> bin_dist(0, m_num_bins); // 0-based

		unsigned int max_size = static_cast<unsigned int>(ceil(inputs.size() + 1 / (m_num_bins + 1))); // size + 1 to include test set.
		// Evenly and randomly divide inputs and outputs into the bins.
		for (unsigned int cx = 0; cx < inputs.size(); cx++) {
			unsigned int index = bin_dist(bin_gen);
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
			} // fi
		} // end for cx
	
		if (DEBUG) {
			for (unsigned int cx = 0; cx < m_num_bins + 1; cx++) {
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
		} cba; // Create architecture to store current best architecture and its error in.
		cba.m_network = nullptr;
		cba.error = 9999999.99;

		#pragma omp parallel shared(cba, m_bins, m_error_calculator, test_inputs, test_outputs, blueprints)
		{
			TrainingInterface * thread_trainer = TrainingFactory::createTraining<A...>(training_method, training_args...); // Generate the trainer for each thread individually
			ValidationErrorCalculatorInterface * error_calculator = dynamic_cast<ValidationErrorCalculatorInterface*>(CalculatorFactory::createCalculator(m_error_calculator)); // Generate the error_calculator for each thread individually.
			/*
			 The reason for calculating a trainer and error calculator for each thread individually is to allow the threads to avoid
			 dependencies as they each train their respective neural networks and calculate their associated error.
			 *//*
			TRAINING_ASSERT((thread_trainer != nullptr), "Training method, " + training_method + ", with specified arguments does not exist."); // Check that the factories did not return items that are nullptrs, or items that cannot be used.
			VALIDATION_ASSERT((error_calculator != nullptr), "Calculator, " + m_error_calculator + ", is not a proper ValidationErrorCalculatorInterface.");
			std::random_device thread_rd; // Create each thread's random device.
			std::mt19937_64 thread_gen(bin_rand); // create the generator.
			std::uniform_int_distribution<unsigned int> thread_dist(0, m_num_bins - 1); // 0-based
			
			#pragma omp for schedule(dynamic) // Set up the for loop appropriately.
			for (unsigned int cx = 0; cx < blueprints.size(); cx++) {
				NeuralNetwork * thread_net = new NeuralNetwork(func_list, blueprints[cx]);
				TRAINING_ASSERT((thread_net != nullptr), "Null neural network was created.");
				double thread_error = 0.0;

				for (unsigned int dx = 0; dx < m_num_bins; dx++) {
					unsigned int non_index = thread_dist(thread_gen);
					std::vector<std::valarray<double>> curr_inputs;
					std::vector<std::valarray<double>> curr_outputs;

					for (unsigned int bx = 0; bx < m_num_bins; bx++) { // Copy currently being tested values into the current vectors.
						if (bx == non_index) continue;
						curr_inputs.insert(curr_inputs.end(), std::make_move_iterator(m_bins[bx]["inputs"].begin()), std::make_move_iterator(m_bins[bx]["inputs"].end()));
						curr_outputs.insert(curr_outputs.end(), std::make_move_iterator(m_bins[bx]["outputs"].begin()), std::make_move_iterator(m_bins[bx]["outputs"].end()));
					} // end for bx

					NeuralNetwork * thread_trained_network = thread_trainer->train(thread_net, curr_inputs, curr_outputs);
					TRAINING_ASSERT((thread_trained_network != nullptr), "A nullptr was returned for the training network.");
					//////////////////////////// Calculate error function /////////////////////////////////////////////////////////
					double t_error = 0.0; // Used to story local thread's current error.
					for (unsigned int bx = 0; bx < m_bins[non_index]["inputs"].size(); bx++) {
						std::valarray<double> thread_trained_results = thread_trained_network->feed(m_bins[non_index]["inputs"][bx]);
						t_error += error_calculator->calculate(thread_trained_results, m_bins[non_index]["outputs"][bx]);
					} // end for bx
					t_error /= m_bins[non_index]["inputs"].size(); // Average the errors.
					thread_error += t_error;
					///////////////////////////////////////////////////////////////////////////////////////////////////////////////
					delete thread_trained_network;
				} // end for dx
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
					} // fi
				} // end pragma critical
			} // end for cx

			delete thread_trainer;
		} // end pragma parallel
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
	*/
} // namespace ASW end

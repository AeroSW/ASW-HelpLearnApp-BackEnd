#include "BackPropagationTraining.h"
#include <random>
#include <stdint.h>
#include "TrainingException.h"
#include "NetworkException.h"
#include <cmath>

#include <iostream>

ASW::BackPropagationTraining_Tol::BackPropagationTraining_Tol(double lr, double t) :
learning_rate(lr), tolerance(t) {}
ASW::BackPropagationTraining_Tol::BackPropagationTraining_Tol(const ASW::BackPropagationTraining_Tol& bptt):
learning_rate(bptt.learning_rate), tolerance(bptt.tolerance){}
ASW::BackPropagationTraining_Tol::~BackPropagationTraining_Tol() {}
ASW::NeuralNetwork  * ASW::BackPropagationTraining_Tol::train(ASW::NeuralNetwork * network, std::vector<double *> inputs, 
																							std::vector<double *> outputs) {
	// Test if the network being trained is NULL
	// If network is a nullptr, negate it, so the Assert conditional is true to throw an error.
	TRAINING_ASSERT(!(network == nullptr), "Nullptr exception - The Neural Network points to a null reference.");
	
	// Check if the number of inputs is equivilent
	// to the number of outputs.
	TRAINING_ASSERT(inputs.size() == outputs.size(), "Number of example inputs and outputs does not match.");

	std::random_device rd; // Instantiate a random_device object.
	std::mt19937_64 gen(rd()); // Create a generator wrapper for the random device.
	std::uniform_real_distribution<double> dist(-0.1, 0.1);  // Create the distribution for the generator.
	size_t num_layers = network->numLayers(); // Get the number of layers from the neural network.
	ASW::NeuralNetwork * m_network = new ASW::NeuralNetwork(*network); // Create a copy of the Neural Network being
	// trained.  This prevents the original network from being overridden by values during the training.

	for (size_t cx = 0; cx < num_layers; cx++) { // Initialize all weights and biases with random values using the
		// generator and distribution created earlier.
		size_t number_of_current_nodes = m_network->numNodes(cx);
		for (size_t ax = 0; ax < number_of_current_nodes; ax++) {
			size_t num_of_weights_for_current_node = m_network->getNumWeights(cx, ax);
			double * new_weights = new double[num_of_weights_for_current_node];
			for (size_t dx = 0; dx < num_of_weights_for_current_node; dx++) {
				new_weights[dx] = dist(gen);
			}
			m_network->setWeights(cx, ax, new_weights);
			m_network->setBias(cx, ax, dist(gen));
			delete[] new_weights; // Clean up memory.
		}
	}
	double current_example_error; // Tracks error over the testing of all example sets.
	size_t number_of_examples = inputs.size();
	do {
		for (size_t example_counter = 0; example_counter < number_of_examples; example_counter++) {
			double * current_inputs = inputs[example_counter];
			double * current_outputs = outputs[example_counter];
			std::string ex_counter_str = std::to_string(example_counter);
			TRAINING_ASSERT(!(current_inputs == nullptr), "Inputs for example set " + ex_counter_str + " is a null array.");
			TRAINING_ASSERT(!(current_outputs == nullptr), "Outputs for example set " + ex_counter_str + " is a null array.");
			
			// Feed the inputs into the Neural Network.
			// Save the outputs from the fed neural network.
			std::vector<double> feed_outputs;
			try {
				feed_outputs = m_network->feed(current_inputs);
			}
			catch (NetworkException &e) {
				std::string e_what(e.what());
				TRAINING_THROW(e_what);
			}
			catch (std::exception &se) {
				std::string e_what(se.what());
				TRAINING_THROW(e_what);
			}
			catch (...) {
				TRAINING_THROW("An unknown exception has occurred during the feed forward step.");
			}
			// Calculate the differences between the calculated outputs and example outputs.
			// Use this difference to create training values to train the Neural Network's
			// neurons for more optimal output.
			try {
				// Get the number of nodes in the output layer.
				uint32_t output_size = m_network->numNodes(num_layers - 1);
				// Calculate the error between the example and calculated outputs.
				// Train the output neuron(s) which these values belong to.
				for (uint32_t cx = 0; cx < output_size; cx++) { 
					double training_error = current_outputs[cx] - feed_outputs[cx];
					m_network->trainNeuron(num_layers - 1, cx, training_error); 
				}
				// Calculate the training values for the hidden layers now.
				for (int l = num_layers - 2; l > 0; l--) {
					// Get the current hidden layer's size.
					size_t num_curr = m_network->numNodes(l);
					// Get the layer's, the one which the current hidden layer feeds into, size.
					size_t num_next = m_network->numNodes(l + 1);
					for (uint32_t curr_index = 0; curr_index < num_curr; curr_index++) { 
						double summ_next_training_values = 0; 
						for (uint32_t next_index = 0; next_index < num_next; next_index++) {
							// Calculate the difference from the expected value.
							summ_next_training_values += m_network->getWeight(l + 1, next_index, curr_index) * m_network->getTraining(l + 1, next_index);
						}
						// Create the training value for the current neuron.
						m_network->trainNeuron(l, curr_index, summ_next_training_values); 
					}
				}
			}
			catch (NetworkException &e) {
				std::string e_what(e.what());
				TRAINING_THROW(e_what);
			}
			catch (std::exception &se) {
				std::string se_what(se.what());
				TRAINING_THROW(se_what);
			}
			catch (...) {
				TRAINING_THROW("An unknown exception has occurred during the backpropagation step.")
			}
			// Calculate New Weights
			try {
				for (uint32_t cx = 1; cx < num_layers; cx++) {
					// Grab the current layer's size.
					size_t num_curr = m_network->numNodes(cx);
					// Grab the previous layer's size.
					size_t num_prev = m_network->numNodes(cx - 1);
					for (size_t dx = 0; dx < num_curr; dx++) {
						// Get the number of waits for the current neuron.
						size_t num_weights = m_network->getNumWeights(cx, dx);
						double * new_weights = new double[num_weights];
						// Calculate the new weights for the current Neuron using its training value, the algorithm's learning
						// rate, it's old weights, and the input values fed into it.
						for (size_t ax = 0; ax < num_prev; ax++) {
							new_weights[ax] = (m_network->getWeight(cx, dx, ax) + (learning_rate * m_network->getValue(cx - 1, ax) * m_network->getTraining(cx, dx)));
						}
						m_network->setWeights(cx, dx, new_weights); // Set the new weights.
						// Calculate the new bias using a similar method as the weights.
						double new_bias = m_network->getBias(cx, dx) + (learning_rate * m_network->getTraining(cx, dx));
						m_network->setBias(cx, dx, new_bias); // Set the new bias.
						delete[] new_weights; // Clean up memory.
					}
				}
			}
			catch (NetworkException &ne) {
				std::string ne_what(ne.what());
				TRAINING_THROW(ne_what);
			}
			catch (std::exception &se) {
				std::string se_what(se.what());
				TRAINING_THROW(se_what);
			}
			catch (...) {
				TRAINING_THROW("An unknown exception occurred during the weight adjustment step.");
			}
		}
		current_example_error = 0.0; // Calculate absolute error for all outputs with the
		// provided example set.
		for (uint32_t cx = 0; cx < inputs.size(); cx++) {
			double t_outs_err = 0.0;
			std::vector<double> net_results = m_network->feed(inputs[cx]);
			double * curr_outs = outputs[cx];
			for (uint32_t dx = 0; dx < net_results.size(); dx++) {
				t_outs_err += fabs(net_results[dx] - curr_outs[dx]); 
			}
			// Average the errors based on the number of outputs.
			t_outs_err /= net_results.size();
			current_example_error += t_outs_err; // Summ them into the example level error totals.
		}
		// Average the errors based on the number of examples.
		current_example_error /= inputs.size();
		
	} while (current_example_error > tolerance); // If we have hit the threshold, then exit.
	return m_network;
}

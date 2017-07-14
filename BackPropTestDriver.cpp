#include <iostream>
#include "NeuralNetwork.h"
#include "TrainingInterface.h"
#include "TrainingFactory.h"

using namespace std;
using namespace ASW;

int main() {
	int z;
	std::cout << "Hello World!" << std::endl;
	std::cin >> z;
	std::vector<std::string> function_names;
	std::vector<uint32_t> blueprint;
	blueprint.push_back(2);
	blueprint.push_back(3);
	blueprint.push_back(1);
	for (uint32_t i = 0; i < 2; i++) {
		function_names.push_back("logistic");
	}
	NeuralNetwork * nn = new NeuralNetwork(function_names, blueprint);
	std::vector<std::valarray<double>> inputs;
	std::vector<std::valarray<double>> outputs;
	for (uint32_t i = 0; i < 2; i++) {
		for (uint32_t j = 0; j < 2; j++) {
			std::valarray<double> input_set(2);
			std::valarray<double> output_set(1);
			input_set[0] = (double) i;
			input_set[1] = (double) j;
			if (i == j) {
				output_set[0] = 0;
			}
			else {
				output_set[0] = 1;
			}
			inputs.push_back(input_set);
			outputs.push_back(output_set);
		}
	}
	TrainingInterface * m_trainer = TrainingFactory::createTraining<double, double>("back_prop_tolerance", 0.05, 0.01);
	NeuralNetwork * new_network = m_trainer->train(nn, inputs, outputs);
	cin >> z;
	std::cout << "Network Trained" << std::endl;
	for (uint32_t cx = 0; cx < 4; cx++) {
		std::cout << "1: " << inputs[cx][0] << "\t2: " << inputs[cx][1] << '\n';
		std::valarray<double> op = new_network->feed(inputs[cx]);
		std::cout << "output: " << op[0] << std::endl;
	}
	delete nn;
	delete new_network;
	cin >> z;
	return 0;
}

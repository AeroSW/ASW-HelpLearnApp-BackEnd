#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include "TrainingInterface.h"

#include <valarray>

namespace ASW {
	class ValidationInterface {
		public:
			virtual ~ValidationInterface() {}
		//	virtual void findValidArchitecture(std::vector<std::string> &func_list, std::vector<std::valarray<double>> &inputs, std::vector<std::valarray<double>> &outputs, const std::string &training_method, A... training_args) = 0;
			virtual void findValidArchitecture(std::vector<std::string> &func_list, std::vector<std::valarray<double>> &inputs, std::vector<std::valarray<double>> &outputs, TrainingInterface * trainer) = 0;
	};
}

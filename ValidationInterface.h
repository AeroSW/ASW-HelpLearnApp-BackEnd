#pragma once
#include <stdint.h>
#include <string>
#include <vector>

#include <valarray>

namespace ASW {
	class ValidationInterface {
		public:
			virtual ~ValidationInterface() {}
			template<typename... A>
			virtual void findValidArchitecture(std::vector<std::string> &func_list, std::vector<std::valarray<double>> &inputs, std::vector<std::valarray<double>> &outputs, const std::string &training_method, A... training_args) = 0;
	};
}

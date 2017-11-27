#pragma once
#include "CalculatorInterface.h"
#include <valarray>
namespace ASW{
	class ValidationErrorCalculatorInterface : public ASW::CalculatorInterface<std::valarray<double>, std::valarray<double>> {
		public:
			virtual ~ValidationErrorCalculatorInterface() {}
			virtual double calculate(std::valarray<double> given, std::valarray<double> predicted) = 0;
	};
}

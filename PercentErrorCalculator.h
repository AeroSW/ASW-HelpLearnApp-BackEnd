#pragma once

#include "ValidationErrorCalculatorInterface.h"
#include "ValidationException.h"
#include "ValidationCalculatorFactory.h"
namespace ASW {
	class PercentErrorCalculator : public ASW::ValidationErrorCalculatorInterface {
		public:
			PercentErrorCalculator();
			virtual ~PercentErrorCalculator();
			virtual double calculate(std::valarray<double> given, std::valarray<double> predicted);
	};
	REGISTER_VALIDATION_ERROR_CALCULATOR(percent_avg, PercentErrorCalculator);
}

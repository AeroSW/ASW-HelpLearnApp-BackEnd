#ifndef ASW_HEADER_LINEAR_FUNCTION_H
#define ASW_HEADER_LINEAR_FUNCTION_H

#include "ActivationFunctionInterface.h"
#include "FunctionFactory.h"

namespace ASW {
	class LinearFunction : public ActivationFunctionInterface {
		public:
			LinearFunction();
			virtual ~LinearFunction();
			virtual double activationFunction(double);
			virtual double trainingFunction(double);
	};
	REGISTER_FUNCTION(linear, LinearFunction);
}

#endif

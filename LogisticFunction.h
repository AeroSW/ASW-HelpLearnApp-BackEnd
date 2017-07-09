#ifndef ASW_HEADER_LOGISTIC_FUNCTION_H
#define ASW_HEADER_LOGISTIC_FUNCTION_H
#include "ActivationFunctionInterface.h"
#include "FunctionFactory.h"
namespace ASW {
	class LogisticFunction : public ActivationFunctionInterface{
		public:
			LogisticFunction();
			virtual ~LogisticFunction();
			virtual double activationFunction(double);
			virtual double trainingFunction(double);
	};
	REGISTER_FUNCTION(logistic, LogisticFunction);
}
#endif

#include "LogisticFunction.h"
#include <cmath>
ASW::LogisticFunction::LogisticFunction() {}
ASW::LogisticFunction::~LogisticFunction() {}
double ASW::LogisticFunction::activationFunction(double x) {
	double e = exp(-1.0f * x);
	double denom = 1 + e;
	return (1 / denom);
}
double ASW::LogisticFunction::trainingFunction(double x) {
	double e = activationFunction(x);
	return (e * (1 - e));
}

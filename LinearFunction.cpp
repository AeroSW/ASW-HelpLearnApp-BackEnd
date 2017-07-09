#include "LinearFunction.h"
ASW::LinearFunction::LinearFunction() {}
ASW::LinearFunction::~LinearFunction() {}
double ASW::LinearFunction::activationFunction(double input) {
	return input;
}
double ASW::LinearFunction::trainingFunction(double input) {
	return 1.0;
}

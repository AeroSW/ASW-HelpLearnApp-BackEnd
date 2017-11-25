#include "PercentErrorCalculator.h"

ASW::PercentErrorCalculator::PercentErrorCalculator() {}


ASW::PercentErrorCalculator::~PercentErrorCalculator() {}

double ASW::PercentErrorCalculator::calculate(std::valarray<double> given, std::valarray<double> predicted) {
	VALIDATION_ASSERT(given.size() == predicted.size(), "Given size does not match predicted size arrays.");
	std::valarray<double> results = (predicted - given) / given;
	return (results.sum() / results.size()); // Average out the result.
}

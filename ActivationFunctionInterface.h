#pragma once
namespace ASW {
	class ActivationFunctionInterface {
		public:
			virtual ~ActivationFunctionInterface() {}
			virtual double activationFunction(double) = 0;
			virtual double trainingFunction(double) = 0;
	};
}
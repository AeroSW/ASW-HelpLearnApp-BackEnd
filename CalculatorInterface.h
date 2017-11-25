#pragma once
namespace ASW {
	template<typename... A>
	class CalculatorInterface {
		public:
		virtual ~CalculatorInterface() {}
		virtual double calculate(A... args) = 0;
	};
}

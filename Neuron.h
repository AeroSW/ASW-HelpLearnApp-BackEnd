#ifndef ASW_HEADER_NEURON_H
#define ASW_HEADER_NEURON_H
#include <stdint.h>
#include <valarray>
#include <memory>
#include <string>
#include <vector>
#include "ActivationFunctionInterface.h"
namespace ASW {
	class Neuron {
		private:
		//	double* m_weights;
		//	uint32_t num_weights;
			std::valarray<double> m_weights;
			double m_value;
			double m_bias;
			double m_training_value;
			std::shared_ptr<ActivationFunctionInterface> m_function;
		public:
			Neuron(double *, uint32_t, const std::string&, double = 0);
			Neuron(uint32_t, const std::string&, double = 0);
			Neuron(const Neuron &);
			virtual ~Neuron();
			virtual double feed(std::valarray<double>);
			virtual double train(double, std::valarray<double>);
			virtual double getBias();
			virtual double getTrainingValue();
			virtual double getValue();
			virtual double getWeight(uint32_t);
			virtual std::valarray<double> getWeights();
			virtual void setWeight(double, uint32_t);
			virtual void setWeights(std::valarray<double> v);
			virtual void setBias(double);
			virtual void setValue(double);
			virtual void setTrainingValue(double);
			virtual uint32_t numWeights();
			virtual Neuron& operator=(const Neuron&);
	};
}
#endif

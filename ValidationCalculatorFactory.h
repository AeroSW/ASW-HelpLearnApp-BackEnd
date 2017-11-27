#pragma once

#include "ValidationErrorCalculatorInterface.h"

#include <map>
#include <string>
namespace ASW {
	class ValidationErrorCalculatorFactory {
		private:
		struct Functor {
			virtual ~Functor() {}
		};
		template<typename F, typename... A>
		struct D_Functor : public Functor {
			D_Functor(F(*f)(A...));
			D_Functor(const D_Functor<F, A...> &df);
			virtual ~D_Functor();
			virtual F operator()(A...);
			virtual D_Functor<F, A...>& operator=(const D_Functor<F, A...> &df);
			private:
			F(*_f)(A...);
		};
		static inline std::map<std::string, Functor*>& getMap() {
			static std::map<std::string, ValidationErrorCalculatorFactory::Functor*> func_map;
			return func_map;
		}
		template<typename U, typename... A>
		static ValidationErrorCalculatorInterface * create(A... args);

		public:
		template<typename U, typename... A>
		struct Registrator {
			Registrator(const std::string &key);
			virtual ~Registrator();
			private:
			std::string m_key;
		};
		template<typename... A>
		static inline ValidationErrorCalculatorInterface * createValidationErrorCalculator(const std::string &skey, A... args) {
			auto it = getMap().find(skey);
			if (it == getMap().end()) return nullptr;
			D_Functor<ValidationErrorCalculatorInterface*, A...> * df = static_cast<D_Functor<ValidationErrorCalculatorInterface*, A...>*>(it->second);
			return (*df)(args...);
		}
	};
	// ValidationErrorCalculatorFactor::D_Functor Methods
	template <typename F, typename... A>
	ValidationErrorCalculatorFactory::D_Functor<F, A...>::D_Functor(F(*f)(A...)) {
		_f = f;
	}
	template <typename F, typename... A>
	ValidationErrorCalculatorFactory::D_Functor<F, A...>::D_Functor(const ValidationErrorCalculatorFactory::D_Functor<F, A...> &df) {
		_f = df._f;
	}
	template <typename F, typename... A>
	ValidationErrorCalculatorFactory::D_Functor<F, A...>::~D_Functor() {}
	template <typename F, typename... A>
	F ValidationErrorCalculatorFactory::D_Functor<F, A...>::operator()(A... args) {
		return _f(args...);
	}
	template <typename F, typename... A>
	ValidationErrorCalculatorFactory::D_Functor<F, A...>& ValidationErrorCalculatorFactory::D_Functor<F, A...>::operator=(const ValidationErrorCalculatorFactory::D_Functor<F, A...>& df) {
		_f = df._f;
		return *this;
	}
	// ValidationErrorCalculatorFactory::Registrator Methods
	template <typename U, typename... A>
	ValidationErrorCalculatorFactory::Registrator<U, A...>::Registrator(const std::string &key) {
		D_Functor<ValidationErrorCalculatorInterface*, A...> * df = new D_Functor<ValidationErrorCalculatorInterface*, A...>(ValidationErrorCalculatorFactory::create<U, A...>);
		getMap()[key] = df;
		m_key = key;
	}
	template<typename U, typename... A>
	ValidationErrorCalculatorFactory::Registrator<U, A...>::~Registrator() {
		auto it = getMap().find(m_key);
		if (it != getMap().end()) {
			delete it->second;
			getMap().erase(it);
		}
	}
	template<typename U, typename... A>
	ValidationErrorCalculatorInterface * ValidationErrorCalculatorFactory::create(A... args) {
		return new U(args...);
	}
	// ValidationErrorCalculatorFactory Public Methods
}
// Registration Macro
#ifndef REGISTER_VALIDATION_ERROR_CALCULATOR
#define REGISTER_VALIDATION_ERROR_CALCULATOR(NAME,TYPE,...)\
	namespace {\
		::ASW::ValidationErrorCalculatorFactory::Registrator<TYPE,##__VA_ARGS__> function_##NAME(#NAME);\
	}
#endif


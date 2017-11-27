#pragma once
#include "ActivationFunctionInterface.h"
#include <map>
#include <string>
namespace ASW {
	class FunctionFactory {
		private:
			struct Functor {
				virtual ~Functor() {}
			};
			template<typename F, typename... A>
			struct D_Functor : public Functor {
				D_Functor(F(*f)(A...));
				D_Functor(const D_Functor<F,A...> &df);
				virtual ~D_Functor();
				virtual F operator()(A...);
				virtual D_Functor<F, A...>& operator=(const D_Functor<F,A...> &df);
				private:
					F(*_f)(A...);
			};
			static inline std::map<std::string, Functor*>& getMap() {
				static std::map<std::string, FunctionFactory::Functor*> func_map;
				return func_map;
			}
			template<typename U, typename... A>
			static ActivationFunctionInterface * create(A... args);

		public:
			template<typename U, typename... A>
			struct Registrator {
				Registrator(const std::string &key);
				virtual ~Registrator();
				private:
					std::string m_key;
			};
			template<typename... A>
			static inline ActivationFunctionInterface * createFunction(const std::string &skey, A... args) {
				auto it = getMap().find(skey);
				if (it == getMap().end()) return nullptr;
				D_Functor<ActivationFunctionInterface*, A...> * df = static_cast<D_Functor<ActivationFunctionInterface*, A...>*>(it->second);
				return (*df)(args...);
			}
	};
	// FunctionFactor::D_Functor Methods
	template <typename F, typename... A>
	FunctionFactory::D_Functor<F, A...>::D_Functor(F(*f)(A...)) {
		_f = f;
	}
	template <typename F, typename... A>
	FunctionFactory::D_Functor<F,A...>::D_Functor(const FunctionFactory::D_Functor<F,A...> &df){
		_f = df._f;
	}
	template <typename F, typename... A>
	FunctionFactory::D_Functor<F, A...>::~D_Functor() {}
	template <typename F, typename... A>
	F FunctionFactory::D_Functor<F, A...>::operator()(A... args) {
		return _f(args...);
	}
	template <typename F, typename... A>
	FunctionFactory::D_Functor<F, A...>& FunctionFactory::D_Functor<F, A...>::operator=(const FunctionFactory::D_Functor<F, A...>& df) {
		_f = df._f;
		return *this;
	}
	// FunctionFactory::Registrator Methods
	template <typename U, typename... A>
	FunctionFactory::Registrator<U, A...>::Registrator(const std::string &key) {
		D_Functor<ActivationFunctionInterface*, A...> * df = new D_Functor<ActivationFunctionInterface*, A...>(FunctionFactory::create<U, A...>);
		getMap()[key] = df;
		m_key = key;
	}
	template<typename U, typename... A>
	FunctionFactory::Registrator<U, A...>::~Registrator() {
		auto it = getMap().find(m_key);
		if (it != getMap().end()) {
			delete it->second;
			getMap().erase(it);
		}
	}
	template<typename U, typename... A>
	ActivationFunctionInterface * FunctionFactory::create(A... args) {
		return new U(args...);
	}
	// FunctionFactory Public Methods
}
// Registration Macro
#ifndef REGISTER_FUNCTION
#define REGISTER_FUNCTION(NAME,TYPE,...)\
	namespace {\
		::ASW::FunctionFactory::Registrator<TYPE,##__VA_ARGS__> function_##NAME(#NAME);\
	}
#endif

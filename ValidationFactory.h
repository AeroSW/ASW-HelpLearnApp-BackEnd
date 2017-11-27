#pragma once
#include "ValidationInterface.h"
#include <map>
#include <string>
namespace ASW {
	class ValidationFactory {
		private:
		struct ValidationFunctor {
			public:
			virtual ~ValidationFunctor() {}
		};
		template<typename T, typename... A>
		struct ValidationFunctorDerived : public ValidationFunctor {
			ValidationFunctorDerived(T(*f)(A...));
			ValidationFunctorDerived(const ValidationFunctorDerived<T, A...> &tfd);
			virtual ~ValidationFunctorDerived();
			virtual T operator()(A...);
			virtual ValidationFunctorDerived<T, A...>& operator=(const ValidationFunctorDerived<T, A...> &tfd);
			private:
			T(*_f)(A...);
		};
		static inline std::map<std::string, ValidationFunctor*>& getMap() {
			static std::map<std::string, ValidationFactory::ValidationFunctor*> training_map;
			return training_map;
		}
		template<typename U, typename... A>
		static ValidationInterface * create(A... args);

		public:
		template<typename U, typename... A>
		struct Registrator {
			Registrator(const std::string &key);
			virtual ~Registrator();
			private:
			std::string m_key;
		};
		template<typename... A>
		static inline ValidationInterface * createValidation(const std::string &skey, A... args) {
			auto it = getMap().find(skey);
			if (it == getMap().end()) return nullptr;
			ValidationFunctorDerived<ValidationInterface*, A...> * tfd = static_cast<ValidationFunctorDerived<ValidationInterface*, A...>*>(it->second);
			return (*tfd)(args...);
		}
	};
	template<typename T, typename... A>
	ValidationFactory::ValidationFunctorDerived<T, A...>::ValidationFunctorDerived(T(*f)(A...)) {
		_f = f;
	}
	template<typename T, typename... A>
	ValidationFactory::ValidationFunctorDerived<T, A...>::ValidationFunctorDerived(const ValidationFactory::ValidationFunctorDerived<T, A...>& tfd) {
		_f = tfd._f;
	}
	template<typename T, typename... A>
	ValidationFactory::ValidationFunctorDerived<T, A...>::~ValidationFunctorDerived() {}
	template<typename T, typename... A>
	T ValidationFactory::ValidationFunctorDerived<T, A...>::operator()(A... args) {
		return _f(args...);
	}
	template<typename T, typename... A>
	ValidationFactory::ValidationFunctorDerived<T, A...>& ValidationFactory::ValidationFunctorDerived<T, A...>::operator=(const ValidationFactory::ValidationFunctorDerived<T, A...>& tfd) {
		_f = tfd._f;
		return *this;
	}
	// Registrator's methods
	template<typename U, typename... A>
	ValidationFactory::Registrator<U, A...>::Registrator(const std::string &key) {
		ValidationFunctorDerived<ValidationInterface*, A...> * tfd = new ValidationFunctorDerived<ValidationInterface*, A...>(ValidationFactory::create<U, A...>);
		getMap()[key] = tfd;
		m_key = key;
	}
	template<typename U, typename... A>
	ValidationFactory::Registrator<U, A...>::~Registrator() {
		auto it = getMap().find(m_key);
		if (it != getMap().end()) {
			delete it->second;
			getMap().erase(it);
		}
	}
	// ValidationFactory Private Methods
	template<typename U, typename... A>
	ValidationInterface *	ValidationFactory::create(A... args) {
		return new U(args...);
	}
}
#ifndef REGISTER_VALIDATION
#define REGISTER_VALIDATION(NAME,TYPE,...)\
	namespace {\
		::ASW::ValidationFactory::Registrator<TYPE,##__VA_ARGS__> validation_##NAME(#NAME);\
	}
#endif

#pragma once
#include "TrainingInterface.h"
#include <map>
#include <string>
namespace ASW {
	class TrainingFactory {
		private:
			struct TrainingFunctor {
				public:
				virtual ~TrainingFunctor() {}
			};
			template<typename T, typename... A>
			struct TrainingFunctorDerived : public TrainingFunctor {
				TrainingFunctorDerived(T(*f)(A...));
				TrainingFunctorDerived(const TrainingFunctorDerived<T, A...> &tfd);
				virtual ~TrainingFunctorDerived();
				virtual T operator()(A...);
				virtual TrainingFunctorDerived<T, A...>& operator=(const TrainingFunctorDerived<T, A...> &tfd);
				private:
				T(*_f)(A...);
			};
			static inline std::map<std::string, TrainingFunctor*>& getMap() {
				static std::map<std::string, TrainingFactory::TrainingFunctor*> training_map;
				return training_map;
			}
			template<typename U, typename... A>
			static TrainingInterface * create(A... args);

		public:
			template<typename U, typename... A>
			struct Registrator {
				Registrator(const std::string &key);
				virtual ~Registrator();
				private:
				std::string m_key;
			};
			template<typename... A>
			static inline TrainingInterface * createTraining(const std::string &skey, A... args) {
				auto it = getMap().find(skey);
				if (it == getMap().end()) return nullptr;
				TrainingFunctorDerived<TrainingInterface*, A...> * tfd = static_cast<TrainingFunctorDerived<TrainingInterface*, A...>*>(it->second);
				return (*tfd)(args...);
			}
	};
	template<typename T, typename... A>
	TrainingFactory::TrainingFunctorDerived<T, A...>::TrainingFunctorDerived(T(*f)(A...)) {
		_f = f;
	}
	template<typename T, typename... A>
	TrainingFactory::TrainingFunctorDerived<T, A...>::TrainingFunctorDerived(const TrainingFactory::TrainingFunctorDerived<T, A...>& tfd) {
		_f = tfd._f;
	}
	template<typename T, typename... A>
	TrainingFactory::TrainingFunctorDerived<T, A...>::~TrainingFunctorDerived() {}
	template<typename T, typename... A>
	T TrainingFactory::TrainingFunctorDerived<T, A...>::operator()(A... args) {
		return _f(args...);
	}
	template<typename T, typename... A>
	TrainingFactory::TrainingFunctorDerived<T, A...>& TrainingFactory::TrainingFunctorDerived<T, A...>::operator=(const TrainingFactory::TrainingFunctorDerived<T, A...>& tfd) {
		_f = tfd._f;
		return *this;
	}
	// Registrator's methods
	template<typename U, typename... A>
	TrainingFactory::Registrator<U, A...>::Registrator(const std::string &key) {
		TrainingFunctorDerived<TrainingInterface*, A...> * tfd = new TrainingFunctorDerived<TrainingInterface*, A...>(TrainingFactory::create<U, A...>);
		getMap()[key] = tfd;
		m_key = key;
	}
	template<typename U, typename... A>
	TrainingFactory::Registrator<U, A...>::~Registrator(){
		auto it = getMap().find(m_key);
		if (it != getMap().end()) {
			delete it->second;
			getMap().erase(it);
		}
	}
	// TrainingFactory Private Methods
	template<typename U, typename... A>
	TrainingInterface *	TrainingFactory::create(A... args) {
		return new U(args...);
	}
}
#ifndef REGISTER_TRAINING
#define REGISTER_TRAINING(NAME,TYPE,...)\
	namespace {\
		::ASW::TrainingFactory::Registrator<TYPE,##__VA_ARGS__> training_##NAME(#NAME);\
	}
#endif

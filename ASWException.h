#pragma once
#include <exception>
#include <stdexcept>
#include <sstream>
#include <string>

namespace ASW {
	class ASWException : public std::runtime_error {
		private:
			static std::ostringstream err_output;
			size_t line_number;
			std::string filename;
		public:
			virtual ~ASWException() {}
			virtual const char * what() = 0;
	};
}
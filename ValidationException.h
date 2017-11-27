#pragma once

#include <sstream>
#include <iostream>
#include <stdexcept>

namespace ASW {
	class ValidationException : public std::runtime_error {
		private:
			static std::ostringstream m_output_string;
			size_t m_line_number;
			std::string m_file;
			
		public:
			ValidationException(const std::string &filename, size_t line_number, const std::string &what_msg);
			~ValidationException();
			const char * what() const throw();
	};
}

#ifndef VALIDATION_THROW
#define VALIDATION_THROW(MSG) { \
	throw ValidationException(__FILE__,__LINE__,MSG); \
}
#endif

#ifndef VALIDATION_ASSERT
#define VALIDATION_ASSERT(COND, MSG) { \
	if(!COND) { \
		VALIDATION_THROW(MSG); \
	} \
}
#endif

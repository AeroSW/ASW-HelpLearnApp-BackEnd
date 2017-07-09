#pragma once
#include <stdexcept>
#include <sstream>
#include <iostream>

namespace ASW {
	class NetworkException : public std::runtime_error {
		private:
			static std::ostringstream m_output_string;
			size_t m_line_number;
			std::string m_file;
		public:
			NetworkException(const std::string &filename, size_t line_number, const std::string &what_msg);
			virtual ~NetworkException();

			virtual const char * what() const throw();
	};
}
#ifndef NETWORK_THROW
#define NETWORK_THROW(MSG) {\
	throw ASW::NetworkException(__FILE__, __LINE__, MSG); \
}
#endif

#ifndef NEURAL_ASSERT
#define NEURAL_ASSERT(COND, MSG) { \
	if(!COND) { \
		NETWORK_THROW(MSG) \
	} \
}
#endif

#pragma once
#include <stdexcept>
#include <sstream>
#include <iostream>

namespace ASW {
	class TrainingException : public std::runtime_error {
		private:
			static std::ostringstream m_output_string;
			size_t m_line_number;
			std::string m_file;

		public:
			TrainingException(const std::string &filename, size_t line_number, const std::string &what_msg);
			virtual ~TrainingException();

			virtual const char * what() const throw();
	};
}

#ifndef TRAINING_THROW
#define TRAINING_THROW(MSG) { \
	throw ASW::TrainingException(__FILE__, __LINE__, MSG); \
}
#endif

#ifndef TRAINING_ASSERT
#define TRAINING_ASSERT(COND, MSG) { \
	if(!COND) { \
		TRAINING_THROW(MSG); \
	} \
}
#endif

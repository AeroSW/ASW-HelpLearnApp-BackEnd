#include "TrainingException.h"

std::ostringstream ASW::TrainingException::m_output_string;

ASW::TrainingException::TrainingException(const std::string &filename, size_t line_number, const std::string &what_msg) :
m_file(filename),m_line_number(line_number),std::runtime_error(what_msg){}

ASW::TrainingException::~TrainingException() {}

const char * ASW::TrainingException::what() const throw() {
	m_output_string.str(""); // Reset the output stream to an empty string.
	m_output_string << std::runtime_error::what() << "\n\t" << m_file << "\t" << m_line_number; // Format output of error.
	return m_output_string.str().c_str(); // Return the output as a c_string.
}

#include "ValidationException.h"

std::ostringstream ASW::ValidationException::m_output_string;

ASW::ValidationException::ValidationException(const std::string &filename, size_t line_number, const std::string &msg) :
m_file(filename), m_line_number(line_number), std::runtime_error(msg.c_str()){}
ASW::ValidationException::~ValidationException() {}
const char * ASW::ValidationException::what() const throw() {
	m_output_string.str("");
	m_output_string << std::runtime_error::what() << "\n\t" << m_file << "\t" << m_line_number;
	return m_output_string.str().c_str();
}
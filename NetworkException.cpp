#include "NetworkException.h"

std::ostringstream ASW::NetworkException::m_output_string;

ASW::NetworkException::NetworkException(const std::string &filename, size_t line_number, const std::string &msg) :
std::runtime_error(msg.c_str()), m_line_number(line_number), m_file(filename){}
ASW::NetworkException::~NetworkException() {}
const char* ASW::NetworkException::what() const throw(){
	m_output_string.str("");
	m_output_string << std::runtime_error::what() << "\n\t" << m_file << "\t" << m_line_number;
	return m_output_string.str().c_str();
}

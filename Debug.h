#pragma once

#include <iostream>

namespace ASW {
	#ifndef DEBUG
		#define DEBUG true
		#define DEBUG_PRINT(ITM, VAL, MSG) { \
			std::cout << MSG << "\n\t" << ITM << ":\t" << VAL << std::endl; \
		}
		#define DEBUG_ERR(WHAT_MSG) { \
			std::cerr << WHAT_MSG << "\n\t" << __FILE__ << "\t" << __LINE__ << "\n" << std::endl; \
		}
	#endif
}

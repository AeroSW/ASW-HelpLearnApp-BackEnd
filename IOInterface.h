#pragma once
#include <string>
namespace ASW {
	class IOInterface {
		protected:
		std::string m_file_name;
		public:
		virtual ~IOInterface() {}
	};
	enum IOType {
		READ = 0,
		WRITE = 1
	};
}

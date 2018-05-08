#pragma once
#include "NetworkIOInterface.h"
#include "tinyxml2.h"
#include <map>

namespace ASW {
	class NetworkXMLIO : public NetworkIOInterface {
		private:
			IOType m_type;
			std::string m_file_name;
			bool doc_opened;

			tinyxml2::XMLDocument * m_document;

			virtual NeuralNetwork * scanNetwork();
			virtual layer_t * scanLayer(tinyxml2::XMLElement*);
			virtual std::valarray<double> scanNeuron(tinyxml2::XMLElement*);

			virtual NeuralNetwork * buildNetwork(std::vector<layer_t*>,std::vector<std::string>);

			virtual bool writeNetworkTag(NeuralNetwork*);
			virtual bool writeLayerTag(NeuralNetwork*);
			virtual bool writeNeuronTag(NeuralNetwork*);

			virtual bool isDocumentOpen();
			virtual bool openDocument();
			virtual void closeDocument();
			virtual bool saveDocument();

		public:
			NetworkXMLIO(const std::string &file_name, const IOType &type);
			virtual ~NetworkXMLIO();

			virtual bool writeNetwork(NeuralNetwork * network);
			virtual NeuralNetwork * readNetwork();

			virtual bool changeFile(const std::string &file_name);
			virtual bool changeType(const IOType &type);
	};
}

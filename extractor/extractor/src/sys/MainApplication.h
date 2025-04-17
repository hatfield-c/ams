#include <stdio.h>
#include <chrono>
#include <thread>

#include "../CONFIG.h"
#include "../engine/CloudExtractor.h"

typedef char byte;

struct MainApplication {

	CloudExtractor extractor{};

	void Init() {
		printf("[AmScan Extractor v0.0.1]\n\n");
		printf("Initializing...\n");

		this->extractor.Init();
	}

	void Run() {
		printf("Running...\n");

		this->extractor.Run();
	}

};
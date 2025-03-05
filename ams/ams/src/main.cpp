#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "sys/MainApplication.h"

int main() {
	MainApplication application{};
	application.Init();
	application.Run();

	return 0;
}
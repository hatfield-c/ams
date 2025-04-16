#include <stdio.h>
#include <chrono>
#include <thread>

#include "../CONFIG.h"
#include "../hardware/DepthSensor.h"
//#include "../hardware/FlightController.h"
#include "../hardware/MavData.h"

typedef char byte;

struct MainApplication {

	MavData* mav_data = new MavData{};
	DepthSensor depth_sensor{};
	//FlightController flight_controller{};

	std::chrono::steady_clock::time_point frame_begin_time = std::chrono::steady_clock::now();

	void Init() {
		printf("[AmScan v0.0.1]\n\n");
		printf("Initializing...\n");

		this->depth_sensor.Init();
		//this->flight_controller.Init(this->mav_data);
	}

	void Run() {
		printf("Running...\n");

		//this->flight_controller.Start();

		for (int i = 0; i < CONFIG::RecordingSize(); i++) {
			this->FrameSleep();
			this->depth_sensor.ReadSensor(this->mav_data);
		}
		
		this->depth_sensor.recorder.SaveMemory();
		//this->flight_controller.Stop();
	}

	void FrameSleep() {
		std::chrono::steady_clock::duration frame_time_passed = std::chrono::steady_clock::now() - this->frame_begin_time;
		unsigned long long time_lapsed = std::chrono::duration_cast<std::chrono::milliseconds>(frame_time_passed).count();
		unsigned long long delta_time_steps = CONFIG::DeltaTimeMilli();

		if (time_lapsed < delta_time_steps) {
			unsigned long long sleep_time = delta_time_steps - time_lapsed;
			std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
		}

		this->frame_begin_time = std::chrono::steady_clock::now();
	}
};
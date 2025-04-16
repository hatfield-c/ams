#pragma once

#include <string>
#include <librealsense2/rs.hpp>

#include "../CONFIG.h"
#include "../engine/Indexer.h"
#include "Recorder.h"
#include "MavData.h"

struct DepthSensor {

	rs2::config depth_config;
	rs2::pipeline depth_pipe;

	Recorder recorder{};

	void Init() {
		printf("Warming up depth sensor...\n");

		this->depth_config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

		this->depth_pipe.start(depth_config);
		
		rs2::frameset depth_frames;
		rs2::frameset motion_frames;
		for (int i = 0; i < 30; i++) {
			depth_frames = this->depth_pipe.wait_for_frames();
		}

		depth_frames = this->depth_pipe.wait_for_frames();

		rs2::depth_frame depth_frame = depth_frames.get_depth_frame();
		rs2::motion_frame motion_frame = motion_frames.as<rs2::motion_frame>();

		this->recorder.Init();
		printf("    Done!\n");
	}

	void ReadSensor(MavData* mav_data) {
		rs2::frameset depth_frames = this->depth_pipe.wait_for_frames();
		unsigned short* raw_depth = (unsigned short*)depth_frames.get_data();

		this->recorder.Perceive(raw_depth, mav_data->position, mav_data->quaternion, mav_data->velocity);
	}
};
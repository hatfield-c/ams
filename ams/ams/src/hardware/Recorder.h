#pragma once

#include <string>

#include "../../lib/stb/stb_image_write.h"

#include "../CONFIG.h"
#include "../engine/Indexer.h"
#include "../engine/Transform.h"
#include "../engine/Quaternion.h"

struct Recorder {

	Vector2 mem_size{ 140, 120 };

	float* depth_memory = new float[CONFIG::RecordingSize() * 140 * 120];
	Vector3* position_memory = new Vector3[CONFIG::RecordingSize()];
	Vector4* rotation_memory = new Vector4[CONFIG::RecordingSize()];
	unsigned long long memory_index = 0;

	unsigned long long depth_count = CONFIG::RecordingSize() * 140 * 120;

	float max_distance = 20.0f;

	void Init() {
		memset(this->depth_memory, 0, this->depth_count * sizeof(float));
		memset(this->position_memory, 0, CONFIG::RecordingSize() * sizeof(Vector3));
		memset(this->rotation_memory, 0, CONFIG::RecordingSize() * sizeof(Vector4));
	}

	void Perceive(unsigned short* raw_depth, Vector3 position, Vector4 rotation) {
		Vector2 raw_size{ 640, 480 };
		Vector2 clip_size{ 560, 480 };
		Vector2 mem_size = (clip_size / 4).Floor();
		int invalid_depth_band = raw_size.x - clip_size.x - 1;

		for (int x = 0; x < mem_size.x; x++) {
			for (int y = 0; y < mem_size.y; y++) {
				float min_depth = 1000000000;

				Vector2 raw_position{ x + invalid_depth_band, y };
				raw_position = raw_position * 4;

				for (int i = 0; i < 4; i++) {
					for (int j = 0; j < 4; j++) {
						
						unsigned long long raw_index = Indexer::FlatIndex2(raw_position.x + i, raw_position.y + j, raw_size.x);
						float depth = raw_depth[raw_index] / 1000.0f;

						if (depth < min_depth && depth > 0.001) {
							min_depth = depth;
						}
					}
				}
				
				unsigned long long memory_pixel_index = Indexer::FlatIndex3(x, y, this->memory_index, mem_size.x, mem_size.y);
				this->depth_memory[memory_pixel_index] = min_depth;
			}
		}

		this->position_memory[this->memory_index] = position;
		this->rotation_memory[this->memory_index] = rotation;
		this->memory_index++;

		if (this->memory_index >= CONFIG::RecordingSize()) {
			this->memory_index = 0;
		}
	}

	void SaveMemory() {
		std::string depth_path = "data/recording/depth.float";
		std::string position_path = "data/recording/position.vector3";
		std::string rotation_path = "data/recording/rotation.vector4";

		FILE* depth_file = fopen(depth_path.c_str(), "wb+");
		if (depth_file == NULL) {
			printf("\n\nWarning: File did not open when saving depth memory:\n    %s!\n", depth_path.c_str());
			exit(1);
		}
		int result = fwrite(this->depth_memory, sizeof(float), this->depth_count, depth_file);
		fclose(depth_file);

		FILE* position_file = fopen(depth_path.c_str(), "wb+");
		if (position_file == NULL) {
			printf("\n\nWarning: File did not open when saving position memory:\n    %s!\n", position_path.c_str());
			exit(1);
		}
		result = fwrite(this->position_memory, sizeof(Vector3), CONFIG::RecordingSize(), position_file);
		fclose(position_file);

		FILE* rotation_file = fopen(depth_path.c_str(), "wb+");
		if (rotation_file == NULL) {
			printf("\n\nWarning: File did not open when saving rotation memory:\n    %s!\n", rotation_path.c_str());
			exit(1);
		}
		result = fwrite(this->rotation_memory, sizeof(Vector4), CONFIG::RecordingSize(), rotation_file);
		fclose(rotation_file);
	}
};

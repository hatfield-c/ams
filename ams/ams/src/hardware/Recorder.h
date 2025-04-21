#pragma once

#include <string>

#include "../../lib/stb/stb_image_write.h"

#include "../CONFIG.h"
#include "../engine/Indexer.h"
#include "../engine/Transform.h"
#include "../engine/Quaternion.h"

struct Recorder {

	Vector2 mem_size{ 16, 16 };

	float* depth_memory = new float[CONFIG::RecordingSize() * 16 * 16];
	Vector3* position_memory = new Vector3[CONFIG::RecordingSize()];
	Vector4* rotation_memory = new Vector4[CONFIG::RecordingSize()];
	Vector3* velocity_memory = new Vector3[CONFIG::RecordingSize()];
	unsigned long long memory_index = 0;

	unsigned long long depth_data_size = (unsigned long long)(CONFIG::RecordingSize() * 16 * 16);

	float max_distance = 20.0f;

	void Init() {
		memset(this->depth_memory, 0, this->depth_data_size * sizeof(float));
		memset(this->position_memory, 0, CONFIG::RecordingSize() * sizeof(Vector3));
		memset(this->rotation_memory, 0, CONFIG::RecordingSize() * sizeof(Vector4));
		memset(this->velocity_memory, 0, CONFIG::RecordingSize() * sizeof(Vector3));
	}

	void Perceive(unsigned short* raw_depth, Vector3 position, Vector4 rotation, Vector3 velocity) {
		Vector2 raw_size{ 640, 480 };
		Vector2 clip_size{ 560, 480 };
		Vector2 mem_size = (clip_size / Vector2{ 35, 30 }).Floor();
		int invalid_depth_band = raw_size.x - clip_size.x - 1;

		for (int x = 0; x < mem_size.x; x++) {
			for (int y = 0; y < mem_size.y; y++) {
				Vector2 raw_position{ x, y };
				raw_position = raw_position * 16;
				raw_position.x += invalid_depth_band;

				float min_depth = 20.0f;
				for (int i = 0; i < 35; i++) {
					for (int j = 0; j < 30; j++) {
						
						unsigned long long raw_index = Indexer::FlatIndex2(raw_position.x + i, raw_position.y + j, raw_size.x);
						float depth = raw_depth[raw_index] / 1000.0f;

						if (depth < 0.5f) {
							depth = 60.0f;
						}

						if (depth > 20.0f) {
							depth = 20.0f;
						}

						if (depth < 20.0f && depth < min_depth) {
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
		this->velocity_memory[this->memory_index] = velocity;
		this->memory_index++;

		if (this->memory_index >= CONFIG::RecordingSize()) {
			this->memory_index = 0;
		}
	}

	void SaveMemory() {
		std::string depth_path = "data/recording/depth.float";
		std::string position_path = "data/recording/position.vector3";
		std::string rotation_path = "data/recording/rotation.vector4";
		std::string velocity_path = "data/recording/velocity.vector3";

		FILE* depth_file = fopen(depth_path.c_str(), "wb+");
		if (depth_file == NULL) {
			printf("\n\nWarning: File did not open when saving depth memory:\n    %s!\n", depth_path.c_str());
			exit(1);
		}
		int result = fwrite(this->depth_memory, sizeof(float), this->depth_data_size, depth_file);
		fclose(depth_file);

		FILE* position_file = fopen(position_path.c_str(), "wb+");
		if (position_file == NULL) {
			printf("\n\nWarning: File did not open when saving position memory:\n    %s!\n", position_path.c_str());
			exit(1);
		}
		result = fwrite(this->position_memory, sizeof(Vector3), CONFIG::RecordingSize(), position_file);
		fclose(position_file);

		FILE* rotation_file = fopen(rotation_path.c_str(), "wb+");
		if (rotation_file == NULL) {
			printf("\n\nWarning: File did not open when saving rotation memory:\n    %s!\n", rotation_path.c_str());
			exit(1);
		}
		result = fwrite(this->rotation_memory, sizeof(Vector4), CONFIG::RecordingSize(), rotation_file);
		fclose(rotation_file);

		FILE* velocity_file = fopen(velocity_path.c_str(), "wb+");
		if (velocity_file == NULL) {
			printf("\n\nWarning: File did not open when saving velocity memory:\n    %s!\n", velocity_path.c_str());
			exit(1);
		}
		result = fwrite(this->velocity_memory, sizeof(Vector3), CONFIG::RecordingSize(), velocity_file);
		fclose(velocity_file);

		for (int i = CONFIG::RecordingSize() - 10; i < CONFIG::RecordingSize(); i++) {
			this->position_memory[i].Print("", "");
			this->rotation_memory[i].Print("", "");
			this->velocity_memory[i].Print("");
		}
	}
};

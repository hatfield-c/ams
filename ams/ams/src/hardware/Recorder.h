#pragma once

#include <string>

#include "../../lib/stb/stb_image_write.h"

#include "../CONFIG.h"
#include "../engine/Indexer.h"
#include "../engine/Transform.h"
#include "../engine/Quaternion.h"

struct Recorder {

	Vector2 raw_size{ 640.0f, 480.0f };
	Vector2 depth_size{ 560.0f, 480.0f };
	Vector2 phash_size{ 16.0f, 16.0f };
	Vector2 block_size{ 560.0f / 16.0f, 480.0f / 16.0f };

	float* depth_memory = new float[CONFIG::RecordingSize() * 560 * 480];
	float* phash_memory = new float[CONFIG::RecordingSize() * 16 * 16];
	Vector3* position_memory = new Vector3[CONFIG::RecordingSize()];
	Vector4* rotation_memory = new Vector4[CONFIG::RecordingSize()];
	Vector3* velocity_memory = new Vector3[CONFIG::RecordingSize()];
	unsigned long long memory_index = 0;

	unsigned long long depth_data_size = (unsigned long long)(CONFIG::RecordingSize() * 560 * 480);
	unsigned long long phash_data_size = (unsigned long long)(CONFIG::RecordingSize() * 16 * 16);

	float max_distance = 20.0f;

	void Init() {
		memset(this->depth_memory, 0, this->depth_data_size * sizeof(float));
		memset(this->phash_memory, 0, this->phash_data_size * sizeof(float));
		memset(this->position_memory, 0, CONFIG::RecordingSize() * sizeof(Vector3));
		memset(this->rotation_memory, 0, CONFIG::RecordingSize() * sizeof(Vector4));
		memset(this->velocity_memory, 0, CONFIG::RecordingSize() * sizeof(Vector3));
	}

	void Perceive(unsigned short* raw_depth, Vector3 position, Vector4 rotation, Vector3 velocity) {
		int invalid_depth_band = this->raw_size.x - this->depth_size.x - 1;

		for (int x = 0; x < this->phash_size.x; x++) {
			for (int y = 0; y < this->phash_size.y; y++) {
				Vector2 chunk_anchor{ x, y };
				Vector2 clipped_anchor{ x, y };

				chunk_anchor = chunk_anchor * this->block_size;
				clipped_anchor = clipped_anchor * this->block_size;

				chunk_anchor.x += invalid_depth_band;

				float min_depth = 20.0f;
				for (int i = 0; i < this->block_size.x; i++) {
					for (int j = 0; j < this->block_size.y; j++) {

						unsigned long long chunk_index = Indexer::FlatIndex2(chunk_anchor.x + i, chunk_anchor.y + j, this->raw_size.x);
						float depth = raw_depth[chunk_index] / 1000.0f;

						if (depth < 0.5f || depth > 20.0f) {
							depth = 20.0f;
						}

						if (depth < min_depth) {
							min_depth = depth;
						}

						unsigned long long clipped_index = Indexer::FlatIndex3(clipped_anchor.x + i, clipped_anchor.y + j, this->memory_index, this->depth_size.x, this->depth_size.y);
						this->depth_memory[clipped_index] = depth;
					}
				}
				
				unsigned long long memory_pixel_index = Indexer::FlatIndex3(x, y, this->memory_index, phash_size.x, phash_size.y);
				this->phash_memory[memory_pixel_index] = min_depth;
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
		std::string phash_path = "data/recording/phash.float";
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

		FILE* phash_file = fopen(phash_path.c_str(), "wb+");
		if (phash_file == NULL) {
			printf("\n\nWarning: File did not open when saving phash memory:\n    %s!\n", phash_path.c_str());
			exit(1);
		}
		result = fwrite(this->phash_memory, sizeof(float), this->phash_data_size, phash_file);
		fclose(phash_file);

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

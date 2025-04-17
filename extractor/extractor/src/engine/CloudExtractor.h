#pragma once

#include <string>
#include <vector>

#include "stb_image_write.h"
#include "happly.h"

#include "../CONFIG.h"
#include "Indexer.h"
#include "Transform.h"
#include "Quaternion.h"

struct CloudExtractor {

	Vector2 img_size{ 140, 120 };
	unsigned long long img_pixel_count = 140 * 120;

	float* depth_memory = new float[CONFIG::RecordingSize() * 140 * 120];
	Vector3* position_memory = new Vector3[CONFIG::RecordingSize()];
	Vector4* rotation_memory = new Vector4[CONFIG::RecordingSize()];
	Vector3* velocity_memory = new Vector3[CONFIG::RecordingSize()];

	unsigned long long depth_count = CONFIG::RecordingSize() * 140 * 120;

	float max_distance = 20.0f;

	void Init() {
		std::string depth_path = "data/recording/depth.float";
		std::string position_path = "data/recording/position.vector3";
		std::string rotation_path = "data/recording/rotation.vector4";
		std::string velocity_path = "data/recording/velocity.vector3";

		FILE* depth_file;
		fopen_s(&depth_file, depth_path.c_str(), "rb");
		if (depth_file == NULL) {
			printf("\n\n[Warning] file did not open when loading:\n    %s!\n", depth_path.c_str());
			exit(1);
		}
		fread(this->depth_memory, sizeof(float), (size_t)this->depth_count, depth_file);
		fclose(depth_file);

		FILE* position_file;
		fopen_s(&position_file, position_path.c_str(), "rb");
		if (position_file == NULL) {
			printf("\n\n[Warning] file did not open when loading:\n    %s!\n", position_path.c_str());
			exit(1);
		}
		fread(this->position_memory, sizeof(Vector3), (size_t)CONFIG::RecordingSize(), position_file);
		fclose(position_file);

		FILE* rotation_file;
		fopen_s(&rotation_file, rotation_path.c_str(), "rb");
		if (rotation_file == NULL) {
			printf("\n\n[Warning] file did not open when loading:\n    %s!\n", rotation_path.c_str());
			exit(1);
		}
		fread(this->rotation_memory, sizeof(Vector4), (size_t)CONFIG::RecordingSize(), rotation_file);
		fclose(rotation_file);

		FILE* velocity_file;
		fopen_s(&velocity_file, velocity_path.c_str(), "rb");
		if (velocity_file == NULL) {
			printf("\n\n[Warning] file did not open when loading:\n    %s!\n", velocity_path.c_str());
			exit(1);
		}
		fread(this->velocity_memory, sizeof(Vector3), (size_t)CONFIG::RecordingSize(), velocity_file);
		fclose(velocity_file);
	}

	void Run() {
		this->ExtractRawData();
		this->ExtractPointCloud();
	}

	void ExtractPointCloud() {
		unsigned long long frame_index = 300;

		happly::PLYData ply_out = happly::PLYData();

		std::vector<std::array<double, 3>> vertices;

		for (double i = 0; i < 10; i+=0.5) {
			for (double j = 0; j < 10; j += 0.5) {
				for (double k = 0; k < 10; k += 0.5) {
					std::array<double, 3> point = { i, j, k };
					vertices.push_back(point);
				}
			}
		}

		ply_out.addVertexPositions(vertices);
		ply_out.write("data/ply/field.ply", happly::DataFormat::Binary);
	}

	void ExtractRawData() {
		for (unsigned long long i = CONFIG::RecordingSize() - 10; i < CONFIG::RecordingSize(); i++) {
			this->position_memory[i].Print("", "");
			this->rotation_memory[i].Print("", "");
			this->velocity_memory[i].Print();
		}

		std::string base_path = "data/frames/i";
		for (unsigned long long i = 0; i < CONFIG::RecordingSize(); i++) {
			std::string img_path = base_path + std::to_string(9999999999 - i) + ".jpg";
			byte* depth_img = new byte[this->img_pixel_count];

			for (int j = 0; j < this->img_size.y; j++) {
				for (int k = 0; k < this->img_size.x; k++) {
					unsigned long long mem_index = Indexer::FlatIndex3(k, j, i, this->img_size.x, this->img_size.y);
					unsigned long long img_index = Indexer::FlatIndex2(k, j, this->img_size.x);

					float depth_float = this->depth_memory[mem_index];
					depth_float = 20.0f - depth_float;
					depth_float = depth_float / 20.0f;
					depth_float = 255.0f * depth_float;
					byte pixel_value = (byte)depth_float;

					depth_img[img_index] = pixel_value;
				}
			}

			stbi_write_jpg(img_path.c_str(), this->img_size.x, this->img_size.y, 1, depth_img, 100);
		}
	}
};
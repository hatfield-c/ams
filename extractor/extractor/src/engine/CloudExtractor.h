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

	Vector2 img_size{ 560, 480 };
	Vector2 phash_size{ 16, 16 };
	unsigned long long img_pixel_count = 560 * 480;

	float* depth_memory = new float[CONFIG::RecordingSize() * 560 * 480];
	float* phash_memory = new float[CONFIG::RecordingSize() * 16 * 16];
	Vector3* position_memory = new Vector3[CONFIG::RecordingSize()];
	Vector4* rotation_memory = new Vector4[CONFIG::RecordingSize()];
	Vector3* velocity_memory = new Vector3[CONFIG::RecordingSize()];

	unsigned long long depth_count = CONFIG::RecordingSize() * 560 * 480;
	unsigned long long phash_count = CONFIG::RecordingSize() * 16 * 16;

	float max_distance = 20.0f;

	void Init() {
		std::string depth_path = "data/recording/depth.float";
		std::string phash_path = "data/recording/phash.float";
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

		FILE* phash_file;
		fopen_s(&phash_file, phash_path.c_str(), "rb");
		if (phash_file == NULL) {
			printf("\n\n[Warning] file did not open when loading:\n    %s!\n", phash_path.c_str());
			exit(1);
		}
		fread(this->phash_memory, sizeof(float), (size_t)this->phash_count, phash_file);
		fclose(phash_file);

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
		unsigned long long mem_index = 300;
		mem_index = 427;
		//mem_index = 0;
		//mem_index = 519;

		Vector3 position_raw = this->position_memory[mem_index];
		Vector4 rotation_raw = this->rotation_memory[mem_index];
		
		Vector3 position{ position_raw.x, position_raw.y, position_raw.z };

		Vector4 rotation{ -rotation_raw.x, rotation_raw.z, -rotation_raw.y, rotation_raw.w };
		//rotation = Quaternion::GetQuaternionConjugate(rotation);
		Vector4 pitch_correction = Quaternion::QuaternionFromEulerParams(Vector3{ 0, 0, 1 }, Math::Degree2Radian(5.0f));
		Vector4 yaw_correction = Quaternion::QuaternionFromEulerParams(Vector3{ 0, 1, 0 }, Math::Degree2Radian(90.0f));
		//rotation = Quaternion::MultiplyQuaternions(rotation, pitch_correction, true);
		rotation = Quaternion::MultiplyQuaternions(yaw_correction, rotation, true);

		happly::PLYData ply_out = happly::PLYData();
		std::vector<std::array<double, 3>> vertices;

		for (unsigned long long i = 300; i < 519; i++) {
			Vector3 pos = this->position_memory[i];
			std::array<double, 3> point = { pos.x, pos.z, pos.y };
			vertices.push_back(point);
		}

		position.Print();
		
		for (unsigned long long y = 0; y < this->img_size.y; y++) {
			for (unsigned long long x = 0; x < this->img_size.x; x++) {
				unsigned long long pixel_index = Indexer::FlatIndex3(x, y, mem_index, this->img_size.x, this->img_size.y);
				float depth = this->depth_memory[pixel_index];

				if (depth == 20.0f) {
					continue;
				}

				//Vector3 ray_direction = this->GetCameraRayDirection(Vector2{ (float)x, (float)y }, this->img_size, rotation_raw);
				Vector3 ray_direction = this->GetCameraRayDirection(Vector2{ (float)x, (float)y }, this->img_size, rotation);
				Vector3 p = ray_direction * depth;
				p += position;

				std::array<double, 3> point = { p.x, p.z, p.y };
				vertices.push_back(point);
			}
		}

		ply_out.addVertexPositions(vertices);
		ply_out.write("data/ply/field.ply", happly::DataFormat::Binary);
	}

	Vector3 DepthToPoint(float depth, Vector2 pixel, Vector3 position, Vector4 rotation) {
		//Vect
	}

	Vector3 GetCameraRayDirection(Vector2 pixel_position, Vector2 canvas_size, Vector4 camera_rotation) {
		//Vector2 fov{ 75.0f / 2.0f, 65 / 2.0f };
		//Vector2 fov{ 1.309f, 1.082f };
		Vector2 fov{ Math::Degree2Radian(75.0f), Math::Degree2Radian(65.0f) };

		Vector2 fov_offset{
			-sinf(fov.x / 2),
			-sinf(fov.y / 2)
		};

		Vector2 screen_interpolation{
			((2 * pixel_position.x) / canvas_size.x) - 1,
			((2 * pixel_position.y) / canvas_size.y) - 1
		};

		Vector3 ray_anchor{
			1,
			fov_offset.y * screen_interpolation.y,
			fov_offset.x * screen_interpolation.x
		};

		ray_anchor = Quaternion::RotatePoint(ray_anchor, camera_rotation);

		Vector3 ray_direction = Transform::Unit3(ray_anchor);

		return ray_direction;
	}

	void ExtractRawData() {
		for (unsigned long long i = CONFIG::RecordingSize() - 10; i < CONFIG::RecordingSize(); i++) {
			//this->position_memory[i].Print("", "");
			//this->rotation_memory[i].Print("", "");
			//this->velocity_memory[i].Print();
		}

		std::string base_path = "data/depth/i";
		for (unsigned long long i = 0; i < CONFIG::RecordingSize(); i++) {
			std::string img_path = base_path + std::to_string(1000000000 + i) + ".jpg";
			byte* depth_img = new byte[this->img_pixel_count];

			for (int j = 0; j < this->img_size.y; j++) {
				for (int k = 0; k < this->img_size.x; k++) {
					unsigned long long mem_index = Indexer::FlatIndex3(k, j, i, this->img_size.x, this->img_size.y);
					unsigned long long img_index = Indexer::FlatIndex2(k, j, this->img_size.x);

					float depth_float = this->depth_memory[mem_index];
					//depth_float = 20.0f - depth_float;
					depth_float = depth_float / 20.0f;
					depth_float = 255.0f * depth_float;
					byte pixel_value = (byte)depth_float;

					depth_img[img_index] = pixel_value;
				}
			}

			stbi_write_jpg(img_path.c_str(), this->img_size.x, this->img_size.y, 1, depth_img, 100);
		}

		base_path = "data/phash/i";
		for (unsigned long long i = 0; i < CONFIG::RecordingSize(); i++) {
			std::string img_path = base_path + std::to_string(1000000000 + i) + ".jpg";
			byte* phash_img = new byte[this->img_pixel_count];

			for (int j = 0; j < this->phash_size.y; j++) {
				for (int k = 0; k < this->phash_size.x; k++) {
					unsigned long long mem_index = Indexer::FlatIndex3(k, j, i, this->phash_size.x, this->phash_size.y);
					unsigned long long img_index = Indexer::FlatIndex2(k, j, this->phash_size.x);

					float depth_float = this->phash_memory[mem_index];
					//depth_float = 20.0f - depth_float;
					depth_float = depth_float / 20.0f;
					depth_float = 255.0f * depth_float;
					byte pixel_value = (byte)depth_float;

					phash_img[img_index] = pixel_value;
				}
			}

			stbi_write_jpg(img_path.c_str(), this->phash_size.x, this->phash_size.y, 1, phash_img, 100);
		}
	}
};
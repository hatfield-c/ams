#pragma once

#include <math.h>
#include "cuda.h"
#include "cudart_platform.h"
#include "device_launch_parameters.h"

struct Math {
	static __host__ __device__ float Pi() {
		return 3.14159265358979323846f;
	}

	static __host__ __device__ float Degree2Radian(float degree) {
		float radian = (Math::Pi() * degree) / 180.0f;

		return radian;
	}

	static __host__ __device__ float Radian2Degree(float radian) {
		float degree = (180.0f * radian) / Math::Pi();

		return degree;
	}

	static __host__ __device__ int Clip(int value, int lower, int upper) {
		if (value < lower) {
			value = lower;
		}

		if (value > upper) {
			value = upper;
		}

		return value;
	}

	static __host__ __device__ double Clip(double value, double lower, double upper) {
		if (value < lower) {
			value = lower;
		}

		if (value > upper) {
			value = upper;
		}

		return value;
	}
};
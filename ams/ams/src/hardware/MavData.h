#pragma once

//#include <mavsdk/mavsdk.h>
//#include <mavsdk/plugins/telemetry/telemetry.h>
#include "../engine/Transform.h"

struct MavData {
	Vector3 position{};
	Vector4 quaternion{ 1, 0, 0, 0 };
	Vector3 velocity{};
	
	//mavsdk::Telemetry::PositionVelocityNedHandle position_handle;
	//mavsdk::Telemetry::AttitudeQuaternionHandle quaternion_handle;
};
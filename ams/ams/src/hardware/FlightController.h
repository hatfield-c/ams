#pragma once

#include <iostream>
#include <thread>
#include <chrono>

#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/info/info.h>
#include <mavsdk/plugins/telemetry/telemetry.h>

#include "MavData.h"

struct FlightController {
    mavsdk::Mavsdk* mav_connection;
    mavsdk::Telemetry* telemtry;

    MavData* mav_data;
    bool is_new_mode = false;

    void Init(MavData* mav_data) {
        this->mav_data = mav_data;

		printf("Connecting to ArduPilot through MAVLink...\n");

        this->mav_connection = new mavsdk::Mavsdk{ mavsdk::Mavsdk::Configuration{ mavsdk::ComponentType::CompanionComputer } };
		mavsdk::ConnectionResult connection_result = this->mav_connection->add_any_connection("serial:///dev/ttyACM0:57600");
		
		if (connection_result != mavsdk::ConnectionResult::Success) {
			std::cerr << "Adding connection failed: " << connection_result << '\n';
			exit(1);
		}

		printf("Device found!\n");

		while (this->mav_connection->systems().size() == 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
		}

		printf("Connection success!\n");

        auto system = this->mav_connection->systems()[0];
        this->telemtry = new mavsdk::Telemetry{ system };

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        this->SetStreamRates();
        this->SubscribeStreams();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
	}
    void Start() {

    }

    void Stop() {
        this->UnsubscribeStreams();
    }

    void SubscribeStreams()
    {
        this->mav_data->position_handle = this->telemtry->subscribe_position_velocity_ned([=](mavsdk::Telemetry::PositionVelocityNed data) {
            this->mav_data->position.x = data.position.east_m;
            this->mav_data->position.y = -data.position.down_m;
            this->mav_data->position.z = data.position.north_m;

            this->mav_data->velocity.x = data.velocity.east_m_s;
            this->mav_data->velocity.y = -data.velocity.down_m_s;
            this->mav_data->velocity.x = data.velocity.north_m_s;
        });

        this->mav_data->quaternion_handle = this->telemtry->subscribe_attitude_quaternion([=](mavsdk::Telemetry::Quaternion quaternion) {
            this->mav_data->quaternion.x = quaternion.x;
            this->mav_data->quaternion.y = quaternion.y;
            this->mav_data->quaternion.z = quaternion.z;
            this->mav_data->quaternion.w = quaternion.w;
        });

    }

    void UnsubscribeStreams() {
        this->telemtry->unsubscribe_position_velocity_ned(this->mav_data->position_handle);
        this->telemtry->unsubscribe_attitude_quaternion(this->mav_data->quaternion_handle);
    }

    void SetStreamRates()
    {
        const auto result0 = this->telemtry->set_rate_position_velocity_ned(20.0);
        if (result0 != mavsdk::Telemetry::Result::Success) {
            std::cerr << "Setting set_rate_position_velocity_ned failed: " << result0 << '\n';
        }

        const auto result1 = this->telemtry->set_rate_attitude_quaternion(20.0);
        if (result1 != mavsdk::Telemetry::Result::Success) {
            std::cerr << "Setting set_rate_attitude_quaternion failed: " << result1 << '\n';
        }
    }
};

#pragma once

typedef char byte;

struct CONFIG {
    static unsigned long long RecordingSize() {
        return 20 * 60 * 1;
    }

    static float DeltaTime() {
        return 1.0f / 20.0f;
    }

    static float DeltaTimeMilli() {
        return CONFIG::DeltaTime() * 1000;
    }
};
#pragma once

typedef char byte;

struct CONFIG {
    static int RecordingSize() {
        return 20 * 40 * 1;
    }

    static float DeltaTime() {
        return 1.0f / 20.0f;
    }

    static float DeltaTimeMilli() {
        return CONFIG::DeltaTime() * 1000;
    }
};
// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with common samples functionality
 * @file common.hpp
 */

#pragma once

#include <string>
#include <map>
#include <vector>
#include <list>
#include <limits>
#include <functional>
#include <fstream>
#include <iomanip>
#include <utility>
#include <algorithm>
#include <random>
#include <iostream>

#include <openvino/openvino.hpp>
#include "utils/slog.hpp"
#include "utils/args_helper.hpp"

#ifndef UNUSED
  #ifdef _WIN32
    #define UNUSED
  #else
    #define UNUSED  __attribute__((unused))
  #endif
#endif

template <typename T, std::size_t N>
constexpr std::size_t arraySize(const T (&)[N]) noexcept {
    return N;
}

template <typename T>
T clamp(T value, T low, T high) {
    return value < low ? low : (value > high ? high : value);
}

// Redefine operator<< for LogStream to print IE version information.
inline slog::LogStream& operator<<(slog::LogStream& os, const ov::Version& version) {
    return os << "OpenVINO" << slog::endl
        << "\tversion: " << OPENVINO_VERSION_MAJOR << "." << OPENVINO_VERSION_MINOR << "." << OPENVINO_VERSION_PATCH << slog::endl
        << "\tbuild: " << version.buildNumber;
}

/**
 * @class Color
 * @brief A Color class stores channels of a given color
 */
class Color {
private:
    unsigned char _r;
    unsigned char _g;
    unsigned char _b;

public:
    /**
     * A default constructor.
     * @param r - value for red channel
     * @param g - value for green channel
     * @param b - value for blue channel
     */
    Color(unsigned char r,
          unsigned char g,
          unsigned char b) : _r(r), _g(g), _b(b) {}

    inline unsigned char red() const {
        return _r;
    }

    inline unsigned char blue() const {
        return _b;
    }

    inline unsigned char green() const {
        return _g;
    }
};

// Known colors for training classes from the Cityscapes dataset
static UNUSED const Color CITYSCAPES_COLORS[] = {
    { 128, 64,  128 },
    { 232, 35,  244 },
    { 70,  70,  70 },
    { 156, 102, 102 },
    { 153, 153, 190 },
    { 153, 153, 153 },
    { 30,  170, 250 },
    { 0,   220, 220 },
    { 35,  142, 107 },
    { 152, 251, 152 },
    { 180, 130, 70 },
    { 60,  20,  220 },
    { 0,   0,   255 },
    { 142, 0,   0 },
    { 70,  0,   0 },
    { 100, 60,  0 },
    { 90,  0,   0 },
    { 230, 0,   0 },
    { 32,  11,  119 },
    { 0,   74,  111 },
    { 81,  0,   81 }
};

inline void showAvailableDevices() {
    ov::runtime::Core core;
    std::vector<std::string> devices = core.get_available_devices();

    std::cout << std::endl;
    std::cout << "Available target devices:";
    for (const auto& device : devices) {
        std::cout << "  " << device;
    }
    std::cout << std::endl;
}

inline std::string fileNameNoExt(const std::string &filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos) return filepath;
    return filepath.substr(0, pos);
}

inline void logExecNetworkInfo(const ov::runtime::ExecutableNetwork& execNetwork, const std::string& modelName,
    const std::string& deviceName, const std::string& modelType = "") {
    slog::info << "The " << modelType << (modelType.empty() ? "" : " ") << "model " << modelName << " is loaded to " << deviceName << slog::endl;
    std::set<std::string> devices;
    for (const std::string& device : parseDevices(deviceName)) {
        devices.insert(device);
    }

    if (devices.find("AUTO") == devices.end()) { // do not print info for AUTO device
        for (const auto& device : devices) {
            try {
                slog::info << "\tDevice: " << device << slog::endl;
                std::string nstreams = execNetwork.get_config(device + "_THROUGHPUT_STREAMS").as<std::string>();
                slog::info << "\t\tNumber of streams: " << nstreams << slog::endl;
                if (device == "CPU") {
                    std::string nthreads = execNetwork.get_config("CPU_THREADS_NUM").as<std::string>();
                    slog::info << "\t\tNumber of threads: " << (nthreads == "0" ? "AUTO" : nthreads) << slog::endl;
                }
            }
            catch (const InferenceEngine::Exception&) {}
        }
    }
}

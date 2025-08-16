#include <moto/utils/timed_block.hpp>
#include <chrono>
#include <thread>
#include <iostream>
namespace moto {
namespace utils {

// Function to get the TSC frequency in cycles per second (Hz)
unsigned long long get_tsc_frequency() {
    static unsigned long long frequency = 0;
    if (frequency == 0) {

        // We'll calibrate the TSC against a known high-resolution timer

        // Get the start time from the system's monotonic clock
        auto start_time = std::chrono::high_resolution_clock::now();

        // Get the start TSC value with a serializing instruction
        unsigned long long start_tsc = moto::utils::rdtscp();

        // Wait for a short, but measurable duration (e.g., 50 milliseconds)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // Get the end time from the system's monotonic clock
        auto end_time = std::chrono::high_resolution_clock::now();

        // Get the end TSC value with a serializing instruction
        unsigned long long end_tsc = moto::utils::rdtscp();

        // Calculate the duration in seconds
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        double duration_in_seconds = static_cast<double>(duration) / 1e9;

        // Calculate the number of elapsed cycles
        unsigned long long elapsed_cycles = end_tsc - start_tsc;

        // Calculate the frequency
        frequency = static_cast<unsigned long long>(static_cast<double>(elapsed_cycles) / duration_in_seconds);
        std::cout << "Calibrated TSC frequency: " << frequency / 1e6 << " MHz" << std::endl;
    }
    return frequency;
}
} // namespace utils
} // namespace moto

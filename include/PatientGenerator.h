// PatientGenerator.h - deterministic patient arrival and attribute generation.
#pragma once
#include <vector>

#include "Patient.h"
#include "Scenario.h"

namespace hs {

// Small deterministic RNG (splitmix64). Same seed -> same scenario on every run,
// which lets the CPU and CUDA backends be compared on identical inputs.
class Rng {
public:
    explicit Rng(uint64_t seed) : s_(seed) {}
    uint64_t next() {
        uint64_t z = (s_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }  // [0,1)
    float uniformf(float a, float b) { return a + (b - a) * (float)uniform(); }

private:
    uint64_t s_;
};

class PatientGenerator {
public:
    // Generates cfg.totalPatients records sorted by arrival time (id == index).
    static std::vector<PatientRecord> generate(const ScenarioConfig& cfg);
    // Rule-based synthetic priority assignment from a configurable mix.
    static int samplePriority(Rng& rng, const float mix[kNumPriorities]);
};

}  // namespace hs

// Benchmark.h - reproducible CPU vs CUDA experiments and configuration comparison.
#pragma once
#include <atomic>
#include <functional>
#include <string>
#include <vector>

#include "CudaEngine.h"
#include "Hospital.h"
#include "Scenario.h"
#include "SimulationEngine.h"

namespace hs {

struct BenchmarkOptions {
    std::vector<int> sizes = {100, 1000, 5000, 10000, 50000, 100000, 500000, 1000000};
    int steps = 200;           // update steps per run
    int repeats = 3;           // median of N runs
    int cpuThreads = 0;        // 0 = hardware concurrency
    int blockSize = 256;       // default block size for size sweeps
    std::vector<int> blockSizes = {32, 64, 128, 256, 512, 1024};
    float benchDt = 0.02f;     // small dt keeps the synthetic population dynamic
    uint64_t seed = 42;
    std::vector<int> scenarioSizes = {1000, 10000, 50000, 100000};
    double scenarioHours = 6.0;
    std::string outDir = "benchmarks/results";
    std::function<void(const std::string&, float)> progress;  // (message, fraction 0..1)
    std::atomic<bool>* cancel = nullptr;
    static BenchmarkOptions quick();
};

struct KernelBenchRow {
    std::string experiment, variant, layout, ordering;
    int blockSize = 0, patients = 0, steps = 0, threads = 1;
    double inactiveFraction = 0;
    double kernelMs = 0, h2dMs = 0, d2hMs = 0, e2eMs = 0;
    double perStepUs = 0;          // e2e / steps
    double updatesPerSec = 0;      // patients * steps / e2e
    double speedupKernel = 0;      // CPU(1 thread) time / kernel-only time
    double speedupE2E = 0;         // CPU(1 thread) time / end-to-end time
    double maxAbsDiff = 0;         // vs CPU(1 thread) final state
    long long intMismatches = 0;
    bool eventsMatch = true;
    bool isGpu = false;
};

struct ScenarioBenchRow {
    int patients = 0;
    std::string backend;
    bool isGpu = false, batching = false;
    double wallMs = 0, backendMs = 0, controllerMs = 0, commandMs = 0;
    double computeMs = 0, h2dMs = 0, d2hMs = 0;
    uint64_t steps = 0, events = 0, commands = 0, syncs = 0;
    uint64_t bytesH2D = 0, bytesD2H = 0;
    int discharged = 0;
    double avgWait = 0, avgTreatmentDelay = 0;
    bool identicalToCpu = true;
    double speedupWall = 0, speedupBackend = 0;
};

struct StateDiff {
    double maxAbsDiff = 0;
    long long intMismatches = 0;
};
StateDiff compareStates(const std::vector<PatientAoS>& a, const std::vector<PatientAoS>& b);

// Synthetic, deterministic snapshot of N patients in a realistic mix of states.
std::vector<PatientAoS> makeSyntheticPopulation(int n, uint64_t seed, const HospitalLayout& layout,
                                                float inactiveFraction, bool sortByZone);
std::vector<int32_t> activeIndexList(const std::vector<PatientAoS>& pop);

struct CpuRunResult {
    double ms = 0;
    uint64_t events = 0, eventHash = 0;
    std::vector<PatientAoS> finalState;
};
CpuRunResult runCpuReference(const std::vector<PatientAoS>& init, const UpdateParams& p, const RouteTable& rt,
                             int steps, int threads);

class Benchmark {
public:
    static std::vector<KernelBenchRow> runKernelSuite(const BenchmarkOptions& o);
    static std::vector<ScenarioBenchRow> runScenarioSuite(const BenchmarkOptions& o);
    static bool writeKernelCsv(const std::string& path, const std::vector<KernelBenchRow>& rows);
    static bool writeScenarioCsv(const std::string& path, const std::vector<ScenarioBenchRow>& rows);
};

// Decision-support: run the same scenario under two resource configurations.
enum class BackendKind : int { Cpu1 = 0, CpuMT, CudaNaive, CudaOptimized, CudaGraph, Count };
const char* backendKindName(BackendKind k);
std::unique_ptr<PatientBackend> makeBackend(BackendKind k, int blockSize, std::string* error);

struct ConfigRunResult {
    ScenarioConfig cfg;
    LiveMetrics metrics;
    double wallMs = 0;
    bool ok = false;
    std::string error;
};
ConfigRunResult runScenarioHeadless(const ScenarioConfig& cfg, BackendKind k, double maxHours = -1);

std::string systemInfoString();

}  // namespace hs

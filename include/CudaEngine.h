// CudaEngine.h - C++ facing interface of the CUDA simulation engine.
// (Pure C++ header: no CUDA types, so it can be included from any .cpp file.)
// When the project is built without CUDA, src/CudaStub.cpp implements these
// functions and reports the GPU as unavailable - nothing is silently emulated.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "PatientBackend.h"

namespace hs {

bool cudaBuildEnabled();
// Per-step CUDA event timing in the live backends (on by default). Disabling it removes
// 4-6 extra API calls per step; used to quantify the instrumentation overhead.
void setCudaEventTiming(bool enabled);

struct GpuDeviceInfo {
    bool available = false;
    std::string name, error;
    int ccMajor = 0, ccMinor = 0, smCount = 0;
    size_t totalMemBytes = 0;
    int driverVersion = 0, runtimeVersion = 0;
    int maxThreadsPerBlock = 0, warpSize = 0;
    size_t sharedMemPerBlock = 0, constMem = 0;
    int memBusWidth = 0, l2CacheBytes = 0;
};
GpuDeviceInfo queryGpu();

enum class CudaVariant : int {
    NaiveAoS = 0,      // AoS, 4 separate kernels, whole array copied H2D + D2H every step
    OptimizedSoA = 1,  // SoA, fused kernel, warp-aggregated event compaction, pinned memory
    OptimizedGraph = 2 // OptimizedSoA + per-step work replayed as one CUDA graph (launch overhead)
};
const char* cudaVariantName(CudaVariant v);

// Returns nullptr (and fills *error) if no usable CUDA device exists.
std::unique_ptr<PatientBackend> createCudaBackend(CudaVariant v, int blockSize = 256, std::string* error = nullptr);

// ---- kernel-level benchmark (used by Benchmark.cpp) ---------------------------------
enum class GpuKernelVariant : int {
    NaiveAoS = 0,     // AoS + 4 split kernels + per-patient bits; e2e copies whole arrays every step
    AoSFused,         // AoS + fused kernel + compaction
    SoASplit,         // SoA + 4 split kernels + compaction kernel
    SoAFused,         // SoA + fused kernel + warp-aggregated compaction  (= optimized)
    SoAFusedActive,   // SoA fused over an explicit active-patient index list
    Count
};
const char* gpuKernelVariantName(GpuKernelVariant v);

struct GpuRunConfig {
    GpuKernelVariant variant = GpuKernelVariant::SoAFused;
    int blockSize = 256;
    int steps = 200;
};

struct GpuRunResult {
    bool ok = false;
    std::string error;
    double kernelMs = 0;     // kernel-only run, CUDA events, all steps
    double h2dMs = 0;        // end-to-end run: host->device transfer time (CUDA events)
    double d2hMs = 0;        // end-to-end run: device->host transfer time (CUDA events)
    double endToEndMs = 0;   // wall clock: initial upload + all steps incl. per-step event transfer + final download
    uint64_t events = 0;
    uint64_t eventHash = 0;
    std::vector<PatientAoS> finalState;
};

GpuRunResult runGpuKernelBenchmark(const std::vector<PatientAoS>& init, const std::vector<int32_t>& activeList,
                                   const UpdateParams& params, const RouteTable& routes, const GpuRunConfig& cfg);

// Event-stream hash shared by CPU and GPU benchmark runs (events sorted by id).
inline uint64_t hashEvents(uint64_t h, uint64_t step, const SimEvent* ev, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint64_t v = (step << 40) ^ ((uint64_t)(uint32_t)ev[i].id << 8) ^ ev[i].bits ^ ((uint64_t)ev[i].priority << 4);
        h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    }
    return h;
}

}  // namespace hs

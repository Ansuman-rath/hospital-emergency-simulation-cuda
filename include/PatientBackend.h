// PatientBackend.h - interface for the parallel per-patient update engine.
//
// The SimulationEngine (CPU controller: arrivals, queues, resource allocation)
// talks to a backend that owns the "hot" patient arrays and performs the
// per-patient updates. Implementations:
//   * CpuBackend             - CPU baseline (single-thread or OpenMP)
//   * CudaBackend NaiveAoS   - CUDA, AoS layout, 4 separate kernels, full copies
//   * CudaBackend OptimizedSoA - CUDA, SoA, fused kernel, compacted events
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "Patient.h"

namespace hs {

struct BackendTimings {
    // last step
    double stepWallMs = 0;  // wall time of step() including transfers and synchronisation
    double computeMs = 0;   // CPU loop time or CUDA kernel time (CUDA events)
    double h2dMs = 0;       // host->device transfer time (CUDA events), 0 for CPU
    double d2hMs = 0;       // device->host transfer time (CUDA events), 0 for CPU
    // accumulated since init()
    double totalStepWallMs = 0, totalComputeMs = 0, totalH2DMs = 0, totalD2HMs = 0;
    double totalCommandMs = 0;  // applyCommands() wall time
    uint64_t steps = 0;
    uint64_t batches = 0;  // host synchronisations (== steps unless multi-step batching)
    uint64_t bytesH2D = 0, bytesD2H = 0;
};

// Parallel reduction over the hot arrays (used by the dashboard).
struct HotReduction {
    double sumWait = 0;      // over all patients that ever arrived
    float maxWait = 0;
    double sumHealth = 0;    // over currently active patients
    int activeCount = 0;
};

class PatientBackend {
public:
    virtual ~PatientBackend() = default;
    virtual std::string name() const = 0;
    virtual bool isGpu() const { return false; }

    virtual void init(const std::vector<PatientAoS>& initial, const UpdateParams& params,
                      const RouteTable& routes) = 0;
    virtual void applyCommands(const std::vector<PatientCommand>& cmds) = 0;
    // Advance all patients by one dt. Events are returned sorted by patient id.
    virtual void step(std::vector<SimEvent>& events) = 0;
    // Event-driven multi-step batching: advance up to maxSteps steps, stopping right after the
    // first step that produces events. Returns the number of steps executed (>= 1). The
    // default implementation loops step(); the optimized CUDA backend runs the whole batch
    // with a single host synchronisation.
    virtual int stepMulti(int maxSteps, std::vector<SimEvent>& events) {
        int done = 0;
        do {
            step(events);
            ++done;
        } while (done < maxSteps && events.empty());
        return done;
    }
    virtual void readRenderData(std::vector<RenderInstance>& out) = 0;
    virtual void readHotState(std::vector<PatientAoS>& out) = 0;  // full download (validation)
    virtual HotReduction reduce() = 0;
    // Read one patient's hot data (inspector panel). Cheap: a few small copies.
    virtual bool readPatient(int id, PatientAoS& out) = 0;

    // Optional CUDA-OpenGL interop (render buffer written directly by the GPU).
    virtual bool supportsInterop() const { return false; }
    virtual bool registerRenderBuffer(unsigned int /*glBuffer*/) { return false; }
    virtual bool writeRenderBufferInterop() { return false; }
    virtual void unregisterRenderBuffer() {}

    const BackendTimings& timings() const { return timings_; }
    int count() const { return n_; }

protected:
    BackendTimings timings_;
    int n_ = 0;
};

std::unique_ptr<PatientBackend> createCpuBackend(int threads);  // threads <= 1: single-threaded

}  // namespace hs

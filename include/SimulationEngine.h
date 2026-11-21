// SimulationEngine.h - hybrid CPU-GPU hospital simulation controller.
//
// Per step:
//   1. backend->step()        : parallel per-patient update (CPU loop or CUDA kernel)
//                               -> compact list of events (arrived / service done / escalated)
//   2. processEvents()        : CPU state machine reacts to events (sorted by id)
//   3. activateArrivals()     : CPU, patients whose arrival time has come
//   4. allocate()             : CPU, deterministic priority-ordered resource allocation
//   5. flushCommands()        : CPU -> backend, only for patients whose state changed
#pragma once
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "Hospital.h"
#include "PatientBackend.h"
#include "ResourceManager.h"
#include "Scenario.h"

namespace hs {

enum QueueId : int { Q_TRIAGE = 0, Q_DIAG, Q_ER, Q_ICU, Q_WARD, Q_COUNT };
const char* queueName(int q);
const char* stateName(int s);
const char* priorityName(int p);

struct QueueStats {
    int length = 0;
    double avgWait = 0;    // served + currently waiting, minutes
    double maxWait = 0;
    int served = 0;
};

struct LiveMetrics {
    double simTime = 0;  // minutes
    int total = 0, arrived = 0, active = 0, waiting = 0, inService = 0, inTreatment = 0, discharged = 0;
    int stateCounts[ST_COUNT] = {};
    int activeByPriority[kNumPriorities] = {};  // triaged active patients by current priority
    int untriaged = 0;
    int escalations = 0;
    QueueStats queues[Q_COUNT];
    int busy[R_COUNT] = {}, capacity[R_COUNT] = {};
    double utilization[R_COUNT] = {};  // time-averaged
    double avgWait = 0, maxWait = 0;   // from the backend reduction (GPU on CUDA backends)
    double avgWaitCpu = 0;             // from CPU queue timestamps (cross-check)
    double avgHealth = 0;
    double avgTreatmentDelay = 0;      // arrival -> start of ER treatment
    int treatmentStarted = 0;
    double avgLengthOfStay = 0;        // discharged patients
    double throughputPerHour = 0;
    int largestQueue = -1, slowestQueue = -1;  // bottleneck identification
};

struct TimeSeries {
    std::vector<float> t, queueTriage, queueER, queueBoarding, erOcc, icuOcc, wardOcc, discharged, active;
    void clear() { *this = TimeSeries(); }
};

struct EngineProfile {
    double backendMs = 0, controllerMs = 0, commandMs = 0;  // accumulated
    uint64_t steps = 0, events = 0, commands = 0, batches = 0;
};

struct Transition {
    int32_t id;
    int32_t from, to;
    float time;
};

class SimulationEngine {
public:
    void reset(const ScenarioConfig& cfg, std::unique_ptr<PatientBackend> backend);
    void step();
    // Event-driven batching: advance up to maxSteps steps in one backend call, but never past
    // a step that produces events, a scheduled arrival, a sampling point or the time limit,
    // i.e. only over steps in which the CPU controller has nothing to do. Returns steps done.
    int stepBatch(int maxSteps);
    bool batching = true;  // used by advance() and runToCompletion()
    // Advance by up to `simMinutes`, stopping early if `maxWallMs` is exceeded. Returns steps done.
    int advance(double simMinutes, double maxWallMs);
    void runToCompletion(double maxSimMinutes = -1);
    bool finished() const;

    double time() const { return time_; }
    uint64_t stepIndex() const { return stepIndex_; }
    const ScenarioConfig& config() const { return cfg_; }
    const HospitalLayout& layout() const { return layout_; }
    const std::vector<PatientRecord>& records() const { return rec_; }
    const ResourceManager& resources() const { return res_; }
    const PatientQueue& queue(int q) const { return queues_[q]; }
    PatientBackend* backend() { return backend_.get(); }
    const PatientBackend* backend() const { return backend_.get(); }
    const TimeSeries& series() const { return series_; }
    const EngineProfile& profile() const { return profile_; }

    LiveMetrics computeMetrics();
    bool checkInvariants(std::string* err) const;

    // Optional transition log (tests).
    bool logTransitions = false;
    std::vector<Transition> transitions;

private:
    void processEvent(const SimEvent& e);
    void activateArrivals();
    void allocate();
    void flushCommands();
    void setState(PatientRecord& r, int32_t s);
    void routeTo(PatientRecord& r, int dept, Vec2 target);
    void enqueue(PatientRecord& r, int q, int zone);
    void dequeue(PatientRecord& r, int q);
    void discharge(PatientRecord& r);
    void releaseRes(PatientRecord& r, int pool, int32_t& slot);
    void markDirty(PatientRecord& r, int mask = 0);
    void sampleSeries();
    void afterBackend(int steps, std::chrono::steady_clock::time_point t0, std::chrono::steady_clock::time_point t1);
    int quietSteps(int limit) const;
    double stepTime(uint64_t s) const { return (double)s * (double)cfg_.dt; }
    uint64_t sampleEvery() const;

    ScenarioConfig cfg_;
    HospitalLayout layout_;
    ResourceManager res_;
    PatientQueue queues_[Q_COUNT];
    SlotAllocator zones_[Z_COUNT];
    std::unique_ptr<PatientBackend> backend_;
    std::vector<PatientRecord> rec_;
    std::vector<int32_t> dirty_;
    std::vector<PatientCommand> cmds_;
    std::vector<SimEvent> events_;
    double time_ = 0;
    uint64_t stepIndex_ = 0;
    int nextArrival_ = 0;
    int64_t seq_ = 0;
    int stateCounts_[ST_COUNT] = {};
    double queueWaitTotal_[Q_COUNT] = {};
    double queueWaitMax_[Q_COUNT] = {};
    int queueServed_[Q_COUNT] = {};
    double sumTreatDelay_ = 0, sumLos_ = 0;
    int treatStarted_ = 0, escalations_ = 0;
    TimeSeries series_;
    EngineProfile profile_;
};

}  // namespace hs

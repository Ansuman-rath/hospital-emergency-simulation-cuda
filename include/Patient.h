// Patient.h - core patient data model shared by the CPU and CUDA code paths.
//
// The "hot" per-patient fields (position, waiting time, remaining service time,
// health, state, priority ...) are the ones updated every simulation step. They
// are stored either as an Array-of-Structures (PatientAoS) or as a
// Structure-of-Arrays (PatientSoA) and live in GPU memory when a CUDA backend
// is used. "Cold" bookkeeping (arrival time, assigned resources, timestamps)
// lives only on the CPU in PatientRecord.
//
// NOTE: this is a teaching / analysis simulation. Priorities, durations and
// deterioration rates are synthetic model parameters, not a clinical protocol.
#pragma once
#include <cstdint>

#if defined(__CUDACC__)
#define HS_HD __host__ __device__ __forceinline__
#else
#define HS_HD inline
#endif

namespace hs {

constexpr int kNumPriorities = 4;
enum PriorityLevel : int32_t { P1 = 0, P2 = 1, P3 = 2, P4 = 3 };

// Patient state machine. Stored as int32 so the same value lives in GPU arrays.
enum PatientState : int32_t {
    ST_NOT_ARRIVED = 0,  // scheduled, not yet in the hospital (inactive)
    ST_ARRIVING,         // walking from the ambulance bay / door to the triage queue
    ST_WAIT_TRIAGE,      // queued for a triage nurse
    ST_IN_TRIAGE,        // being triaged
    ST_WAIT_DIAG,        // queued for the diagnostic / laboratory area
    ST_IN_DIAG,          // diagnostics in progress
    ST_WAIT_ER,          // queued for an ER bed + doctor
    ST_IN_TREATMENT,     // treated in the ER (holds a doctor and an ER bed)
    ST_WAIT_ICU,         // treatment finished, boarding in ER bed until ICU bed frees
    ST_WAIT_WARD,        // boarding (ER or ICU bed) until a ward bed frees
    ST_IN_ICU,           // ICU stay
    ST_IN_WARD,          // general ward stay
    ST_DISCHARGING,      // walking to the discharge area / exit
    ST_DISCHARGED,       // left the hospital (inactive)
    ST_COUNT
};

constexpr uint32_t stBit(int s) { return 1u << s; }
// States in which waiting time accumulates (one-to-one with the CPU queues).
constexpr uint32_t kWaitMask = stBit(ST_WAIT_TRIAGE) | stBit(ST_WAIT_DIAG) | stBit(ST_WAIT_ER) |
                               stBit(ST_WAIT_ICU) | stBit(ST_WAIT_WARD);
// States in which a service timer counts down (once the patient has reached the location).
constexpr uint32_t kServiceMask = stBit(ST_IN_TRIAGE) | stBit(ST_IN_DIAG) | stBit(ST_IN_TREATMENT) |
                                  stBit(ST_IN_ICU) | stBit(ST_IN_WARD);
// Untreated states in which the synthetic health score deteriorates.
constexpr uint32_t kDeteriorateMask = stBit(ST_ARRIVING) | stBit(ST_WAIT_TRIAGE) | stBit(ST_WAIT_DIAG) |
                                      stBit(ST_WAIT_ER);
// Cared-for states in which health recovers.
constexpr uint32_t kRecoverMask = stBit(ST_IN_TREATMENT) | stBit(ST_IN_ICU) | stBit(ST_IN_WARD) |
                                  stBit(ST_WAIT_ICU) | stBit(ST_WAIT_WARD);
constexpr uint32_t kAllStatesMask = (1u << ST_COUNT) - 1u;
constexpr uint32_t kActiveMask = kAllStatesMask & ~(stBit(ST_NOT_ARRIVED) | stBit(ST_DISCHARGED));

enum PatientFlags : int32_t {
    FLAG_MOVING = 1,     // following a path
    FLAG_TRIAGED = 2,    // priority is known (triage done or pre-triaged by ambulance)
    FLAG_AMBULANCE = 4,  // arrived by ambulance
};

// Bits reported by the per-patient update to the CPU controller.
enum EventBits : uint32_t {
    EV_ARRIVED = 1,       // reached the final point of its path
    EV_SERVICE_DONE = 2,  // service timer reached zero
    EV_ESCALATED = 4,     // deterioration crossed the escalation threshold, priority raised
};

enum Disposition : int32_t { DISP_HOME = 0, DISP_ICU = 1, DISP_WARD = 2 };

// ---------------------------------------------------------------- hot data ----
struct PatientAoS {
    float x, z;              // current position (metres, floor plane)
    float tx, tz;            // personal destination (bed, desk, queue slot)
    float speed;             // walking speed (m / sim-minute)
    float waitingTime;       // accumulated queue waiting time (sim-minutes)
    float serviceRemaining;  // remaining triage/diagnostic/treatment/stay time
    float health;            // synthetic acuity score 0..100 (lower = worse)
    int32_t state;           // PatientState
    int32_t priority;        // current priority (0 = P1 critical)
    int32_t route;           // index into the route table (-1 = direct)
    int32_t waypoint;        // next waypoint index in the route
    int32_t flags;           // PatientFlags
};
static_assert(sizeof(PatientAoS) == 52, "unexpected PatientAoS padding");

// Structure of Arrays view (host or device pointers).
struct PatientSoA {
    float* x = nullptr;
    float* z = nullptr;
    float* tx = nullptr;
    float* tz = nullptr;
    float* speed = nullptr;
    float* waitingTime = nullptr;
    float* serviceRemaining = nullptr;
    float* health = nullptr;
    int32_t* state = nullptr;
    int32_t* priority = nullptr;
    int32_t* route = nullptr;
    int32_t* waypoint = nullptr;
    int32_t* flags = nullptr;
};

// Compact event record produced by the update step (GPU -> CPU).
struct SimEvent {
    int32_t id;
    uint32_t bits;
    int32_t priority;     // priority after this step (valid for EV_ESCALATED)
    float waitingTime;    // accumulated waiting time after this step
};

// CPU -> backend state change for one patient.
enum CommandMask : int32_t { CMD_INIT = 1, CMD_ROUTE = 2 };
struct PatientCommand {
    int32_t id;
    int32_t mask;            // CommandMask
    int32_t state;
    int32_t priority;
    int32_t flags;
    int32_t route;           // used when CMD_ROUTE (waypoint reset to 0)
    float tx, tz;            // used when CMD_ROUTE
    float serviceRemaining;
    float x, z, health, speed;  // used when CMD_INIT (waitingTime reset to 0)
};

// Instance data for rendering: position + packed (state | priority<<8 | flags<<16).
struct RenderInstance {
    float x, z;
    uint32_t packed;
};
HS_HD uint32_t packRender(int32_t state, int32_t priority, int32_t flags) {
    return (uint32_t)(state & 0xFF) | ((uint32_t)(priority & 0xFF) << 8) | ((uint32_t)(flags & 0xFF) << 16);
}

// ------------------------------------------------------------- route table ----
// Routes between departments: door -> corridor -> corridor -> door (max 4 points),
// followed implicitly by the patient's personal target (tx, tz).
constexpr int kMaxRoutes = 81;  // D_COUNT * D_COUNT (see Hospital.h)
constexpr int kRouteLen = 4;
struct RouteTable {
    float x[kMaxRoutes][kRouteLen];
    float z[kMaxRoutes][kRouteLen];
    int32_t len[kMaxRoutes];
};

// Model parameters for the per-patient update (constant memory on the GPU).
struct UpdateParams {
    float dt = 0.05f;                                     // sim-minutes per step
    float decay[kNumPriorities] = {0.40f, 0.20f, 0.08f, 0.04f};  // health points / minute while untreated
    float escalateBelow[kNumPriorities] = {0.f, 20.f, 30.f, 40.f};  // escalate P(k) -> P(k-1) below this
    float recoverRate = 0.30f;                            // health points / minute while cared for
    float minHealth = 5.0f;
};

// --------------------------------------------------------------- cold data ----
struct PatientRecord {
    // generated attributes (deterministic from the scenario seed)
    int32_t id = 0;
    float arrivalTime = 0;       // sim-minutes
    int32_t truePriority = P4;   // priority revealed at triage
    bool ambulance = false;
    bool needsDiag = false;
    int32_t disposition = DISP_HOME;
    bool icuStepDown = false;    // after ICU: go to ward (true) or home (false)
    float triageDur = 0, diagDur = 0, treatDur = 0, icuDur = 0, wardDur = 0;
    float initHealth = 100, speed = 60;

    // dynamic bookkeeping (CPU controller)
    int32_t state = ST_NOT_ARRIVED;
    int32_t priority = P4;
    int32_t dept = 0;            // department the patient is in / heading to
    int32_t doctor = -1, erBed = -1, icuBed = -1, wardBed = -1, triageDesk = -1, labSlot = -1;
    int32_t zone = -1, zoneSlot = -1;
    double enqueueTime = 0;
    double cpuWaitTotal = 0;     // waiting time computed from queue timestamps
    double treatmentStart = -1, treatmentEnd = -1, dischargeTime = -1;
    bool treatmentDone = false;
    int32_t escalations = 0;

    // mirror of the fields last commanded to the backend
    int32_t flags = 0, route = -1;
    float tx = 0, tz = 0, svc = 0;
    int32_t cmdMask = 0;
    bool dirty = false;
};

}  // namespace hs

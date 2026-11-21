// Scenario.h - emergency scenario configuration and presets.
#pragma once
#include <cstdint>
#include <string>

#include "Hospital.h"
#include "Patient.h"

namespace hs {

enum class ScenarioType : int { Normal = 0, PatientSurge, AmbulanceSurge, MassCasualty, Count };
const char* scenarioName(ScenarioType t);

struct ScenarioConfig {
    ScenarioType type = ScenarioType::Normal;
    uint64_t seed = 20261111;
    int totalPatients = 300;
    float arrivalRatePerHour = 14.f;  // base walk-in / ambulance arrival rate (Poisson)
    float priorityMix[kNumPriorities] = {0.05f, 0.20f, 0.45f, 0.30f};
    float ambulanceFraction = 0.15f;  // share of base arrivals that come by ambulance
    Capacities cap;

    // service times (sim-minutes, means)
    float avgTreatmentMin = 45.f;
    float treatmentScale[kNumPriorities] = {1.6f, 1.25f, 0.9f, 0.5f};
    float triageMin = 6.f, diagMin = 25.f, icuStayMin = 360.f, wardStayMin = 480.f;

    // disposition / pathway probabilities per priority
    float pDiag[kNumPriorities] = {0.30f, 0.50f, 0.40f, 0.10f};
    float pICU[kNumPriorities] = {0.55f, 0.15f, 0.02f, 0.00f};
    float pWard[kNumPriorities] = {0.35f, 0.40f, 0.18f, 0.03f};
    float icuStepDown = 0.6f;

    // Scenario 2: rate surge
    float surgeStartMin = 120.f, surgeDurationMin = 180.f, surgeMultiplier = 1.f;
    // Scenario 3: ambulance bursts
    int burstCount = 0, burstSize = 0;
    float burstStartMin = 60.f, burstIntervalMin = 45.f, burstWindowMin = 10.f;
    float burstMix[kNumPriorities] = {0.25f, 0.40f, 0.25f, 0.10f};
    // Scenario 4: mass-casualty incident
    int mciPatients = 0;
    float mciTimeMin = 60.f, mciWindowMin = 30.f;
    float mciMix[kNumPriorities] = {0.20f, 0.35f, 0.30f, 0.15f};

    // simulation
    float dt = 0.05f;          // sim-minutes per step (3 s)
    float walkSpeed = 60.f;    // metres per sim-minute (1 m/s)
    float maxSimHours = 96.f;  // safety stop
    UpdateParams model;        // deterioration / escalation model (dt is overwritten)
};

ScenarioConfig makePreset(ScenarioType t);
// Scale a scenario to a different patient count (arrivals and resources scale
// proportionally). Used by the large-scale scenario benchmark.
ScenarioConfig scaleScenario(const ScenarioConfig& base, int patients);

}  // namespace hs

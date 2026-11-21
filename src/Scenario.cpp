#include "Scenario.h"

#include <algorithm>
#include <cmath>

namespace hs {

const char* scenarioName(ScenarioType t) {
    switch (t) {
        case ScenarioType::Normal: return "Normal Hospital Operation";
        case ScenarioType::PatientSurge: return "Patient Surge";
        case ScenarioType::AmbulanceSurge: return "Ambulance Surge";
        case ScenarioType::MassCasualty: return "Mass-Casualty Event";
        default: return "Custom";
    }
}

ScenarioConfig makePreset(ScenarioType t) {
    ScenarioConfig c;
    c.type = t;
    switch (t) {
        // Calibration: mean ER treatment ~ 40 min (45 min x priority-mix scale 0.885),
        // so 8 doctors serve ~12 patients/h. 9 arrivals/h -> ~75% doctor load at baseline.
        case ScenarioType::Normal:
            c.totalPatients = 220;
            c.arrivalRatePerHour = 9.f;
            break;
        case ScenarioType::PatientSurge:  // 4x walk-in rate for 3 hours
            c.totalPatients = 300;
            c.arrivalRatePerHour = 9.f;
            c.surgeStartMin = 120.f;
            c.surgeDurationMin = 180.f;
            c.surgeMultiplier = 4.f;
            break;
        case ScenarioType::AmbulanceSurge:  // 6 waves of 15 ambulance patients
            c.totalPatients = 270;
            c.arrivalRatePerHour = 8.f;
            c.burstCount = 6;
            c.burstSize = 15;
            c.burstStartMin = 60.f;
            c.burstIntervalMin = 45.f;
            c.burstWindowMin = 10.f;
            break;
        case ScenarioType::MassCasualty:  // 180 casualties within 30 minutes
            c.totalPatients = 400;
            c.arrivalRatePerHour = 8.f;
            c.mciPatients = 180;
            c.mciTimeMin = 60.f;
            c.mciWindowMin = 30.f;
            break;
        default: break;
    }
    return c;
}

ScenarioConfig scaleScenario(const ScenarioConfig& base, int patients) {
    ScenarioConfig c = base;
    const double f = (double)patients / std::max(1, base.totalPatients);
    auto sc = [f](int v) { return std::max(1, (int)std::lround(v * f)); };
    c.totalPatients = patients;
    c.arrivalRatePerHour = (float)(base.arrivalRatePerHour * f);
    c.burstSize = base.burstSize > 0 ? sc(base.burstSize) : 0;
    c.mciPatients = base.mciPatients > 0 ? sc(base.mciPatients) : 0;
    c.cap.doctors = sc(base.cap.doctors);
    c.cap.nurses = sc(base.cap.nurses);
    c.cap.erBeds = sc(base.cap.erBeds);
    c.cap.icuBeds = sc(base.cap.icuBeds);
    c.cap.wardBeds = sc(base.cap.wardBeds);
    c.cap.labSlots = sc(base.cap.labSlots);
    return c;
}

}  // namespace hs

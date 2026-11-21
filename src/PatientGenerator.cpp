#include "PatientGenerator.h"

#include <algorithm>
#include <cmath>

namespace hs {

int PatientGenerator::samplePriority(Rng& rng, const float mix[kNumPriorities]) {
    double total = 0;
    for (int i = 0; i < kNumPriorities; ++i) total += std::max(0.f, mix[i]);
    if (total <= 0) return P4;
    double u = rng.uniform() * total, acc = 0;
    for (int i = 0; i < kNumPriorities; ++i) {
        acc += std::max(0.f, mix[i]);
        if (u < acc) return i;
    }
    return P4;
}

namespace {
struct ArrivalSpec {
    double time;
    int kind;  // 0 = base stream, 1 = ambulance burst, 2 = mass-casualty
    int order;
};
}  // namespace

std::vector<PatientRecord> PatientGenerator::generate(const ScenarioConfig& cfg) {
    const int total = std::max(0, cfg.totalPatients);
    Rng rng(cfg.seed);
    std::vector<ArrivalSpec> arr;
    arr.reserve(total);

    // Mass-casualty arrivals.
    for (int i = 0; i < cfg.mciPatients && (int)arr.size() < total; ++i)
        arr.push_back({cfg.mciTimeMin + rng.uniform() * cfg.mciWindowMin, 2, (int)arr.size()});
    // Ambulance bursts.
    for (int b = 0; b < cfg.burstCount; ++b)
        for (int k = 0; k < cfg.burstSize && (int)arr.size() < total; ++k)
            arr.push_back({cfg.burstStartMin + b * cfg.burstIntervalMin + rng.uniform() * cfg.burstWindowMin, 1,
                           (int)arr.size()});
    // Base non-homogeneous Poisson stream (thinning) with an optional surge window.
    const double base = std::max(1e-6, (double)cfg.arrivalRatePerHour / 60.0);  // per minute
    const double mult = std::max(1.0, (double)cfg.surgeMultiplier);
    const double lambdaMax = base * mult;
    double t = 0;
    while ((int)arr.size() < total) {
        t += -std::log(1.0 - rng.uniform()) / lambdaMax;
        const bool inSurge = t >= cfg.surgeStartMin && t < cfg.surgeStartMin + cfg.surgeDurationMin;
        const double lambda = inSurge ? base * cfg.surgeMultiplier : base;
        if (rng.uniform() * lambdaMax <= lambda) arr.push_back({t, 0, (int)arr.size()});
    }
    std::sort(arr.begin(), arr.end(), [](const ArrivalSpec& a, const ArrivalSpec& b) {
        return a.time != b.time ? a.time < b.time : a.order < b.order;
    });

    std::vector<PatientRecord> out(total);
    for (int id = 0; id < total; ++id) {
        // Independent stream per patient -> attributes do not depend on generation order.
        Rng pr(hashU64(cfg.seed ^ (0xA5A5A5A5ull + (uint64_t)id * 0x100000001B3ull)));
        PatientRecord& r = out[id];
        r.id = id;
        r.arrivalTime = (float)arr[id].time;
        const float* mix = arr[id].kind == 2 ? cfg.mciMix : arr[id].kind == 1 ? cfg.burstMix : cfg.priorityMix;
        r.truePriority = samplePriority(pr, mix);
        r.ambulance = arr[id].kind != 0 || pr.uniform() < cfg.ambulanceFraction;
        const int p = r.truePriority;
        r.needsDiag = pr.uniform() < cfg.pDiag[p];
        const double u = pr.uniform();
        r.disposition = u < cfg.pICU[p] ? DISP_ICU : (u < cfg.pICU[p] + cfg.pWard[p] ? DISP_WARD : DISP_HOME);
        r.icuStepDown = pr.uniform() < cfg.icuStepDown;
        r.triageDur = cfg.triageMin * pr.uniformf(0.6f, 1.4f);
        r.diagDur = cfg.diagMin * pr.uniformf(0.5f, 1.5f);
        r.treatDur = cfg.avgTreatmentMin * cfg.treatmentScale[p] * pr.uniformf(0.4f, 1.6f);
        r.icuDur = cfg.icuStayMin * pr.uniformf(0.5f, 1.5f);
        r.wardDur = cfg.wardStayMin * pr.uniformf(0.5f, 1.5f);
        static const float hLo[kNumPriorities] = {30, 45, 60, 75}, hHi[kNumPriorities] = {45, 65, 80, 95};
        r.initHealth = pr.uniformf(hLo[p], hHi[p]);
        r.speed = cfg.walkSpeed * pr.uniformf(0.75f, 1.25f);
        r.priority = p;
    }
    return out;
}

}  // namespace hs

// test_main.cpp - unit and integration tests (minimal self-contained framework).
//
// Numerical tolerance for CPU <-> GPU comparison: identical code paths are compiled
// with --fmad=false (CUDA) and /fp:precise (MSVC), so we EXPECT bit-identical results
// and assert exact equality of all integer fields, event streams and outcomes.
// Floating-point fields are additionally checked against kFloatTol (1e-4) so a
// different toolchain that contracts FMAs would still be diagnosed clearly.
#include <cmath>
#include <cstdio>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include "Benchmark.h"
#include "CudaEngine.h"
#include "PatientGenerator.h"
#include "PatientUpdate.h"
#include "ResourceManager.h"
#include "SimulationEngine.h"

using namespace hs;

static int g_pass = 0, g_fail = 0;
static constexpr double kFloatTol = 1e-4;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            std::printf("    FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);      \
            ++g_fail;                                                         \
        } else {                                                              \
            ++g_pass;                                                         \
        }                                                                     \
    } while (0)

struct TestCase {
    const char* name;
    std::function<void()> fn;
};
static std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}
struct Reg {
    Reg(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};
#define TEST(name)                         \
    static void name();                    \
    static Reg reg_##name(#name, name);    \
    static void name()

static bool gpuAvailable() {
    static int a = -1;
    if (a < 0) a = queryGpu().available ? 1 : 0;
    return a == 1;
}

// ------------------------------------------------------------- generation ----
TEST(patient_generation_deterministic) {
    ScenarioConfig c = makePreset(ScenarioType::MassCasualty);
    auto a = PatientGenerator::generate(c), b = PatientGenerator::generate(c);
    CHECK((int)a.size() == c.totalPatients);
    bool same = a.size() == b.size(), sorted = true, ids = true;
    for (size_t i = 0; i < a.size(); ++i) {
        same &= a[i].arrivalTime == b[i].arrivalTime && a[i].truePriority == b[i].truePriority &&
                a[i].treatDur == b[i].treatDur && a[i].disposition == b[i].disposition;
        if (i) sorted &= a[i - 1].arrivalTime <= a[i].arrivalTime;
        ids &= a[i].id == (int)i;
    }
    CHECK(same);
    CHECK(sorted);
    CHECK(ids);
    c.seed += 1;
    auto d = PatientGenerator::generate(c);
    int diff = 0;
    for (size_t i = 0; i < a.size(); ++i) diff += a[i].arrivalTime != d[i].arrivalTime;
    CHECK(diff > (int)a.size() / 2);
    // MCI arrivals concentrated in the incident window
    int inWindow = 0;
    for (auto& r : a) inWindow += r.arrivalTime >= c.mciTimeMin && r.arrivalTime <= c.mciTimeMin + c.mciWindowMin;
    CHECK(inWindow >= c.mciPatients);
}

TEST(priority_assignment_follows_mix) {
    Rng rng(7);
    const float mix[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    int cnt[4] = {};
    const int N = 200000;
    for (int i = 0; i < N; ++i) cnt[PatientGenerator::samplePriority(rng, mix)]++;
    for (int p = 0; p < 4; ++p) CHECK(std::fabs(cnt[p] / (double)N - mix[p]) < 0.01);
    const float onlyP1[4] = {1.f, 0.f, 0.f, 0.f};
    bool allP1 = true;
    for (int i = 0; i < 1000; ++i) allP1 &= PatientGenerator::samplePriority(rng, onlyP1) == P1;
    CHECK(allP1);
    // generated attributes are in valid ranges
    auto recs = PatientGenerator::generate(makePreset(ScenarioType::Normal));
    bool ok = true;
    for (auto& r : recs)
        ok &= r.truePriority >= 0 && r.truePriority < 4 && r.treatDur > 0 && r.initHealth > 0 && r.initHealth <= 100;
    CHECK(ok);
}

// ---------------------------------------------------------- queues / pools ----
TEST(queue_ordering_priority_then_fifo) {
    PatientQueue q;
    q.init(10);
    q.push(0, 2, 0);
    q.push(1, 0, 1);
    q.push(2, 2, 2);
    q.push(3, 1, 3);
    q.push(4, 0, 4);
    std::vector<int> order;
    PatientQueue q2 = q;
    while (!q2.empty()) order.push_back(q2.popFront());
    CHECK((order == std::vector<int>{1, 4, 3, 0, 2}));
    q.updatePriority(2, 0);  // escalation keeps original FIFO sequence
    std::vector<int> order2;
    while (!q.empty()) order2.push_back(q.popFront());
    CHECK((order2 == std::vector<int>{1, 2, 4, 3, 0}));
    bool threw = false;
    q.push(5, 1, 9);
    try {
        q.push(5, 1, 10);
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(resource_allocation_and_release) {
    ResourcePool p;
    p.init("beds", 3);
    CHECK(p.acquire(10) == 0);
    CHECK(p.acquire(11) == 1);
    CHECK(p.acquire(12) == 2);
    CHECK(p.acquire(13) == -1);  // full: patient must wait
    CHECK(p.available() == 0);
    bool threw = false;
    try {
        p.release(1, 99);  // not the occupant
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw);
    p.release(1, 11);
    CHECK(p.occupant(1) == -1);
    CHECK(p.acquire(14) == 1);  // lowest free slot -> deterministic
    threw = false;
    try {
        p.release(1, 11);  // double release
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw);
    ResourceManager rm;
    rm.init(Capacities{});
    CHECK(rm.validate(nullptr));
}

// ------------------------------------------------------ per-patient rules ----
TEST(update_rules_treatment_movement_escalation) {
    HospitalLayout L;
    L.build(Capacities{});
    UpdateParams p;
    p.dt = 0.5f;
    // service: counts down, fires exactly once
    float rem = 1.2f;
    int fired = 0;
    for (int i = 0; i < 10; ++i) fired += serviceStep(ST_IN_TREATMENT, 0, rem, p.dt) == EV_SERVICE_DONE;
    CHECK(fired == 1);
    CHECK(rem == 0.0f);
    // no progress while still walking to the bed
    rem = 1.f;
    serviceStep(ST_IN_TREATMENT, FLAG_MOVING, rem, p.dt);
    CHECK(rem == 1.f);
    // movement along a route ends exactly at the target, arrival fires once
    float x = -55, z = 0;
    int wp = 0, flags = FLAG_MOVING, arrived = 0;
    const int route = HospitalLayout::routeId(D_ENTRANCE, D_ER);
    for (int i = 0; i < 200; ++i)
        arrived += moveStep(x, z, -10.f, -20.f, 60.f, route, wp, flags, p.dt, L.routes()) == EV_ARRIVED;
    CHECK(arrived == 1);
    CHECK(x == -10.f && z == -20.f);
    CHECK((flags & FLAG_MOVING) == 0);
    // waiting accumulates only in waiting states
    float w = 0;
    waitStep(ST_WAIT_ER, w, p.dt);
    waitStep(ST_IN_TREATMENT, w, p.dt);
    CHECK(w == 0.5f);
    // escalation P3 -> P2 when health drops below threshold (triaged only)
    int prio = P3;
    float h = p.escalateBelow[P3] + 0.01f;
    CHECK(healthStep(ST_WAIT_ER, 0, prio, h, p) == 0u);  // not triaged: no escalation
    h = p.escalateBelow[P3] + 0.01f;
    CHECK(healthStep(ST_WAIT_ER, FLAG_TRIAGED, prio, h, p) == EV_ESCALATED);
    CHECK(prio == P2);
    // inactive patients are never touched
    PatientAoS q{1, 2, 3, 4, 60, 0, 5, 50, ST_DISCHARGED, 0, -1, 0, FLAG_MOVING};
    CHECK(updatePatientAoS(q, p, L.routes()) == 0u);
    CHECK(q.x == 1 && q.serviceRemaining == 5);
}

// -------------------------------------------------- full simulation runs ----
static const std::set<std::pair<int, int>>& allowedTransitions() {
    static const std::set<std::pair<int, int>> t = {
        {ST_NOT_ARRIVED, ST_ARRIVING},   {ST_NOT_ARRIVED, ST_WAIT_ER},   {ST_ARRIVING, ST_WAIT_TRIAGE},
        {ST_WAIT_TRIAGE, ST_IN_TRIAGE},  {ST_IN_TRIAGE, ST_WAIT_DIAG},   {ST_IN_TRIAGE, ST_WAIT_ER},
        {ST_WAIT_DIAG, ST_IN_DIAG},      {ST_IN_DIAG, ST_WAIT_ER},       {ST_WAIT_ER, ST_IN_TREATMENT},
        {ST_IN_TREATMENT, ST_WAIT_ICU},  {ST_IN_TREATMENT, ST_WAIT_WARD}, {ST_IN_TREATMENT, ST_DISCHARGING},
        {ST_WAIT_ICU, ST_IN_ICU},        {ST_WAIT_WARD, ST_IN_WARD},     {ST_IN_ICU, ST_WAIT_WARD},
        {ST_IN_ICU, ST_DISCHARGING},     {ST_IN_WARD, ST_DISCHARGING},   {ST_DISCHARGING, ST_DISCHARGED}};
    return t;
}

TEST(state_machine_full_run_cpu) {
    ScenarioConfig c = makePreset(ScenarioType::Normal);
    c.totalPatients = 150;
    c.cap = {12, 6, 20, 30, 80, 6};
    SimulationEngine e;
    e.logTransitions = true;
    e.reset(c, createCpuBackend(1));
    std::string err;
    bool inv = true, workConserving = true;
    while (!e.finished()) {
        e.step();
        if (e.stepIndex() % 20 == 0) inv &= e.checkInvariants(&err);
        const auto& res = e.resources();
        if (!e.queue(Q_ER).empty() && res.pool(R_DOCTOR).available() > 0 && res.pool(R_ER_BED).available() > 0)
            workConserving = false;
    }
    if (!inv) std::printf("    invariant: %s\n", err.c_str());
    CHECK(inv);
    CHECK(workConserving);
    CHECK(e.checkInvariants(&err));
    const LiveMetrics m = e.computeMetrics();
    CHECK(m.discharged == c.totalPatients);
    bool legal = true;
    for (const Transition& t : e.transitions) legal &= allowedTransitions().count({t.from, t.to}) == 1;
    CHECK(legal);
    bool treatedBeforeDischarge = true, treatedOnce = true;
    for (const auto& r : e.records()) {
        treatedBeforeDischarge &= r.treatmentDone && r.treatmentStart >= 0 && r.dischargeTime >= r.treatmentEnd;
        treatedOnce &= r.treatmentEnd >= r.treatmentStart;
    }
    CHECK(treatedBeforeDischarge);
    CHECK(treatedOnce);
}

TEST(constrained_mass_casualty_invariants) {
    ScenarioConfig c = makePreset(ScenarioType::MassCasualty);
    c.cap = {4, 2, 6, 2, 10, 2};  // deliberately scarce
    c.maxSimHours = 400;
    SimulationEngine e;
    e.reset(c, createCpuBackend(1));
    std::string err;
    bool inv = true;
    int peakQueue = 0;
    while (!e.finished()) {
        e.step();
        peakQueue = std::max(peakQueue, (int)e.queue(Q_ER).size());
        if (e.stepIndex() % 50 == 0) inv &= e.checkInvariants(&err);
        // capacity never exceeded
        for (int r = 0; r < R_COUNT; ++r) inv &= e.resources().pool(r).busy() <= e.resources().pool(r).capacity();
    }
    if (!inv) std::printf("    invariant: %s\n", err.c_str());
    CHECK(inv);
    CHECK(peakQueue > 20);  // the scenario really creates pressure
    CHECK(e.computeMetrics().discharged == c.totalPatients);
    CHECK(e.computeMetrics().escalations > 0);
}

TEST(waiting_time_consistency) {
    // GPU-style accumulated waiting time (hot array) == sum of CPU queue timestamps.
    ScenarioConfig c = makePreset(ScenarioType::PatientSurge);
    SimulationEngine e;
    e.reset(c, createCpuBackend(1));
    e.runToCompletion();
    std::vector<PatientAoS> hot;
    e.backend()->readHotState(hot);
    double worst = 0;
    for (const auto& r : e.records()) {
        const double d = std::fabs(hot[r.id].waitingTime - r.cpuWaitTotal);
        worst = std::max(worst, d / (1.0 + r.cpuWaitTotal));
    }
    std::printf("    max relative waiting-time deviation: %.2e\n", worst);
    CHECK(worst < 1e-3);  // float accumulation of dt vs double timestamps
    const LiveMetrics m = e.computeMetrics();
    CHECK(std::fabs(m.avgWait - m.avgWaitCpu) < 1e-2 * (1 + m.avgWaitCpu));
}

static bool sameRecords(const SimulationEngine& a, const SimulationEngine& b) {
    for (size_t i = 0; i < a.records().size(); ++i) {
        const auto &x = a.records()[i], &y = b.records()[i];
        if (x.state != y.state || x.treatmentStart != y.treatmentStart || x.dischargeTime != y.dischargeTime ||
            x.priority != y.priority || x.erBed != y.erBed || x.escalations != y.escalations ||
            x.cpuWaitTotal != y.cpuWaitTotal)
            return false;
    }
    return true;
}

TEST(cpu_single_vs_multithread_identical) {
    ScenarioConfig c = makePreset(ScenarioType::AmbulanceSurge);
    SimulationEngine a, b;
    a.reset(c, createCpuBackend(1));
    b.reset(c, createCpuBackend(8));
    a.runToCompletion();
    b.runToCompletion();
    CHECK(sameRecords(a, b));
}

TEST(event_batching_identical) {
    // Event-driven multi-step batching must not change any outcome (CPU and GPU).
    for (ScenarioType t : {ScenarioType::Normal, ScenarioType::AmbulanceSurge}) {
        ScenarioConfig c = makePreset(t);
        SimulationEngine ref, cpuB, gpuB;
        ref.batching = false;
        ref.reset(c, createCpuBackend(1));
        ref.runToCompletion();
        cpuB.reset(c, createCpuBackend(1));
        cpuB.runToCompletion();
        CHECK(sameRecords(ref, cpuB));
        CHECK(cpuB.profile().batches < ref.profile().batches);
        CHECK(cpuB.series().t.size() == ref.series().t.size());
        if (gpuAvailable()) {
            gpuB.reset(c, createCudaBackend(CudaVariant::OptimizedSoA, 256));
            gpuB.runToCompletion();
            CHECK(sameRecords(ref, gpuB));
            std::vector<PatientAoS> a, b;
            ref.backend()->readHotState(a);
            gpuB.backend()->readHotState(b);
            const StateDiff d = compareStates(a, b);
            CHECK(d.intMismatches == 0 && d.maxAbsDiff == 0.0);
            std::printf("    %s: %llu steps, %llu host syncs batched (vs %llu per-step)\n", scenarioName(t),
                        (unsigned long long)gpuB.profile().steps,
                        (unsigned long long)gpuB.backend()->timings().batches,
                        (unsigned long long)ref.profile().batches);
        }
    }
}

// ------------------------------------------------------------------- GPU ----
TEST(gpu_kernels_match_cpu_all_variants) {
    if (!gpuAvailable()) {
        std::printf("    SKIPPED: %s\n", queryGpu().error.c_str());
        return;
    }
    HospitalLayout L;
    L.build(Capacities{});
    UpdateParams p;
    p.dt = 0.02f;
    for (int n : {1, 31, 33, 1000, 4097, 20000}) {  // includes sizes that are not block multiples
        const auto pop = makeSyntheticPopulation(n, 99 + n, L, 0.3f, false);
        const auto act = activeIndexList(pop);
        const CpuRunResult ref = runCpuReference(pop, p, L.routes(), 150, 1);
        for (int v = 0; v < (int)GpuKernelVariant::Count; ++v) {
            for (int bs : {32, 128, 256, 1024}) {
                GpuRunConfig cfg;
                cfg.variant = (GpuKernelVariant)v;
                cfg.blockSize = bs;
                cfg.steps = 150;
                const GpuRunResult g = runGpuKernelBenchmark(pop, act, p, L.routes(), cfg);
                CHECK(g.ok);
                if (!g.ok) {
                    std::printf("    %s\n", g.error.c_str());
                    continue;
                }
                const StateDiff d = compareStates(ref.finalState, g.finalState);
                const bool ok = d.intMismatches == 0 && d.maxAbsDiff <= kFloatTol && g.eventHash == ref.eventHash &&
                                g.events == ref.events;
                if (!ok)
                    std::printf("    mismatch n=%d variant=%s bs=%d diff=%g int=%lld ev %llu/%llu\n", n,
                                gpuKernelVariantName(cfg.variant), bs, d.maxAbsDiff, d.intMismatches,
                                (unsigned long long)g.events, (unsigned long long)ref.events);
                CHECK(ok);
                CHECK(d.maxAbsDiff == 0.0);  // bit-identical expectation (fmad=false)
            }
        }
    }
}

TEST(gpu_zero_patients_and_bounds) {
    if (!gpuAvailable()) {
        std::printf("    SKIPPED\n");
        return;
    }
    HospitalLayout L;
    L.build(Capacities{});
    UpdateParams p;
    std::vector<PatientAoS> empty;
    GpuRunConfig cfg;
    cfg.steps = 5;
    CHECK(runGpuKernelBenchmark(empty, {}, p, L.routes(), cfg).ok);
    // Out-of-range command ids are rejected by the scatter kernel's bounds check.
    auto be = createCudaBackend(CudaVariant::OptimizedSoA, 128);
    auto pop = makeSyntheticPopulation(100, 5, L, 0.f, false);
    be->init(pop, p, L.routes());
    PatientCommand bad{};
    bad.id = 100;  // == n -> out of range
    bad.mask = CMD_INIT;
    PatientCommand neg = bad;
    neg.id = -7;
    be->applyCommands({bad, neg});
    std::vector<SimEvent> ev;
    be->step(ev);
    std::vector<PatientAoS> out;
    be->readHotState(out);
    CHECK(out.size() == 100);
    auto cpu = createCpuBackend(1);
    cpu->init(pop, p, L.routes());
    cpu->step(ev);
    std::vector<PatientAoS> ref;
    cpu->readHotState(ref);
    CHECK(compareStates(ref, out).intMismatches == 0);
}

TEST(gpu_full_simulation_identical_to_cpu) {
    if (!gpuAvailable()) {
        std::printf("    SKIPPED\n");
        return;
    }
    for (ScenarioType t : {ScenarioType::Normal, ScenarioType::MassCasualty}) {
        ScenarioConfig c = makePreset(t);
        SimulationEngine cpu, naive, opt, opt2, graph;
        cpu.reset(c, createCpuBackend(1));
        naive.reset(c, createCudaBackend(CudaVariant::NaiveAoS, 256));
        opt.reset(c, createCudaBackend(CudaVariant::OptimizedSoA, 256));
        opt2.reset(c, createCudaBackend(CudaVariant::OptimizedSoA, 128));
        graph.reset(c, createCudaBackend(CudaVariant::OptimizedGraph, 256));
        std::string err;
        bool inv = true;
        while (!cpu.finished()) {
            cpu.step();
            naive.step();
            opt.step();
            opt2.step();
            graph.step();
            if (cpu.stepIndex() % 500 == 0) inv &= opt.checkInvariants(&err);
        }
        CHECK(inv);
        CHECK(naive.finished() && opt.finished());
        CHECK(sameRecords(cpu, naive));
        CHECK(sameRecords(cpu, opt));
        CHECK(sameRecords(opt, opt2));
        CHECK(sameRecords(cpu, graph));  // deterministic despite atomics / different block size
        std::vector<PatientAoS> a, b;
        cpu.backend()->readHotState(a);
        opt.backend()->readHotState(b);
        const StateDiff d = compareStates(a, b);
        CHECK(d.intMismatches == 0 && d.maxAbsDiff <= kFloatTol);
        // GPU shared-memory reduction vs CPU reduction
        const HotReduction rc = cpu.backend()->reduce(), rg = opt.backend()->reduce();
        CHECK(rc.activeCount == rg.activeCount);
        CHECK(std::fabs(rc.sumWait - rg.sumWait) <= 1e-4 * (1 + rc.sumWait));
        CHECK(rc.maxWait == rg.maxWait);
    }
}

TEST(gpu_reduction_mid_run) {
    if (!gpuAvailable()) {
        std::printf("    SKIPPED\n");
        return;
    }
    ScenarioConfig c = makePreset(ScenarioType::MassCasualty);
    SimulationEngine cpu, gpu;
    cpu.reset(c, createCpuBackend(1));
    gpu.reset(c, createCudaBackend(CudaVariant::OptimizedSoA, 256));
    for (int i = 0; i < 3000; ++i) {
        cpu.step();
        gpu.step();
    }
    const HotReduction rc = cpu.backend()->reduce(), rg = gpu.backend()->reduce();
    CHECK(rc.activeCount == rg.activeCount && rc.activeCount > 0);
    CHECK(std::fabs(rc.sumHealth - rg.sumHealth) <= 1e-4 * rc.sumHealth);
    std::vector<RenderInstance> r1, r2;
    cpu.backend()->readRenderData(r1);
    gpu.backend()->readRenderData(r2);
    bool same = r1.size() == r2.size();
    for (size_t i = 0; same && i < r1.size(); ++i)
        same &= r1[i].x == r2[i].x && r1[i].z == r2[i].z && r1[i].packed == r2[i].packed;
    CHECK(same);
}

int main(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : "";
    std::printf("%s\n", systemInfoString().c_str());
    int run = 0;
    for (const TestCase& t : registry()) {
        if (!filter.empty() && std::string(t.name).find(filter) == std::string::npos) continue;
        const int before = g_fail;
        std::printf("[ RUN  ] %s\n", t.name);
        try {
            t.fn();
        } catch (const std::exception& ex) {
            std::printf("    EXCEPTION: %s\n", ex.what());
            ++g_fail;
        }
        std::printf("[ %s ] %s\n", g_fail == before ? " OK " : "FAIL", t.name);
        ++run;
    }
    std::printf("\n%d tests, %d checks passed, %d failed\n", run, g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

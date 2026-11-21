#include "Benchmark.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#include "PatientGenerator.h"
#include "PatientUpdate.h"

namespace hs {

using Clock = std::chrono::steady_clock;
static double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

BenchmarkOptions BenchmarkOptions::quick() {
    BenchmarkOptions o;
    o.sizes = {100, 1000, 10000, 100000};
    o.steps = 100;
    o.repeats = 1;
    o.blockSizes = {64, 128, 256, 512, 1024};
    o.scenarioSizes = {1000, 10000};
    o.scenarioHours = 3.0;
    return o;
}

static int hwThreads(int requested) {
    if (requested > 0) return requested;
    unsigned h = std::thread::hardware_concurrency();
    return h ? (int)h : 4;
}

StateDiff compareStates(const std::vector<PatientAoS>& a, const std::vector<PatientAoS>& b) {
    StateDiff d;
    if (a.size() != b.size()) {
        d.intMismatches = (long long)std::max(a.size(), b.size());
        return d;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        const PatientAoS &p = a[i], &q = b[i];
        const float fa[8] = {p.x, p.z, p.tx, p.tz, p.speed, p.waitingTime, p.serviceRemaining, p.health};
        const float fb[8] = {q.x, q.z, q.tx, q.tz, q.speed, q.waitingTime, q.serviceRemaining, q.health};
        for (int k = 0; k < 8; ++k) d.maxAbsDiff = std::max(d.maxAbsDiff, (double)std::fabs(fa[k] - fb[k]));
        d.intMismatches += (p.state != q.state) + (p.priority != q.priority) + (p.route != q.route) +
                           (p.waypoint != q.waypoint) + (p.flags != q.flags);
    }
    return d;
}

// ------------------------------------------------------------ synthetic data ----
std::vector<PatientAoS> makeSyntheticPopulation(int n, uint64_t seed, const HospitalLayout& L,
                                                float inactiveFraction, bool sortByZone) {
    // Relative weights of active states (realistic busy-ED mix).
    static const int activeStates[] = {ST_ARRIVING,     ST_WAIT_TRIAGE, ST_IN_TRIAGE, ST_WAIT_DIAG, ST_IN_DIAG,
                                       ST_WAIT_ER,      ST_IN_TREATMENT, ST_WAIT_ICU, ST_WAIT_WARD, ST_IN_ICU,
                                       ST_IN_WARD,      ST_DISCHARGING};
    static const float weights[] = {14, 16, 5, 6, 5, 18, 12, 3, 3, 3, 8, 7};
    const int nStates = (int)(sizeof(activeStates) / sizeof(int));
    float wsum = 0;
    for (float w : weights) wsum += w;
    static const int realDepts[] = {D_ENTRANCE, D_TRIAGE, D_WAITING, D_ER, D_DIAG, D_ICU, D_WARD, D_DISCHARGE};

    std::vector<PatientAoS> pop(n);
    Rng rng(seed);
    for (int i = 0; i < n; ++i) {
        PatientAoS q{};
        const double u = rng.uniform();
        int st;
        if (u < inactiveFraction) {
            st = rng.uniform() < 0.5 ? ST_NOT_ARRIVED : ST_DISCHARGED;
        } else {
            float pick = (float)rng.uniform() * wsum, acc = 0;
            st = activeStates[nStates - 1];
            for (int k = 0; k < nStates; ++k) {
                acc += weights[k];
                if (pick < acc) {
                    st = activeStates[k];
                    break;
                }
            }
        }
        const Rect b = L.bounds();
        q.x = rng.uniformf(b.x0, b.x1);
        q.z = rng.uniformf(b.z0, b.z1);
        const int from = realDepts[rng.next() % 8], to = realDepts[rng.next() % 8];
        const Rect ta = L.dept(to).area;
        q.tx = rng.uniformf(ta.x0, ta.x1);
        q.tz = rng.uniformf(ta.z0, ta.z1);
        q.speed = rng.uniformf(45.f, 75.f);
        q.waitingTime = inMask(kActiveMask, st) ? rng.uniformf(0.f, 120.f) : 0.f;
        q.serviceRemaining = inMask(kServiceMask, st) ? rng.uniformf(0.5f, 60.f) : 0.f;
        q.health = rng.uniformf(15.f, 100.f);
        q.state = st;
        q.priority = (int)(rng.next() % 4);
        q.route = HospitalLayout::routeId(from, to);
        q.waypoint = (int)(rng.next() % 3);
        bool moving = false;
        if (st == ST_ARRIVING || st == ST_DISCHARGING) moving = true;
        else if (inMask(kWaitMask, st)) moving = rng.uniform() < 0.3;
        else if (inMask(kServiceMask, st)) moving = rng.uniform() < 0.2;
        const bool triaged = !(st == ST_ARRIVING || st == ST_WAIT_TRIAGE || st == ST_IN_TRIAGE) || rng.uniform() < 0.1;
        q.flags = (moving ? FLAG_MOVING : 0) | (triaged ? FLAG_TRIAGED : 0);
        pop[i] = q;
    }
    if (sortByZone) {
        // "Zone-based partitioning": patients in the same state / department become
        // contiguous, so the threads of a warp tend to take the same branches.
        std::stable_sort(pop.begin(), pop.end(), [](const PatientAoS& a, const PatientAoS& b) {
            if (a.state != b.state) return a.state < b.state;
            return (a.flags & FLAG_MOVING) < (b.flags & FLAG_MOVING);
        });
    }
    return pop;
}

std::vector<int32_t> activeIndexList(const std::vector<PatientAoS>& pop) {
    std::vector<int32_t> idx;
    for (int i = 0; i < (int)pop.size(); ++i)
        if (inMask(kActiveMask, pop[i].state)) idx.push_back(i);
    return idx;
}

CpuRunResult runCpuReference(const std::vector<PatientAoS>& init, const UpdateParams& p, const RouteTable& rt,
                             int steps, int threads) {
    CpuRunResult r;
    auto be = createCpuBackend(threads);
    be->init(init, p, rt);
    std::vector<SimEvent> ev;
    uint64_t hash = 1469598103934665603ull;
    auto t0 = Clock::now();
    for (int s = 0; s < steps; ++s) {
        be->step(ev);
        r.events += ev.size();
        hash = hashEvents(hash, (uint64_t)s, ev.data(), ev.size());
    }
    r.ms = msSince(t0);
    r.eventHash = hash;
    be->readHotState(r.finalState);
    return r;
}

// --------------------------------------------------------------- kernel suite ----
namespace {

template <typename T>
T median(std::vector<T> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? T() : v[v.size() / 2];
}

struct SuiteCtx {
    const BenchmarkOptions& o;
    UpdateParams params;
    HospitalLayout layout;
    int threads;
    int total = 1, done = 0;
    void tick(const std::string& msg) {
        ++done;
        if (o.progress) o.progress(msg, std::min(1.f, (float)done / std::max(1, total)));
    }
    bool cancelled() const { return o.cancel && o.cancel->load(); }
};

KernelBenchRow cpuRow(SuiteCtx& c, const std::string& exp, const std::vector<PatientAoS>& pop, int threads,
                      CpuRunResult* keep) {
    std::vector<double> ms;
    CpuRunResult last;
    for (int r = 0; r < c.o.repeats; ++r) {
        last = runCpuReference(pop, c.params, c.layout.routes(), c.o.steps, threads);
        ms.push_back(last.ms);
    }
    KernelBenchRow row;
    row.experiment = exp;
    row.variant = threads > 1 ? "CPU multi-thread (OpenMP)" : "CPU single-thread";
    row.layout = "SoA";
    row.threads = threads;
    row.patients = (int)pop.size();
    row.steps = c.o.steps;
    row.kernelMs = row.e2eMs = median(ms);
    if (keep) *keep = std::move(last);
    return row;
}

KernelBenchRow gpuRow(SuiteCtx& c, const std::string& exp, const std::vector<PatientAoS>& pop,
                      const std::vector<int32_t>& active, GpuKernelVariant v, int bs, const CpuRunResult& ref) {
    std::vector<double> k, e, h, d;
    GpuRunResult last;
    GpuRunConfig cfg;
    cfg.variant = v;
    cfg.blockSize = bs;
    cfg.steps = c.o.steps;
    for (int r = 0; r < c.o.repeats; ++r) {
        last = runGpuKernelBenchmark(pop, active, c.params, c.layout.routes(), cfg);
        if (!last.ok) break;
        k.push_back(last.kernelMs);
        e.push_back(last.endToEndMs);
        h.push_back(last.h2dMs);
        d.push_back(last.d2hMs);
    }
    KernelBenchRow row;
    row.experiment = exp;
    row.variant = gpuKernelVariantName(v);
    row.layout = (v == GpuKernelVariant::NaiveAoS || v == GpuKernelVariant::AoSFused) ? "AoS" : "SoA";
    row.blockSize = bs;
    row.patients = (int)pop.size();
    row.steps = c.o.steps;
    row.isGpu = true;
    if (!last.ok) {
        row.variant += " [FAILED: " + last.error + "]";
        row.eventsMatch = false;
        return row;
    }
    row.kernelMs = median(k);
    row.e2eMs = median(e);
    row.h2dMs = median(h);
    row.d2hMs = median(d);
    const StateDiff diff = compareStates(ref.finalState, last.finalState);
    row.maxAbsDiff = diff.maxAbsDiff;
    row.intMismatches = diff.intMismatches;
    row.eventsMatch = ref.eventHash == last.eventHash && ref.events == last.events;
    return row;
}

void finishRows(std::vector<KernelBenchRow>& rows, size_t from, double cpu1Ms) {
    for (size_t i = from; i < rows.size(); ++i) {
        KernelBenchRow& r = rows[i];
        r.perStepUs = r.steps > 0 ? r.e2eMs * 1000.0 / r.steps : 0;
        r.updatesPerSec = r.e2eMs > 0 ? (double)r.patients * r.steps / (r.e2eMs / 1000.0) : 0;
        r.speedupKernel = r.kernelMs > 0 ? cpu1Ms / r.kernelMs : 0;
        r.speedupE2E = r.e2eMs > 0 ? cpu1Ms / r.e2eMs : 0;
    }
}

}  // namespace

std::vector<KernelBenchRow> Benchmark::runKernelSuite(const BenchmarkOptions& o) {
    SuiteCtx c{o};
    c.params.dt = o.benchDt;
    c.layout.build(Capacities{});
    c.threads = hwThreads(o.cpuThreads);
    const bool gpu = queryGpu().available;
    const int maxN = o.sizes.empty() ? 0 : *std::max_element(o.sizes.begin(), o.sizes.end());
    const std::vector<float> inactive = {0.2f, 0.5f, 0.8f, 0.95f};
    c.total = (int)o.sizes.size() + 1 + 1 + (int)inactive.size();
    std::vector<KernelBenchRow> rows;
    const std::vector<int32_t> none;

    // 1) Size sweep: CPU 1T, CPU MT, and all GPU variants.
    for (int n : o.sizes) {
        if (c.cancelled()) return rows;
        const auto pop = makeSyntheticPopulation(n, o.seed, c.layout, 0.2f, false);
        const auto act = activeIndexList(pop);
        CpuRunResult ref;
        const size_t from = rows.size();
        rows.push_back(cpuRow(c, "size_sweep", pop, 1, &ref));
        const double cpu1 = rows.back().e2eMs;
        rows.push_back(cpuRow(c, "size_sweep", pop, c.threads, nullptr));
        {
            const StateDiff d = compareStates(ref.finalState, runCpuReference(pop, c.params, c.layout.routes(),
                                                                              o.steps, c.threads).finalState);
            rows.back().maxAbsDiff = d.maxAbsDiff;
            rows.back().intMismatches = d.intMismatches;
        }
        if (gpu) {
            for (GpuKernelVariant v : {GpuKernelVariant::NaiveAoS, GpuKernelVariant::AoSFused, GpuKernelVariant::SoASplit,
                                       GpuKernelVariant::SoAFused})
                rows.push_back(gpuRow(c, "size_sweep", pop, act, v, o.blockSize, ref));
        }
        finishRows(rows, from, cpu1);
        c.tick("size sweep N=" + std::to_string(n));
    }
    if (!gpu || maxN == 0) return rows;

    // 2) Block-size sweep at the largest N.
    {
        const auto pop = makeSyntheticPopulation(maxN, o.seed, c.layout, 0.2f, false);
        CpuRunResult ref;
        const size_t from = rows.size();
        rows.push_back(cpuRow(c, "block_size", pop, 1, &ref));
        const double cpu1 = rows.back().e2eMs;
        for (int bs : o.blockSizes) {
            if (c.cancelled()) return rows;
            for (GpuKernelVariant v : {GpuKernelVariant::NaiveAoS, GpuKernelVariant::AoSFused, GpuKernelVariant::SoAFused})
                rows.push_back(gpuRow(c, "block_size", pop, none, v, bs, ref));
        }
        finishRows(rows, from, cpu1);
        c.tick("block-size sweep");
    }
    // 3) Ordering: random order vs zone/state-sorted (warp divergence + locality).
    {
        for (bool sorted : {false, true}) {
            if (c.cancelled()) return rows;
            const auto pop = makeSyntheticPopulation(maxN, o.seed, c.layout, 0.2f, sorted);
            CpuRunResult ref;
            const size_t from = rows.size();
            rows.push_back(cpuRow(c, "ordering", pop, 1, &ref));
            const double cpu1 = rows.back().e2eMs;
            rows.push_back(gpuRow(c, "ordering", pop, none, GpuKernelVariant::SoAFused, o.blockSize, ref));
            rows.push_back(gpuRow(c, "ordering", pop, none, GpuKernelVariant::AoSFused, o.blockSize, ref));
            for (size_t i = from; i < rows.size(); ++i) rows[i].ordering = sorted ? "zone-sorted" : "random";
            finishRows(rows, from, cpu1);
        }
        c.tick("ordering experiment");
    }
    // 4) Active-list processing vs full range, varying the inactive fraction.
    for (float f : inactive) {
        if (c.cancelled()) return rows;
        const auto pop = makeSyntheticPopulation(maxN, o.seed, c.layout, f, false);
        const auto act = activeIndexList(pop);
        CpuRunResult ref;
        const size_t from = rows.size();
        rows.push_back(cpuRow(c, "active_list", pop, 1, &ref));
        const double cpu1 = rows.back().e2eMs;
        rows.push_back(gpuRow(c, "active_list", pop, none, GpuKernelVariant::SoAFused, o.blockSize, ref));
        rows.push_back(gpuRow(c, "active_list", pop, act, GpuKernelVariant::SoAFusedActive, o.blockSize, ref));
        for (size_t i = from; i < rows.size(); ++i) rows[i].inactiveFraction = f;
        finishRows(rows, from, cpu1);
        c.tick("active-list f=" + std::to_string(f));
    }
    return rows;
}

// ------------------------------------------------------------- scenario suite ----
const char* backendKindName(BackendKind k) {
    switch (k) {
        case BackendKind::Cpu1: return "CPU single-thread";
        case BackendKind::CpuMT: return "CPU multi-thread (OpenMP)";
        case BackendKind::CudaNaive: return "CUDA naive";
        case BackendKind::CudaOptimized: return "CUDA optimized";
        case BackendKind::CudaGraph: return "CUDA optimized + Graph";
        default: break;
    }
    return "?";
}

std::unique_ptr<PatientBackend> makeBackend(BackendKind k, int blockSize, std::string* error) {
    switch (k) {
        case BackendKind::Cpu1: return createCpuBackend(1);
        case BackendKind::CpuMT: return createCpuBackend(hwThreads(0));
        case BackendKind::CudaNaive: return createCudaBackend(CudaVariant::NaiveAoS, blockSize, error);
        case BackendKind::CudaOptimized: return createCudaBackend(CudaVariant::OptimizedSoA, blockSize, error);
        case BackendKind::CudaGraph: return createCudaBackend(CudaVariant::OptimizedGraph, blockSize, error);
        default: break;
    }
    return nullptr;
}

static bool sameOutcome(const SimulationEngine& a, const SimulationEngine& b) {
    const auto &ra = a.records(), &rb = b.records();
    if (ra.size() != rb.size()) return false;
    for (size_t i = 0; i < ra.size(); ++i) {
        if (ra[i].state != rb[i].state || ra[i].priority != rb[i].priority ||
            ra[i].treatmentStart != rb[i].treatmentStart || ra[i].dischargeTime != rb[i].dischargeTime ||
            ra[i].erBed != rb[i].erBed || ra[i].escalations != rb[i].escalations)
            return false;
    }
    return true;
}

std::vector<ScenarioBenchRow> Benchmark::runScenarioSuite(const BenchmarkOptions& o) {
    std::vector<ScenarioBenchRow> rows;
    const bool gpu = queryGpu().available;
    struct RunCfg {
        BackendKind kind;
        bool batching;
    };
    // First entry is the reference: CPU single-thread, one controller cycle per step.
    std::vector<RunCfg> kinds = {{BackendKind::Cpu1, false}, {BackendKind::CpuMT, false}};
    if (gpu) {
        kinds.push_back({BackendKind::CudaNaive, false});
        kinds.push_back({BackendKind::CudaOptimized, false});
        kinds.push_back({BackendKind::CudaGraph, false});
    }
    kinds.push_back({BackendKind::Cpu1, true});
    kinds.push_back({BackendKind::CpuMT, true});
    if (gpu) kinds.push_back({BackendKind::CudaOptimized, true});
    const ScenarioConfig base = makePreset(ScenarioType::MassCasualty);
    const int total = (int)(o.scenarioSizes.size() * kinds.size());
    int done = 0;
    for (int n : o.scenarioSizes) {
        ScenarioConfig cfg = scaleScenario(base, n);
        cfg.seed = o.seed;
        std::unique_ptr<SimulationEngine> reference;
        double cpuWall = 0, cpuBackend = 0;
        for (const RunCfg& rc : kinds) {
            const BackendKind k = rc.kind;
            if (o.cancel && o.cancel->load()) return rows;
            std::string err;
            auto be = makeBackend(k, o.blockSize, &err);
            ScenarioBenchRow row;
            row.patients = n;
            row.backend = std::string(backendKindName(k)) + (rc.batching ? " [event-batched]" : " [per-step]");
            row.batching = rc.batching;
            if (!be) {
                row.backend += " [unavailable: " + err + "]";
                rows.push_back(row);
                continue;
            }
            row.isGpu = be->isGpu();
            auto engPtr = std::make_unique<SimulationEngine>();
            SimulationEngine& eng = *engPtr;
            eng.batching = rc.batching;
            // Wall-clock comparison without per-step CUDA-event instrumentation (it costs ~30 us/step on
            // WDDM, measured). Kernel/transfer breakdowns come from the kernel suite instead.
            setCudaEventTiming(false);
            eng.reset(cfg, std::move(be));
            auto t0 = Clock::now();
            eng.runToCompletion(o.scenarioHours * 60.0);
            row.wallMs = msSince(t0);
            setCudaEventTiming(true);
            const EngineProfile& p = eng.profile();
            const BackendTimings& t = eng.backend()->timings();
            row.backendMs = p.backendMs;
            row.controllerMs = p.controllerMs;
            row.commandMs = p.commandMs;
            row.computeMs = t.totalComputeMs;
            row.h2dMs = t.totalH2DMs;
            row.d2hMs = t.totalD2HMs;
            row.bytesH2D = t.bytesH2D;
            row.bytesD2H = t.bytesD2H;
            row.steps = p.steps;
            row.syncs = t.batches;
            row.events = p.events;
            row.commands = p.commands;
            const LiveMetrics m = eng.computeMetrics();
            row.discharged = m.discharged;
            row.avgWait = m.avgWait;
            row.avgTreatmentDelay = m.avgTreatmentDelay;
            if (!reference) {
                cpuWall = row.wallMs;
                cpuBackend = row.backendMs;
            }
            row.identicalToCpu = reference ? sameOutcome(*reference, eng) : true;
            if (!reference) reference = std::move(engPtr);
            row.speedupWall = row.wallMs > 0 ? cpuWall / row.wallMs : 0;
            row.speedupBackend = row.backendMs > 0 ? cpuBackend / row.backendMs : 0;
            rows.push_back(row);
            ++done;
            if (o.progress) o.progress("scenario N=" + std::to_string(n) + " " + row.backend,
                                       (float)done / std::max(1, total));
        }
    }
    return rows;
}

// ------------------------------------------------------------------------ CSV ----
static void ensureDir(const std::string& path) {
    std::error_code ec;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent, ec);
}

bool Benchmark::writeKernelCsv(const std::string& path, const std::vector<KernelBenchRow>& rows) {
    ensureDir(path);
    std::ofstream f(path);
    if (!f) return false;
    f << "experiment,variant,layout,ordering,inactive_fraction,block_size,patients,steps,threads,is_gpu,"
         "kernel_ms,h2d_ms,d2h_ms,end_to_end_ms,per_step_us,patient_updates_per_s,speedup_kernel_vs_cpu1,"
         "speedup_e2e_vs_cpu1,max_abs_diff_vs_cpu,int_mismatches_vs_cpu,events_match_cpu\n";
    for (const auto& r : rows) {
        f << r.experiment << ",\"" << r.variant << "\"," << r.layout << "," << r.ordering << "," << r.inactiveFraction
          << "," << r.blockSize << "," << r.patients << "," << r.steps << "," << r.threads << "," << (r.isGpu ? 1 : 0)
          << "," << r.kernelMs << "," << r.h2dMs << "," << r.d2hMs << "," << r.e2eMs << "," << r.perStepUs << ","
          << r.updatesPerSec << "," << r.speedupKernel << "," << r.speedupE2E << "," << r.maxAbsDiff << ","
          << r.intMismatches << "," << (r.eventsMatch ? 1 : 0) << "\n";
    }
    return true;
}

bool Benchmark::writeScenarioCsv(const std::string& path, const std::vector<ScenarioBenchRow>& rows) {
    ensureDir(path);
    std::ofstream f(path);
    if (!f) return false;
    f << "patients,backend,batching,is_gpu,wall_ms,backend_ms,controller_ms,command_ms,compute_ms,h2d_ms,d2h_ms,steps,events,"
         "commands,host_syncs,bytes_h2d,bytes_d2h,discharged,avg_wait_min,avg_treatment_delay_min,identical_to_cpu,"
         "speedup_wall_vs_cpu1,speedup_backend_vs_cpu1\n";
    for (const auto& r : rows) {
        f << r.patients << ",\"" << r.backend << "\"," << (r.batching ? 1 : 0) << "," << (r.isGpu ? 1 : 0) << "," << r.wallMs << "," << r.backendMs
          << "," << r.controllerMs << "," << r.commandMs << "," << r.computeMs << "," << r.h2dMs << "," << r.d2hMs
          << "," << r.steps << "," << r.events << "," << r.commands << "," << r.syncs << "," << r.bytesH2D << "," << r.bytesD2H << ","
          << r.discharged << "," << r.avgWait << "," << r.avgTreatmentDelay << "," << (r.identicalToCpu ? 1 : 0)
          << "," << r.speedupWall << "," << r.speedupBackend << "\n";
    }
    return true;
}

// ------------------------------------------------------------- headless runs ----
ConfigRunResult runScenarioHeadless(const ScenarioConfig& cfg, BackendKind k, double maxHours) {
    ConfigRunResult r;
    r.cfg = cfg;
    try {
        std::string err;
        auto be = makeBackend(k, 256, &err);
        if (!be) {
            r.error = err;
            return r;
        }
        SimulationEngine eng;
        eng.reset(cfg, std::move(be));
        auto t0 = Clock::now();
        eng.runToCompletion(maxHours > 0 ? maxHours * 60.0 : -1);
        r.wallMs = msSince(t0);
        r.metrics = eng.computeMetrics();
        r.ok = true;
    } catch (const std::exception& ex) {
        r.error = ex.what();
    }
    return r;
}

std::string systemInfoString() {
    std::ostringstream s;
    s << "CPU hardware threads: " << std::thread::hardware_concurrency() << "\n";
#ifdef _OPENMP
    s << "OpenMP: enabled (" << _OPENMP << ")\n";
#else
    s << "OpenMP: disabled\n";
#endif
    const GpuDeviceInfo g = queryGpu();
    s << "CUDA build: " << (cudaBuildEnabled() ? "yes" : "no") << "\n";
    if (g.available) {
        s << "GPU: " << g.name << " (sm_" << g.ccMajor << g.ccMinor << ", " << g.smCount << " SMs, "
          << (g.totalMemBytes >> 20) << " MiB, bus " << g.memBusWidth << "-bit, L2 " << (g.l2CacheBytes >> 10)
          << " KiB)\n";
        s << "CUDA driver " << g.driverVersion / 1000 << "." << (g.driverVersion % 1000) / 10 << ", runtime "
          << g.runtimeVersion / 1000 << "." << (g.runtimeVersion % 1000) / 10 << "\n";
    } else {
        s << "GPU: unavailable (" << g.error << ")\n";
    }
    return s.str();
}

}  // namespace hs

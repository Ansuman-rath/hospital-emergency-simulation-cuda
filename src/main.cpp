// main.cpp - entry point: interactive 3D application or headless experiment modes.
//
//   hospital_sim                         interactive OpenGL + ImGui application
//   hospital_sim --info                  hardware / CUDA information
//   hospital_sim --benchmark [--quick]   kernel suite + scenario suite -> CSV
//   hospital_sim --kernel-bench [--quick]
//   hospital_sim --scenario-bench [--quick]
//   hospital_sim --run <0-3> [--backend cpu|cpumt|naive|gpu] [--patients N]
//   hospital_sim --compare <0-3> [--backend ...]   configuration A vs B
//   common: --out <dir> --steps N --repeats R --seed S --block B
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "App.h"
#include "Benchmark.h"

#ifdef _WIN32
// Ask hybrid-graphics laptops (NVIDIA Optimus / AMD) to create the OpenGL context on
// the discrete GPU, which also makes CUDA-OpenGL interop possible.
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

using namespace hs;

namespace {

struct Args {
    int argc;
    char** argv;
    bool has(const char* f) const {
        for (int i = 1; i < argc; ++i)
            if (std::strcmp(argv[i], f) == 0) return true;
        return false;
    }
    const char* value(const char* f) const {
        for (int i = 1; i + 1 < argc; ++i)
            if (std::strcmp(argv[i], f) == 0) return argv[i + 1];
        return nullptr;
    }
    int intValue(const char* f, int def) const {
        const char* v = value(f);
        return v ? std::atoi(v) : def;
    }
};

BackendKind parseBackend(const char* s) {
    if (!s) return queryGpu().available ? BackendKind::CudaOptimized : BackendKind::Cpu1;
    std::string v = s;
    if (v == "cpu") return BackendKind::Cpu1;
    if (v == "cpumt") return BackendKind::CpuMT;
    if (v == "naive") return BackendKind::CudaNaive;
    if (v == "graph") return BackendKind::CudaGraph;
    return BackendKind::CudaOptimized;
}

void progress(const std::string& m, float f) { std::printf("  [%3.0f%%] %s\n", f * 100.f, m.c_str()); }

void printMetrics(const char* title, const ConfigRunResult& r) {
    const LiveMetrics& m = r.metrics;
    std::printf("\n== %s ==\n", title);
    if (!r.ok) {
        std::printf("  failed: %s\n", r.error.c_str());
        return;
    }
    std::printf("  scenario: %s, %d patients, seed %llu\n", scenarioName(r.cfg.type), m.total,
                (unsigned long long)r.cfg.seed);
    std::printf("  resources: %d doctors, %d nurses, %d ER, %d ICU, %d ward beds, %d lab slots\n", r.cfg.cap.doctors,
                r.cfg.cap.nurses, r.cfg.cap.erBeds, r.cfg.cap.icuBeds, r.cfg.cap.wardBeds, r.cfg.cap.labSlots);
    std::printf("  simulated %.1f h in %.0f ms wall; discharged %d/%d\n", m.simTime / 60, r.wallMs, m.discharged, m.total);
    std::printf("  avg wait %.1f min (cpu log %.1f), max wait %.1f min, avg treatment delay %.1f min\n", m.avgWait,
                m.avgWaitCpu, m.maxWait, m.avgTreatmentDelay);
    std::printf("  avg length of stay %.1f min, throughput %.1f/h, escalations %d\n", m.avgLengthOfStay,
                m.throughputPerHour, m.escalations);
    for (int q = 0; q < Q_COUNT; ++q)
        std::printf("  queue %-18s served %6d  avg wait %7.1f  max %7.1f min\n", queueName(q), m.queues[q].served,
                    m.queues[q].avgWait, m.queues[q].maxWait);
    for (int p = 0; p < R_COUNT; ++p)
        std::printf("  utilisation %-14s %5.1f%%\n", resourceName(p), 100 * m.utilization[p]);
    if (m.slowestQueue >= 0) std::printf("  bottleneck (longest avg wait): %s\n", queueName(m.slowestQueue));
}

BenchmarkOptions benchOptions(const Args& a) {
    BenchmarkOptions o = a.has("--quick") ? BenchmarkOptions::quick() : BenchmarkOptions();
    if (const char* d = a.value("--out")) o.outDir = d;
    o.steps = a.intValue("--steps", o.steps);
    o.repeats = a.intValue("--repeats", o.repeats);
    o.blockSize = a.intValue("--block", o.blockSize);
    if (const char* s = a.value("--seed")) o.seed = std::strtoull(s, nullptr, 10);
    o.progress = progress;
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    Args a{argc, argv};
    if (a.has("--no-event-timing")) setCudaEventTiming(false);
    try {
        if (a.has("--help") || a.has("-h")) {
            std::printf("see header of src/main.cpp or docs/USER_GUIDE.md for options\n");
            return 0;
        }
        if (a.has("--info")) {
            std::printf("%s", systemInfoString().c_str());
            return 0;
        }
        const bool kb = a.has("--kernel-bench") || a.has("--benchmark");
        const bool sb = a.has("--scenario-bench") || a.has("--benchmark");
        if (kb || sb) {
            BenchmarkOptions o = benchOptions(a);
            std::printf("%s\n", systemInfoString().c_str());
            {
                std::ofstream f(o.outDir + "/system_info.txt");
                if (!f) {
                    std::filesystem::create_directories(o.outDir);
                    f.open(o.outDir + "/system_info.txt");
                }
                f << systemInfoString() << "steps per run: " << o.steps << "\nrepeats (median): " << o.repeats
                  << "\nbench dt: " << o.benchDt << "\nseed: " << o.seed << "\n";
            }
            if (kb) {
                std::printf("Kernel suite (%d steps, median of %d)\n", o.steps, o.repeats);
                auto rows = Benchmark::runKernelSuite(o);
                Benchmark::writeKernelCsv(o.outDir + "/kernel_benchmark.csv", rows);
                std::printf("%-30s %9s %6s %11s %11s %11s %8s %8s %5s\n", "variant", "N", "block", "kernel ms",
                            "e2e ms", "h2d+d2h ms", "xKernel", "xE2E", "ok");
                for (const auto& r : rows) {
                    if (r.experiment != "size_sweep") continue;
                    const bool ok = r.eventsMatch && r.intMismatches == 0 && r.maxAbsDiff <= 1e-4;
                    std::printf("%-30s %9d %6d %11.3f %11.3f %11.3f %8.2f %8.2f %5s\n", r.variant.c_str(), r.patients,
                                r.blockSize, r.kernelMs, r.e2eMs, r.h2dMs + r.d2hMs, r.speedupKernel, r.speedupE2E,
                                ok ? "yes" : "NO");
                }
            }
            if (sb) {
                std::printf("Scenario suite (mass-casualty scaled, %.1f simulated hours)\n", o.scenarioHours);
                auto rows = Benchmark::runScenarioSuite(o);
                Benchmark::writeScenarioCsv(o.outDir + "/scenario_benchmark.csv", rows);
                std::printf("%-28s %8s %10s %10s %10s %8s %6s\n", "backend", "N", "wall ms", "backend", "controller",
                            "xWall", "same");
                for (const auto& r : rows)
                    std::printf("%-28s %8d %10.0f %10.0f %10.0f %8.2f %6s\n", r.backend.c_str(), r.patients, r.wallMs,
                                r.backendMs, r.controllerMs, r.speedupWall, r.identicalToCpu ? "yes" : "NO");
            }
            std::printf("CSV written to %s\n", o.outDir.c_str());
            return 0;
        }
        if (a.value("--profile")) {
            // Small, targeted workload for Nsight Compute / Nsight Systems / compute-sanitizer.
            const int n = a.intValue("--profile", 100000);
            HospitalLayout L;
            L.build(Capacities{});
            UpdateParams p;
            p.dt = 0.02f;
            const auto pop = makeSyntheticPopulation(n, 42, L, 0.2f, a.has("--sorted"));
            const auto act = activeIndexList(pop);
            for (int v = 0; v < (int)GpuKernelVariant::Count; ++v) {
                GpuRunConfig g;
                g.variant = (GpuKernelVariant)v;
                g.steps = a.intValue("--steps", 5);
                const GpuRunResult r = runGpuKernelBenchmark(pop, act, p, L.routes(), g);
                std::printf("%-30s N=%d kernel %.3f ms  e2e %.3f ms  %s\n", gpuKernelVariantName(g.variant), n,
                            r.kernelMs, r.endToEndMs, r.ok ? "ok" : r.error.c_str());
            }
            return 0;
        }
        if (const char* s = a.value("--run")) {
            ScenarioConfig c = makePreset((ScenarioType)std::atoi(s));
            if (a.value("--patients")) c = scaleScenario(c, a.intValue("--patients", c.totalPatients));
            if (const char* sd = a.value("--seed")) c.seed = std::strtoull(sd, nullptr, 10);
            const BackendKind k = parseBackend(a.value("--backend"));
            printMetrics(backendKindName(k), runScenarioHeadless(c, k));
            return 0;
        }
        if (const char* s = a.value("--compare")) {
            ScenarioConfig A = makePreset((ScenarioType)std::atoi(s));
            ScenarioConfig B = A;
            B.cap.doctors += 5;
            B.cap.icuBeds += 10;
            const BackendKind k = parseBackend(a.value("--backend"));
            printMetrics("Configuration A", runScenarioHeadless(A, k));
            printMetrics("Configuration B (+5 doctors, +10 ICU beds)", runScenarioHeadless(B, k));
            return 0;
        }
        App app;
        if (const char* shot = a.value("--screenshot")) {
            app.screenshotPath = shot;
            app.screenshotScenario = a.intValue("--scenario", 3);
            app.screenshotSimMinutes = a.intValue("--sim-minutes", 120);
            app.screenshotCamera = a.intValue("--camera", 1);
            app.screenshotTab = a.intValue("--tab", 0);
            app.screenshotPatients = a.intValue("--patients", 0);
        }
        return app.run();
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}

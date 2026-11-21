#include "Dashboard.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <string>

#include "App.h"
#include "PatientUpdate.h"
#include "imgui.h"

namespace hs {
namespace {

const ImVec4 kPrioColor[4] = {{0.95f, 0.25f, 0.25f, 1}, {1.0f, 0.60f, 0.15f, 1}, {0.98f, 0.88f, 0.20f, 1},
                              {0.30f, 0.80f, 0.35f, 1}};
const ImVec4 kDim(0.65f, 0.67f, 0.72f, 1.f);
const ImVec4 kGood(0.4f, 0.9f, 0.5f, 1.f);
const ImVec4 kBad(1.f, 0.4f, 0.35f, 1.f);

std::string fmtTime(double minutes) {
    const int m = (int)std::floor(minutes);
    char b[32];
    std::snprintf(b, sizeof(b), "%dd %02d:%02d", m / 1440, (m / 60) % 24, m % 60);
    return b;
}

void help(const char* text) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::BeginItemTooltip()) {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

void kv(const char* k, const char* fmt, ...) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextColored(kDim, "%s", k);
    ImGui::TableNextColumn();
    va_list a;
    va_start(a, fmt);
    ImGui::TextV(fmt, a);
    va_end(a);
}

bool backendCombo(const char* label, BackendKind& k, bool gpu) {
    bool changed = false;
    if (ImGui::BeginCombo(label, backendKindName(k))) {
        for (int i = 0; i < (int)BackendKind::Count; ++i) {
            const BackendKind c = (BackendKind)i;
            const bool needsGpu = c != BackendKind::Cpu1 && c != BackendKind::CpuMT;
            ImGui::BeginDisabled(needsGpu && !gpu);
            if (ImGui::Selectable(backendKindName(c), k == c)) {
                k = c;
                changed = true;
            }
            ImGui::EndDisabled();
        }
        ImGui::EndCombo();
    }
    return changed;
}

// --------------------------------------------------------------- left panel ----
void controlPanel(App& app) {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float pad = 8.f;
    ImGui::SetNextWindowPos({vp->WorkPos.x + pad, vp->WorkPos.y + pad}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({390.f * ImGui::GetFontSize() / 14.f, vp->WorkSize.y - 2 * pad}, ImGuiCond_Once);
    ImGui::Begin("Control Panel");
    ImGui::TextColored({0.55f, 0.8f, 1.f, 1.f}, "GPU-Accelerated Hospital Emergency Simulation");
    ImGui::TextColored(kDim, "Simulation / analysis tool - not a clinical system.");
    ImGui::Separator();

    ScenarioConfig& c = app.cfg;
    const bool busy = app.job.busy();
    ImGui::BeginDisabled(busy);

    // ---- simulation controls
    const bool started = app.engine.stepIndex() > 0;
    if (!app.running) {
        if (ImGui::Button(started ? "Resume" : "Start", {90, 0})) app.running = true;
    } else if (ImGui::Button("Pause", {90, 0})) {
        app.running = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset", {90, 0})) app.resetSimulation();
    help("Rebuilds the hospital with the configuration below and restarts the deterministic scenario.");
    ImGui::SameLine();
    if (ImGui::Button("Step")) app.stepOnce();
    ImGui::SetNextItemWidth(-90);
    ImGui::SliderFloat("Speed", &app.simSpeed, 0.5f, 600.f, "%.1f sim-min/s", ImGuiSliderFlags_Logarithmic);
    if (app.lagging) ImGui::TextColored(kBad, "Backend cannot keep up: running at %.0f steps/s", app.stepsPerSecond);
    if (app.engine.finished()) ImGui::TextColored(kGood, "Scenario complete.");

    // ---- scenario
    if (ImGui::CollapsingHeader("Scenario", ImGuiTreeNodeFlags_DefaultOpen)) {
        int t = (int)c.type;
        const char* names[] = {scenarioName(ScenarioType::Normal), scenarioName(ScenarioType::PatientSurge),
                               scenarioName(ScenarioType::AmbulanceSurge), scenarioName(ScenarioType::MassCasualty)};
        if (ImGui::Combo("Preset", &t, names, 4)) {
            app.applyPreset((ScenarioType)t);
            app.resetSimulation();
        }
        ImGui::InputInt("Patients", &c.totalPatients, 50, 1000);
        c.totalPatients = std::clamp(c.totalPatients, 1, 2000000);
        ImGui::SliderFloat("Arrivals / hour", &c.arrivalRatePerHour, 1.f, 5000.f, "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::InputScalar("Seed", ImGuiDataType_U64, &c.seed);
        if (ImGui::TreeNode("Priority mix (P1..P4)")) {
            for (int p = 0; p < 4; ++p) {
                ImGui::PushStyleColor(ImGuiCol_SliderGrab, kPrioColor[p]);
                ImGui::SliderFloat(priorityName(p), &c.priorityMix[p], 0.f, 1.f, "%.2f");
                ImGui::PopStyleColor();
            }
            float s = c.priorityMix[0] + c.priorityMix[1] + c.priorityMix[2] + c.priorityMix[3];
            ImGui::TextColored(kDim, "weights are normalised (sum %.2f)", s);
            ImGui::SliderFloat("Ambulance share", &c.ambulanceFraction, 0.f, 1.f);
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Surge / ambulance / mass-casualty")) {
            ImGui::SliderFloat("Surge multiplier", &c.surgeMultiplier, 1.f, 10.f);
            ImGui::SliderFloat("Surge start (min)", &c.surgeStartMin, 0.f, 1440.f);
            ImGui::SliderFloat("Surge duration (min)", &c.surgeDurationMin, 0.f, 1440.f);
            ImGui::SliderInt("Ambulance bursts", &c.burstCount, 0, 50);
            ImGui::SliderInt("Patients / burst", &c.burstSize, 0, 200);
            ImGui::SliderFloat("Burst interval (min)", &c.burstIntervalMin, 5.f, 240.f);
            ImGui::SliderInt("Mass-casualty patients", &c.mciPatients, 0, 5000);
            ImGui::SliderFloat("Incident time (min)", &c.mciTimeMin, 0.f, 1440.f);
            ImGui::SliderFloat("Incident window (min)", &c.mciWindowMin, 1.f, 240.f);
            ImGui::TreePop();
        }
    }
    // ---- resources
    if (ImGui::CollapsingHeader("Hospital resources", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderInt("Doctors", &c.cap.doctors, 1, 200);
        ImGui::SliderInt("Triage nurses", &c.cap.nurses, 1, 100);
        ImGui::SliderInt("ER beds", &c.cap.erBeds, 1, 300);
        ImGui::SliderInt("ICU beds", &c.cap.icuBeds, 1, 200);
        ImGui::SliderInt("Ward beds", &c.cap.wardBeds, 1, 1000);
        ImGui::SliderInt("Lab slots", &c.cap.labSlots, 1, 60);
        ImGui::SliderFloat("Avg treatment (min)", &c.avgTreatmentMin, 5.f, 240.f);
        if (ImGui::TreeNode("Other service times")) {
            ImGui::SliderFloat("Triage (min)", &c.triageMin, 1.f, 30.f);
            ImGui::SliderFloat("Diagnostics (min)", &c.diagMin, 5.f, 120.f);
            ImGui::SliderFloat("ICU stay (min)", &c.icuStayMin, 30.f, 2880.f);
            ImGui::SliderFloat("Ward stay (min)", &c.wardStayMin, 30.f, 4320.f);
            ImGui::TreePop();
        }
        ImGui::TextColored(kDim, "Changes apply on Reset.");
    }
    // ---- engine
    if (ImGui::CollapsingHeader("Simulation engine", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (backendCombo("Backend", app.backendKind, app.gpuAvailable())) app.resetSimulation();
        if (!app.gpuAvailable()) ImGui::TextColored(kBad, "CUDA GPU unavailable: %s", app.gpu.error.c_str());
        const int sizes[] = {32, 64, 128, 256, 512, 1024};
        int bi = 3;
        for (int i = 0; i < 6; ++i)
            if (sizes[i] == app.blockSize) bi = i;
        const char* bs[] = {"32", "64", "128", "256", "512", "1024"};
        if (ImGui::Combo("Threads/block", &bi, bs, 6)) {
            app.blockSize = sizes[bi];
            app.resetSimulation();
        }
        if (ImGui::Checkbox("CUDA-OpenGL interop", &app.useInterop)) app.resetSimulation();
        help("Optimized CUDA backend only: the render-pack kernel writes patient instances straight into the "
             "OpenGL vertex buffer (no device->host->device round trip).");
        ImGui::SliderFloat("Timestep dt (min)", &c.dt, 0.01f, 0.5f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::TextWrapped("Active: %s%s", app.backendStatus.c_str(), app.interopActive ? " | interop ON" : "");
    }
    // ---- view
    if (ImGui::CollapsingHeader("Camera & view", ImGuiTreeNodeFlags_DefaultOpen)) {
        int m = (int)app.cam.mode;
        for (int i = 0; i < (int)CameraMode::Count; ++i) {
            if (i) ImGui::SameLine();
            ImGui::BeginDisabled(i == (int)CameraMode::Follow && app.renderSettings.selected < 0);
            if (ImGui::RadioButton(cameraModeName((CameraMode)i), m == i)) app.cam.setMode((CameraMode)i);
            ImGui::EndDisabled();
        }
        ImGui::TextColored(kDim, "LMB drag pan | RMB drag rotate | wheel zoom | WASD/QE\n"
                                 "click patient to select | 1-4 cameras | Space start/pause");
        ImGui::Checkbox("Labels", &app.showLabels);
        ImGui::SameLine();
        ImGui::Checkbox("Staff", &app.renderSettings.showStaff);
        ImGui::Text("Show:");
        for (int p = 0; p < 4; ++p) {
            ImGui::SameLine();
            bool shown = ((app.renderSettings.hidePriorityMask >> p) & 1) == 0;
            ImGui::PushStyleColor(ImGuiCol_CheckMark, kPrioColor[p]);
            char lbl[8];
            std::snprintf(lbl, sizeof(lbl), "P%d", p + 1);
            if (ImGui::Checkbox(lbl, &shown)) app.renderSettings.hidePriorityMask ^= (1 << p);
            ImGui::PopStyleColor();
        }
    }
    ImGui::EndDisabled();
    ImGui::End();
}

// ------------------------------------------------------------- live metrics ----
void plot(const char* label, const std::vector<float>& v, float maxV, float h = 55.f) {
    if (v.empty()) return;
    const int n = (int)v.size();
    const int show = std::min(n, 600);
    float mx = maxV;
    if (mx <= 0) {
        mx = 1;
        for (int i = n - show; i < n; ++i) mx = std::max(mx, v[i]);
    }
    char ov[64];
    std::snprintf(ov, sizeof(ov), "%s: %.0f", label, v.back());
    ImGui::PlotLines("##p", v.data() + (n - show), show, 0, ov, 0.f, mx * 1.05f, {-1, h});
}

void metricsTab(App& app) {
    const LiveMetrics& m = app.metrics;
    ImGui::TextColored({0.55f, 0.8f, 1.f, 1.f}, "LIVE SIMULATION STATISTICS");
    ImGui::Text("Sim time %s  |  step %llu  |  %s", fmtTime(m.simTime).c_str(),
                (unsigned long long)app.engine.stepIndex(), scenarioName(app.engine.config().type));
    const float arrived = m.total ? (float)m.arrived / m.total : 0.f, dis = m.total ? (float)m.discharged / m.total : 0.f;
    char b[64];
    std::snprintf(b, sizeof(b), "arrived %d / %d", m.arrived, m.total);
    ImGui::ProgressBar(arrived, {-1, 0}, b);
    std::snprintf(b, sizeof(b), "discharged %d / %d", m.discharged, m.total);
    ImGui::ProgressBar(dis, {-1, 0}, b);

    if (ImGui::BeginTable("counts", 4, ImGuiTableFlags_SizingStretchSame)) {
        auto cell = [](const char* k, int v) {
            ImGui::TableNextColumn();
            ImGui::TextColored(kDim, "%s", k);
            ImGui::Text("%d", v);
        };
        cell("Active", m.active);
        cell("Waiting", m.waiting);
        cell("Under treatment", m.inTreatment);
        cell("In service", m.inService);
        ImGui::EndTable();
    }
    ImGui::Text("Active by priority:");
    for (int p = 0; p < 4; ++p) {
        ImGui::SameLine();
        ImGui::TextColored(kPrioColor[p], "P%d %d", p + 1, m.activeByPriority[p]);
    }
    ImGui::SameLine();
    ImGui::TextColored(kDim, "untriaged %d", m.untriaged);

    ImGui::SeparatorText("Resources");
    if (ImGui::BeginTable("res", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Resource");
        ImGui::TableSetupColumn("In use");
        ImGui::TableSetupColumn("Occupancy", ImGuiTableColumnFlags_WidthStretch, 2.f);
        ImGui::TableSetupColumn("Utilisation");
        ImGui::TableHeadersRow();
        for (int r = 0; r < R_COUNT; ++r) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(resourceName(r));
            ImGui::TableNextColumn();
            ImGui::Text("%d / %d", m.busy[r], m.capacity[r]);
            ImGui::TableNextColumn();
            const float occ = m.capacity[r] ? (float)m.busy[r] / m.capacity[r] : 0.f;
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                                  occ > 0.95f ? kBad : (occ > 0.75f ? ImVec4(1, 0.7f, 0.2f, 1) : kGood));
            ImGui::ProgressBar(occ, {-1, 0});
            ImGui::PopStyleColor();
            ImGui::TableNextColumn();
            ImGui::Text("%.0f%%", 100.0 * m.utilization[r]);
        }
        ImGui::EndTable();
    }
    ImGui::TextColored(kDim, "Available: %d doctors, %d ER beds, %d ICU beds, %d ward beds",
                       m.capacity[R_DOCTOR] - m.busy[R_DOCTOR], m.capacity[R_ER_BED] - m.busy[R_ER_BED],
                       m.capacity[R_ICU_BED] - m.busy[R_ICU_BED], m.capacity[R_WARD_BED] - m.busy[R_WARD_BED]);

    ImGui::SeparatorText("Queues & bottlenecks");
    if (ImGui::BeginTable("q", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Queue", ImGuiTableColumnFlags_WidthStretch, 2.f);
        ImGui::TableSetupColumn("Waiting");
        ImGui::TableSetupColumn("Avg wait");
        ImGui::TableSetupColumn("Max wait");
        ImGui::TableSetupColumn("Served");
        ImGui::TableHeadersRow();
        for (int q = 0; q < Q_COUNT; ++q) {
            const QueueStats& s = m.queues[q];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (q == m.slowestQueue) ImGui::TextColored(kBad, "%s", queueName(q));
            else ImGui::TextUnformatted(queueName(q));
            ImGui::TableNextColumn();
            ImGui::Text("%d", s.length);
            ImGui::TableNextColumn();
            ImGui::Text("%.1f m", s.avgWait);
            ImGui::TableNextColumn();
            ImGui::Text("%.0f m", s.maxWait);
            ImGui::TableNextColumn();
            ImGui::Text("%d", s.served);
        }
        ImGui::EndTable();
    }
    if (m.largestQueue >= 0)
        ImGui::TextColored(kBad, "Largest queue: %s (%d waiting)", queueName(m.largestQueue),
                           m.queues[m.largestQueue].length);
    if (m.slowestQueue >= 0)
        ImGui::TextColored(kBad, "Longest average wait: %s (%.1f min)", queueName(m.slowestQueue),
                           m.queues[m.slowestQueue].avgWait);

    ImGui::SeparatorText("Patient-level indicators");
    if (ImGui::BeginTable("ind", 2, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)) {
        kv("Average waiting time", "%.1f min  (GPU/backend reduction)", m.avgWait);
        kv("  cross-check (CPU queue log)", "%.1f min", m.avgWaitCpu);
        kv("Maximum waiting time", "%.1f min", m.maxWait);
        kv("Avg treatment delay", "%.1f min (arrival -> ER treatment, n=%d)", m.avgTreatmentDelay, m.treatmentStarted);
        kv("Avg length of stay", "%.1f min (discharged)", m.avgLengthOfStay);
        kv("Throughput", "%.1f discharges / hour", m.throughputPerHour);
        kv("Priority escalations", "%d", m.escalations);
        kv("Avg synthetic health", "%.1f (active patients)", m.avgHealth);
        ImGui::EndTable();
    }
    ImGui::SeparatorText("Time series (5-min samples)");
    const TimeSeries& ts = app.engine.series();
    plot("ER queue", ts.queueER, 0);
    plot("Triage queue", ts.queueTriage, 0);
    plot("Boarding (ICU+ward)", ts.queueBoarding, 0);
    plot("ER occupancy %", ts.erOcc, 100);
    plot("ICU occupancy %", ts.icuOcc, 100);
    plot("Ward occupancy %", ts.wardOcc, 100);
}

// --------------------------------------------------------- live performance ----
void performanceTab(App& app) {
    ImGui::TextColored({1.f, 0.8f, 0.4f, 1.f}, "LIVE PERFORMANCE (current run, not benchmark results)");
    PatientBackend* be = app.engine.backend();
    const BackendTimings& t = be->timings();
    const EngineProfile& p = app.engine.profile();
    const double steps = std::max<double>(1, (double)p.steps);
    const int n = be->count();
    if (ImGui::BeginTable("perf", 2, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)) {
        kv("Backend", "%s", be->name().c_str());
        kv("Executes per-patient update on", "%s", be->isGpu() ? "GPU (CUDA kernels)" : "CPU");
        kv("Patients (array size / active)", "%d / %d", n, app.metrics.active);
        kv("Last step: wall time", "%.3f ms", t.stepWallMs);
        kv(be->isGpu() ? "Last step: CUDA kernel time" : "Last step: CPU update loop", "%.3f ms", t.computeMs);
        kv("Last step: H2D / D2H transfer", "%.3f / %.3f ms", t.h2dMs, t.d2hMs);
        kv("Avg backend step (wall)", "%.3f ms", p.backendMs / steps);
        kv("Avg kernel / compute", "%.3f ms", t.totalComputeMs / std::max<double>(1, (double)t.steps));
        kv("Avg CPU controller (queues, allocation)", "%.3f ms", p.controllerMs / steps);
        kv("Avg command upload", "%.3f ms", p.commandMs / steps);
        kv("Avg bytes / step H2D / D2H", "%.1f KB / %.1f KB", t.bytesH2D / steps / 1024.0, t.bytesD2H / steps / 1024.0);
        kv("Avg events / commands per step", "%.1f / %.1f", p.events / steps, p.commands / steps);
        kv("Steps last frame", "%d (%.0f steps/s)", app.stepsLastFrame, app.stepsPerSecond);
        kv("Patient updates / s", "%.2f M", app.stepsPerSecond * n / 1e6);
        kv("Total simulation time (wall)", "%.1f ms backend + %.1f ms controller", p.backendMs, p.controllerMs);
        kv("Frame time / FPS", "%.2f ms / %.0f", app.frameMs, app.frameMs > 0 ? 1000.0 / app.frameMs : 0.0);
        kv("Render data update", "%.3f ms %s", app.uploadMs, app.interopActive ? "(CUDA-GL interop)" : "(copy + glBufferSubData)");
        ImGui::EndTable();
    }
    ImGui::SeparatorText("Live GPU speedup at the current state");
    ImGui::TextWrapped("Takes a snapshot of the current patient arrays and runs 200 update steps on the CPU "
                       "(1 thread and all threads) and with the optimized CUDA kernel. Results are checked for "
                       "bit-identical output.");
    ImGui::BeginDisabled(app.job.busy() || !app.gpuAvailable());
    if (ImGui::Button("Measure live speedup")) app.startLiveSpeedup();
    ImGui::EndDisabled();
    LiveSpeedup ls;
    {
        std::lock_guard<std::mutex> l(app.job.resultMutex);
        ls = app.liveSpeedup;
    }
    if (ls.valid && ImGui::BeginTable("ls", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Implementation");
        ImGui::TableSetupColumn("200 steps");
        ImGui::TableSetupColumn("Speedup vs CPU-1T");
        ImGui::TableHeadersRow();
        auto row = [&](const char* k, double ms) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(k);
            ImGui::TableNextColumn();
            ImGui::Text("%.2f ms", ms);
            ImGui::TableNextColumn();
            ImGui::Text("%.2fx", ms > 0 ? ls.cpu1Ms / ms : 0.0);
        };
        row("CPU single-thread", ls.cpu1Ms);
        row("CPU multi-thread", ls.cpuMtMs);
        row("CUDA kernel only", ls.gpuKernelMs);
        row("CUDA end-to-end", ls.gpuE2eMs);
        ImGui::EndTable();
        ImGui::Text("N = %d (%d active).  Output identical: ", ls.patients, ls.active);
        ImGui::SameLine();
        ImGui::TextColored(ls.identical ? kGood : kBad, ls.identical ? "yes" : "NO");
    }
    if (!ls.error.empty()) ImGui::TextColored(kBad, "%s", ls.error.c_str());

    ImGui::SeparatorText("Device");
    const GpuDeviceInfo& g = app.gpu;
    if (g.available) {
        ImGui::Text("%s  (compute capability %d.%d)", g.name.c_str(), g.ccMajor, g.ccMinor);
        ImGui::TextColored(kDim, "%d SMs | %zu MiB | %d-bit bus | L2 %d KiB | %zu KiB smem/block | warp %d",
                           g.smCount, g.totalMemBytes >> 20, g.memBusWidth, g.l2CacheBytes >> 10,
                           g.sharedMemPerBlock >> 10, g.warpSize);
    } else {
        ImGui::TextColored(kBad, "No CUDA GPU: %s", g.error.c_str());
    }
}

// --------------------------------------------------------------- benchmarks ----
void benchmarkTab(App& app) {
    ImGui::TextColored({1.f, 0.8f, 0.4f, 1.f}, "BENCHMARK RESULTS (deterministic synthetic workload)");
    ImGui::TextWrapped("Timing definitions: kernel = CUDA events around back-to-back launches (data resident). "
                       "E2E = wall clock incl. initial upload, per-step event transfer, final download. "
                       "Speedup = CPU single-thread time / GPU time.");
    static bool scenario = false;
    const bool busy = app.job.busy();
    ImGui::BeginDisabled(busy);
    if (ImGui::Button("Run quick benchmark")) app.startBenchmark(true, scenario);
    ImGui::SameLine();
    if (ImGui::Button("Run full benchmark")) app.startBenchmark(false, scenario);
    ImGui::SameLine();
    ImGui::Checkbox("+ scenario suite", &scenario);
    ImGui::EndDisabled();
    if (busy) {
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) app.job.cancel = true;
        std::string msg;
        float f;
        app.job.getProgress(msg, f);
        ImGui::ProgressBar(f, {-1, 0}, (app.job.name() + ": " + msg).c_str());
    }
    std::vector<KernelBenchRow> rows;
    std::vector<ScenarioBenchRow> srows;
    std::string message;
    {
        std::lock_guard<std::mutex> l(app.job.resultMutex);
        rows = app.kernelRows;
        srows = app.scenarioRows;
        message = app.benchMessage;
    }
    if (!message.empty()) ImGui::TextColored(kDim, "%s", message.c_str());
    if (rows.empty()) {
        ImGui::TextColored(kDim, "No results yet. CLI: hospital_sim --benchmark (writes benchmarks/results/*.csv)");
        return;
    }
    static int exp = 0;
    const char* exps[] = {"size_sweep", "block_size", "ordering", "active_list"};
    ImGui::Combo("Experiment", &exp, exps, 4);
    const ImGuiTableFlags tf = ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY |
                               ImGuiTableFlags_SizingFixedFit;
    if (ImGui::BeginTable("bench", 9, tf, {0, 330})) {
        ImGui::TableSetupScrollFreeze(0, 1);
        for (const char* h : {"Variant", "N", "Block", "Extra", "Kernel ms", "E2E ms", "xKernel", "xE2E", "Match"})
            ImGui::TableSetupColumn(h);
        ImGui::TableHeadersRow();
        for (const auto& r : rows) {
            if (r.experiment != exps[exp]) continue;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r.variant.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%d", r.patients);
            ImGui::TableNextColumn();
            if (r.isGpu) ImGui::Text("%d", r.blockSize);
            else ImGui::Text("%d thr", r.threads);
            ImGui::TableNextColumn();
            if (!r.ordering.empty()) ImGui::TextUnformatted(r.ordering.c_str());
            else if (r.experiment == "active_list") ImGui::Text("%.0f%% idle", 100 * r.inactiveFraction);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", r.kernelMs);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", r.e2eMs);
            ImGui::TableNextColumn();
            ImGui::Text("%.2f", r.speedupKernel);
            ImGui::TableNextColumn();
            ImGui::Text("%.2f", r.speedupE2E);
            ImGui::TableNextColumn();
            const bool ok = r.eventsMatch && r.intMismatches == 0 && r.maxAbsDiff <= 1e-4;
            ImGui::TextColored(ok ? kGood : kBad, ok ? "yes" : "NO");
        }
        ImGui::EndTable();
    }
    if (!srows.empty() && ImGui::CollapsingHeader("Scenario suite (full hybrid simulation)")) {
        if (ImGui::BeginTable("sb", 7, tf, {0, 220})) {
            for (const char* h : {"Backend", "N", "Wall ms", "Backend ms", "Controller ms", "xWall", "Identical"})
                ImGui::TableSetupColumn(h);
            ImGui::TableHeadersRow();
            for (const auto& r : srows) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(r.backend.c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%d", r.patients);
                ImGui::TableNextColumn();
                ImGui::Text("%.0f", r.wallMs);
                ImGui::TableNextColumn();
                ImGui::Text("%.0f", r.backendMs);
                ImGui::TableNextColumn();
                ImGui::Text("%.0f", r.controllerMs);
                ImGui::TableNextColumn();
                ImGui::Text("%.2f", r.speedupWall);
                ImGui::TableNextColumn();
                ImGui::TextColored(r.identicalToCpu ? kGood : kBad, r.identicalToCpu ? "yes" : "NO");
            }
            ImGui::EndTable();
        }
    }
}

// ---------------------------------------------------------- decision support ----
void compareTab(App& app) {
    ImGui::TextColored({0.55f, 0.8f, 1.f, 1.f}, "CONFIGURATION COMPARISON (what-if)");
    ImGui::TextWrapped("Runs the same deterministic scenario (same seed, same arrivals) to completion under "
                       "configuration A (Control Panel) and B (below). These are simulation results, not medical "
                       "recommendations.");
    ScenarioConfig& b = app.cfgB;
    const Capacities& a = app.cfg.cap;
    if (ImGui::BeginTable("cfg", 3, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Resource");
        ImGui::TableSetupColumn("A");
        ImGui::TableSetupColumn("B", ImGuiTableColumnFlags_WidthStretch, 3.f);
        ImGui::TableHeadersRow();
        auto row = [](const char* n, int av, int& bv, int mx) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(n);
            ImGui::TableNextColumn();
            ImGui::Text("%d", av);
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderInt((std::string("##") + n).c_str(), &bv, 1, mx);
        };
        row("Doctors", a.doctors, b.cap.doctors, 200);
        row("Triage nurses", a.nurses, b.cap.nurses, 100);
        row("ER beds", a.erBeds, b.cap.erBeds, 300);
        row("ICU beds", a.icuBeds, b.cap.icuBeds, 200);
        row("Ward beds", a.wardBeds, b.cap.wardBeds, 1000);
        row("Lab slots", a.labSlots, b.cap.labSlots, 60);
        ImGui::EndTable();
    }
    if (ImGui::Button("Copy A -> B")) {
        const Capacities keep = app.cfg.cap;
        app.cfgB = app.cfg;
        app.cfgB.cap = keep;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(200);
    backendCombo("Engine", app.compareBackend, app.gpuAvailable());
    // B uses the same scenario as A, only resources differ
    const Capacities capB = b.cap;
    b = app.cfg;
    b.cap = capB;
    ImGui::BeginDisabled(app.job.busy());
    if (ImGui::Button("Run comparison", {-1, 0})) app.startComparison();
    ImGui::EndDisabled();
    if (app.job.busy() && app.job.name() == "Configuration comparison") {
        std::string msg;
        float f;
        app.job.getProgress(msg, f);
        ImGui::ProgressBar(f, {-1, 0}, msg.c_str());
    }
    ConfigRunResult ra, rb;
    bool valid;
    {
        std::lock_guard<std::mutex> l(app.job.resultMutex);
        ra = app.compareA;
        rb = app.compareB;
        valid = app.compareValid;
    }
    if (!valid) {
        if (!ra.error.empty() || !rb.error.empty()) ImGui::TextColored(kBad, "%s %s", ra.error.c_str(), rb.error.c_str());
        return;
    }
    const LiveMetrics &A = ra.metrics, &B = rb.metrics;
    if (ImGui::BeginTable("cmp", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthStretch, 2.f);
        ImGui::TableSetupColumn("A");
        ImGui::TableSetupColumn("B");
        ImGui::TableSetupColumn("B - A");
        ImGui::TableHeadersRow();
        auto row = [](const char* n, double x, double y, bool lowerIsBetter, const char* fmt = "%.1f") {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(n);
            ImGui::TableNextColumn();
            ImGui::Text(fmt, x);
            ImGui::TableNextColumn();
            ImGui::Text(fmt, y);
            ImGui::TableNextColumn();
            const double d = y - x;
            const bool better = lowerIsBetter ? d < 0 : d > 0;
            ImGui::TextColored(std::fabs(d) < 1e-9 ? kDim : (better ? kGood : kBad), "%+.1f", d);
        };
        row("Avg waiting time (min)", A.avgWait, B.avgWait, true);
        row("Max waiting time (min)", A.maxWait, B.maxWait, true);
        row("Avg treatment delay (min)", A.avgTreatmentDelay, B.avgTreatmentDelay, true);
        row("Avg length of stay (min)", A.avgLengthOfStay, B.avgLengthOfStay, true);
        row("Time to clear all patients (h)", A.simTime / 60, B.simTime / 60, true);
        row("Throughput (discharges/h)", A.throughputPerHour, B.throughputPerHour, false);
        row("Priority escalations", A.escalations, B.escalations, true, "%.0f");
        for (int q = 0; q < Q_COUNT; ++q) {
            std::string n = std::string("Avg wait: ") + queueName(q);
            row(n.c_str(), A.queues[q].avgWait, B.queues[q].avgWait, true);
        }
        row("Doctor utilisation %", 100 * A.utilization[R_DOCTOR], 100 * B.utilization[R_DOCTOR], false);
        row("ER bed utilisation %", 100 * A.utilization[R_ER_BED], 100 * B.utilization[R_ER_BED], false);
        row("ICU bed utilisation %", 100 * A.utilization[R_ICU_BED], 100 * B.utilization[R_ICU_BED], false);
        row("Ward bed utilisation %", 100 * A.utilization[R_WARD_BED], 100 * B.utilization[R_WARD_BED], false);
        ImGui::EndTable();
    }
    ImGui::Text("Bottleneck (longest avg wait):  A = %s,  B = %s", A.slowestQueue >= 0 ? queueName(A.slowestQueue) : "-",
                B.slowestQueue >= 0 ? queueName(B.slowestQueue) : "-");
    ImGui::TextColored(kDim, "Headless run time: A %.0f ms, B %.0f ms (%s)", ra.wallMs, rb.wallMs,
                       backendKindName(app.compareBackend));
}

// -------------------------------------------------------------- patient tab ----
void patientTab(App& app) {
    const auto& recs = app.engine.records();
    int sel = app.renderSettings.selected;
    if (ImGui::Button("Select a random active P1")) {
        std::vector<int> c;
        for (const auto& r : recs)
            if (inMask(kActiveMask, r.state) && r.priority == P1) c.push_back(r.id);
        if (!c.empty()) app.selectPatient(c[(size_t)(app.engine.stepIndex() * 2654435761u) % c.size()]);
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("ID", &sel)) app.selectPatient(std::clamp(sel, -1, (int)recs.size() - 1));
    sel = app.renderSettings.selected;
    if (sel < 0 || sel >= (int)recs.size()) {
        ImGui::TextColored(kDim, "Click a patient in the 3D view to inspect it.");
        return;
    }
    if (ImGui::Button(app.cam.mode == CameraMode::Follow ? "Stop following" : "Follow with camera"))
        app.cam.setMode(app.cam.mode == CameraMode::Follow ? CameraMode::Orbit : CameraMode::Follow);
    const PatientRecord& r = recs[sel];
    PatientAoS h{};
    app.engine.backend()->readPatient(sel, h);
    if (ImGui::BeginTable("pt", 2, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)) {
        kv("Patient ID", "%d", r.id);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(kDim, "Priority");
        ImGui::TableNextColumn();
        if (r.flags & FLAG_TRIAGED)
            ImGui::TextColored(kPrioColor[r.priority], "%s%s", priorityName(r.priority),
                               r.escalations ? " (escalated)" : "");
        else
            ImGui::Text("not yet triaged");
        kv("State", "%s", stateName(r.state));
        kv("Department", "%s", app.engine.layout().dept(r.dept).name);
        kv("Arrival", "%s%s", fmtTime(r.arrivalTime).c_str(), r.ambulance ? " (ambulance)" : "");
        kv("Position", "(%.1f, %.1f) -> (%.1f, %.1f)", h.x, h.z, h.tx, h.tz);
        kv("Waiting time (GPU)", "%.1f min", h.waitingTime);
        kv("Remaining service", "%.1f min", h.serviceRemaining);
        kv("Treatment duration", "%.1f min", r.treatDur);
        kv("Synthetic health", "%.1f", h.health);
        kv("Pathway", "%s%s", r.needsDiag ? "diagnostics, " : "",
           r.disposition == DISP_ICU ? "ICU" : (r.disposition == DISP_WARD ? "ward" : "home"));
        if (r.doctor >= 0) kv("Assigned doctor", "#%d", r.doctor);
        if (r.erBed >= 0) kv("ER bed", "#%d", r.erBed);
        if (r.icuBed >= 0) kv("ICU bed", "#%d", r.icuBed);
        if (r.wardBed >= 0) kv("Ward bed", "#%d", r.wardBed);
        if (r.treatmentStart >= 0) kv("Treatment start", "%s", fmtTime(r.treatmentStart).c_str());
        if (r.dischargeTime >= 0) kv("Discharged", "%s", fmtTime(r.dischargeTime).c_str());
        ImGui::EndTable();
    }
}

}  // namespace

void drawDashboard(App& app) {
    controlPanel(app);
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float pad = 8.f, w = 500.f * ImGui::GetFontSize() / 14.f;
    ImGui::SetNextWindowPos({vp->WorkPos.x + vp->WorkSize.x - w - pad, vp->WorkPos.y + pad}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({w, vp->WorkSize.y - 2 * pad}, ImGuiCond_Once);
    ImGui::Begin("Hospital Dashboard");
    // screenshot mode can pre-select a tab (documentation / smoke test of every panel)
    auto tabFlag = [&](int i) {
        return (!app.screenshotPath.empty() && app.screenshotTab == i) ? ImGuiTabItemFlags_SetSelected : 0;
    };
    if (ImGui::BeginTabBar("tabs")) {
        if (ImGui::BeginTabItem("Live Metrics", nullptr, tabFlag(0))) {
            metricsTab(app);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Performance", nullptr, tabFlag(1))) {
            performanceTab(app);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Benchmarks", nullptr, tabFlag(2))) {
            benchmarkTab(app);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Decision Support", nullptr, tabFlag(3))) {
            compareTab(app);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Patient", nullptr, tabFlag(4))) {
            patientTab(app);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void drawOverlay(App& app, const glm::mat4& viewProj) {
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    auto project = [&](glm::vec3 p, ImVec2& out) {
        glm::vec4 c = viewProj * glm::vec4(p, 1.f);
        if (c.w <= 0.1f) return false;
        c /= c.w;
        if (c.x < -1.2f || c.x > 1.2f || c.y < -1.2f || c.y > 1.2f) return false;
        out = {(c.x * 0.5f + 0.5f) * ds.x, (1.f - (c.y * 0.5f + 0.5f)) * ds.y};
        return true;
    };
    const HospitalLayout& L = app.engine.layout();
    if (app.showLabels) {
        const LiveMetrics& m = app.metrics;
        for (int d = 0; d < D_COUNT; ++d) {
            if (d == D_CORRIDOR) continue;
            const Vec2 c = L.dept(d).area.center();
            ImVec2 s;
            if (!project({c.x, 3.0f, c.z}, s)) continue;
            std::string label = L.dept(d).name;
            if (d == D_ER) label += "  " + std::to_string(m.busy[R_ER_BED]) + "/" + std::to_string(m.capacity[R_ER_BED]);
            if (d == D_ICU) label += "  " + std::to_string(m.busy[R_ICU_BED]) + "/" + std::to_string(m.capacity[R_ICU_BED]);
            if (d == D_WARD) label += "  " + std::to_string(m.busy[R_WARD_BED]) + "/" + std::to_string(m.capacity[R_WARD_BED]);
            if (d == D_WAITING) label += "  queue " + std::to_string(m.queues[Q_ER].length);
            if (d == D_ENTRANCE) label += "  triage queue " + std::to_string(m.queues[Q_TRIAGE].length);
            const ImVec2 ts = ImGui::CalcTextSize(label.c_str());
            const ImVec2 p0(s.x - ts.x / 2 - 5, s.y - ts.y / 2 - 3), p1(s.x + ts.x / 2 + 5, s.y + ts.y / 2 + 3);
            dl->AddRectFilled(p0, p1, IM_COL32(15, 20, 30, 170), 4.f);
            dl->AddText({p0.x + 5, p0.y + 3}, IM_COL32(240, 245, 255, 255), label.c_str());
        }
    }
    const int sel = app.renderSettings.selected;
    if (sel >= 0 && sel < (int)app.renderData.size()) {
        ImVec2 s;
        const RenderInstance& ri = app.renderData[sel];
        if (project({ri.x, 2.6f, ri.z}, s)) {
            char b[64];
            std::snprintf(b, sizeof(b), "#%d %s", sel, stateName(app.engine.records()[sel].state));
            dl->AddText({s.x - ImGui::CalcTextSize(b).x / 2 + 1, s.y + 1}, IM_COL32(0, 0, 0, 200), b);
            dl->AddText({s.x - ImGui::CalcTextSize(b).x / 2, s.y}, IM_COL32(255, 255, 120, 255), b);
        }
    }
    // status + legend (bottom centre), sized from the actual font metrics
    char status[256];
    std::snprintf(status, sizeof(status), "%s  |  %s  |  t = %s  |  active %d  waiting %d  discharged %d/%d",
                  app.running ? "RUNNING" : "PAUSED", app.engine.backend()->isGpu() ? "GPU (CUDA)" : "CPU",
                  fmtTime(app.metrics.simTime).c_str(), app.metrics.active, app.metrics.waiting,
                  app.metrics.discharged, app.metrics.total);
    const char* pn[5] = {"P1 Critical", "P2 Serious", "P3 Moderate", "P4 Minor", "untriaged"};
    const ImU32 pc[5] = {ImGui::ColorConvertFloat4ToU32(kPrioColor[0]), ImGui::ColorConvertFloat4ToU32(kPrioColor[1]),
                         ImGui::ColorConvertFloat4ToU32(kPrioColor[2]), ImGui::ColorConvertFloat4ToU32(kPrioColor[3]),
                         IM_COL32(217, 219, 230, 255)};
    const char* note = "head marker = state (pulsing = waiting)";
    const float fs = ImGui::GetFontSize(), sw = fs * 0.8f, gap = fs;
    float legendW = ImGui::CalcTextSize(note).x;
    for (const char* s : pn) legendW += sw + 4 + ImGui::CalcTextSize(s).x + gap;
    const float boxW = std::max(ImGui::CalcTextSize(status).x, legendW) + 2 * gap;
    const float boxH = fs * 3.2f;
    const float lx = ds.x * 0.5f - boxW * 0.5f, ly = ds.y - boxH - 8.f;
    dl->AddRectFilled({lx, ly}, {lx + boxW, ly + boxH}, IM_COL32(15, 20, 30, 210), 6.f);
    dl->AddText({lx + gap, ly + fs * 0.35f}, app.running ? IM_COL32(120, 255, 150, 255) : IM_COL32(255, 210, 120, 255),
                status);
    float x = lx + gap;
    const float y2 = ly + fs * 1.75f;
    for (int p = 0; p < 5; ++p) {
        dl->AddRectFilled({x, y2 + fs * 0.1f}, {x + sw, y2 + fs * 0.1f + sw}, pc[p], 2.f);
        x += sw + 4;
        dl->AddText({x, y2}, IM_COL32(230, 230, 230, 255), pn[p]);
        x += ImGui::CalcTextSize(pn[p]).x + gap;
    }
    dl->AddText({x, y2}, IM_COL32(170, 175, 185, 255), note);
}
}  // namespace hs

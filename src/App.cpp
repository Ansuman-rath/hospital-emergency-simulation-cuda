#include "App.h"

#include <glad/gl.h>
// glad must come first
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>

#include "Dashboard.h"
#include "PatientUpdate.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace hs {

void BackgroundJob::start(const std::string& name, std::function<void(BackgroundJob&)> fn) {
    join();
    cancel = false;
    {
        std::lock_guard<std::mutex> l(m_);
        name_ = name;
        msg_ = "starting...";
        progress_ = 0;
    }
    running_ = true;
    th_ = std::thread([this, fn]() {
        try {
            fn(*this);
        } catch (const std::exception& e) {
            setProgress(std::string("error: ") + e.what(), 1.f);
        }
        running_ = false;
    });
}

static std::string findShaderDir() {
    namespace fs = std::filesystem;
    std::vector<fs::path> cands;
#ifdef _WIN32
    char buf[MAX_PATH] = {};
    if (GetModuleFileNameA(nullptr, buf, MAX_PATH)) cands.push_back(fs::path(buf).parent_path() / "shaders");
#endif
    cands.push_back(fs::current_path() / "shaders");
#ifdef HS_SHADER_DIR
    cands.push_back(fs::path(HS_SHADER_DIR));
#endif
    for (const auto& c : cands)
        if (fs::exists(c / "patient.vert")) return c.string();
    return "shaders";
}

static void glfwErrorCb(int code, const char* msg) { std::fprintf(stderr, "GLFW error %d: %s\n", code, msg); }

// --------------------------------------------------------------------- run ----
int App::run() {
    gpu = queryGpu();
    if (!gpu.available) backendKind = BackendKind::Cpu1;
    cfg = makePreset(ScenarioType::Normal);
    cfgB = cfg;
    cfgB.cap.doctors += 5;
    cfgB.cap.icuBeds += 10;

    glfwSetErrorCallback(glfwErrorCb);
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
    window_ = glfwCreateWindow(width, height, "GPU-Accelerated 3D Hospital Emergency Simulation", nullptr, nullptr);
    if (!window_) {
        std::fprintf(stderr, "Failed to create an OpenGL 3.3 core window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);
    if (!gladLoadGL(glfwGetProcAddress)) {
        std::fprintf(stderr, "Failed to load OpenGL functions\n");
        return 1;
    }
    std::printf("OpenGL %s | %s | %s\n", (const char*)glGetString(GL_VERSION), (const char*)glGetString(GL_VENDOR),
                (const char*)glGetString(GL_RENDERER));
    glEnable(GL_MULTISAMPLE);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    float xs = 1.f, ys = 1.f;
    glfwGetWindowContentScale(window_, &xs, &ys);
    const float uiScale = std::max(1.f, xs);
    ImFontConfig fc;
    fc.SizePixels = 14.f * uiScale;
    io.Fonts->AddFontDefault(&fc);
    ImGui::StyleColorsDark();
    ImGuiStyle& st = ImGui::GetStyle();
    st.WindowRounding = 6.f;
    st.FrameRounding = 4.f;
    st.GrabRounding = 4.f;
    st.WindowBorderSize = 0.f;
    st.Colors[ImGuiCol_WindowBg].w = 0.92f;
    st.ScaleAllSizes(uiScale);
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    std::string err;
    const std::string shaderDir = findShaderDir();
    if (!renderer_.init(shaderDir, &err)) {
        std::fprintf(stderr, "Renderer init failed (%s): %s\n", shaderDir.c_str(), err.c_str());
        return 1;
    }
    if (screenshotScenario >= 0) {
        applyPreset((ScenarioType)screenshotScenario);
        if (screenshotPatients > 0) cfg = scaleScenario(cfg, screenshotPatients);
        backendKind = gpu.available ? BackendKind::CudaOptimized : BackendKind::Cpu1;
        useInterop = gpu.available;
    }
    resetSimulation();
    if (!screenshotPath.empty()) {
        if (screenshotSimMinutes > 0) engine.advance(screenshotSimMinutes, -1);
        if (screenshotCamera >= 0) cam.setMode((CameraMode)screenshotCamera);
        if (screenshotCamera == (int)CameraMode::Orbit) {
            cam.target = {-14.f, 0.f, 2.f};
            cam.distance = 120.f;
            cam.pitch = 48.f;
            cam.yaw = -100.f;
        }
        dirtyRender_ = true;
        metricsAge = 1e9;
    }
    int frameNo = 0;

    auto last = std::chrono::steady_clock::now();
    const auto start = last;
    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();
        const auto now = std::chrono::steady_clock::now();
        const double dt = std::min(0.1, std::chrono::duration<double>(now - last).count());
        last = now;
        frameMs = frameMs * 0.9 + dt * 1000.0 * 0.1;

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        handleInput(dt);
        if (jobWasBusy_ && !job.busy()) {
            job.join();
            running = wasRunningBeforeJob_;
            jobWasBusy_ = false;
            metricsAge = 1e9;
        }
        if (!job.busy()) advanceSimulation(dt);
        updateRenderData(false);
        updateStaff(dt);
        metricsAge += dt;
        if (metricsAge > 0.2 && !job.busy()) {
            metrics = engine.computeMetrics();
            metricsAge = 0;
        }
        if (cam.mode == CameraMode::Follow && renderSettings.selected >= 0 &&
            renderSettings.selected < (int)renderData.size()) {
            const RenderInstance& ri = renderData[renderSettings.selected];
            cam.follow({ri.x, 0.8f, ri.z}, (float)dt);
        }

        int fbw = 0, fbh = 0;
        glfwGetFramebufferSize(window_, &fbw, &fbh);
        width = fbw;
        height = fbh;
        const float t = (float)std::chrono::duration<double>(now - start).count();
        if (fbw > 0 && fbh > 0) {
            renderer_.render(cam, fbw, fbh, t, renderSettings);
            drawOverlay(*this, cam.projection((float)fbw / fbh) * cam.view());
        }
        drawDashboard(*this);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        if (!screenshotPath.empty() && ++frameNo == 20) {
            saveScreenshot(screenshotPath);
            glfwSetWindowShouldClose(window_, GLFW_TRUE);
        }
        glfwSwapBuffers(window_);
    }

    job.cancel = true;
    job.join();
    if (engine.backend()) engine.backend()->unregisterRenderBuffer();
    renderer_.shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window_);
    glfwTerminate();
    return 0;
}

void App::saveScreenshot(const std::string& path) {
    // 24-bit BMP of the back buffer (after ImGui), bottom-up rows as stored by OpenGL.
    const int w = width, h = height, stride = (w * 3 + 3) & ~3;
    std::vector<unsigned char> px((size_t)stride * h);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, px.data());
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    const uint32_t dataSize = (uint32_t)px.size(), fileSize = 54 + dataSize;
    unsigned char hdr[54] = {'B', 'M'};
    auto put32 = [&](int off, uint32_t v) {
        for (int i = 0; i < 4; ++i) hdr[off + i] = (unsigned char)(v >> (8 * i));
    };
    put32(2, fileSize);
    put32(10, 54);
    put32(14, 40);
    put32(18, (uint32_t)w);
    put32(22, (uint32_t)h);
    hdr[26] = 1;
    hdr[28] = 24;
    put32(34, dataSize);
    std::fwrite(hdr, 1, 54, f);
    std::fwrite(px.data(), 1, px.size(), f);
    std::fclose(f);
    std::printf("screenshot saved: %s (%dx%d)\n", path.c_str(), w, h);
}

// --------------------------------------------------------------- simulation ----
void App::resetSimulation() {
    if (job.busy()) return;
    if (engine.backend()) engine.backend()->unregisterRenderBuffer();
    interopActive = false;
    std::string err;
    auto be = makeBackend(backendKind, blockSize, &err);
    if (!be) {
        backendStatus = "Requested backend unavailable (" + err + "); using CPU single-thread.";
        backendKind = BackendKind::Cpu1;
        be = createCpuBackend(1);
    } else {
        backendStatus = be->name();
    }
    try {
        engine.reset(cfg, std::move(be));
    } catch (const std::exception& ex) {
        backendStatus = std::string("Reset failed: ") + ex.what();
        engine.reset(cfg, createCpuBackend(1));
    }
    renderer_.buildScene(engine.layout());
    renderer_.reserveInstances((int)engine.records().size());
    if (useInterop && engine.backend()->supportsInterop()) {
        interopActive = engine.backend()->registerRenderBuffer(renderer_.patientInstanceBuffer());
        if (!interopActive) backendStatus += " | CUDA-GL interop unavailable (GL context not on the CUDA GPU?)";
    }
    simAccum_ = 0;
    running = false;
    renderSettings.selected = -1;
    staffPos.clear();
    dirtyRender_ = true;
    updateRenderData(true);
    metrics = engine.computeMetrics();
    metricsAge = 0;
}

void App::applyPreset(ScenarioType t) {
    cfg = makePreset(t);
    cfgB = cfg;
    cfgB.cap.doctors += 5;
    cfgB.cap.icuBeds += 10;
}

void App::advanceSimulation(double realDt) {
    stepsLastFrame = 0;
    if (!running) return;
    if (engine.finished()) {
        running = false;
        return;
    }
    const double dt = engine.config().dt;
    simAccum_ += realDt * simSpeed;
    const auto t0 = std::chrono::steady_clock::now();
    const int done = engine.advance(simAccum_, 14.0);  // keep the UI responsive (~14 ms budget)
    advanceMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    simAccum_ -= done * dt;
    lagging = simAccum_ > 2 * dt;
    if (lagging) simAccum_ = 0;  // drop the backlog instead of spiralling
    stepsLastFrame = done;
    stepsPerSecond = stepsPerSecond * 0.9 + (realDt > 0 ? done / realDt : 0) * 0.1;
    if (done) dirtyRender_ = true;
}

void App::updateRenderData(bool force) {
    if (!dirtyRender_ && !force) return;
    const auto t0 = std::chrono::steady_clock::now();
    PatientBackend* be = engine.backend();
    const int n = be->count();
    if (interopActive) {
        if (renderer_.reserveInstances(n))
            interopActive = be->registerRenderBuffer(renderer_.patientInstanceBuffer());
        if (interopActive && be->writeRenderBufferInterop()) {
            renderer_.setPatientCount(n);
            // follow camera needs CPU-side positions
            if (cam.mode == CameraMode::Follow || renderData.size() != (size_t)n) be->readRenderData(renderData);
        } else {
            interopActive = false;
        }
    }
    if (!interopActive) {
        be->readRenderData(renderData);
        renderer_.uploadPatients(renderData);
    }
    uploadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    dirtyRender_ = false;
}

void App::updateStaff(double dt) {
    const HospitalLayout& L = engine.layout();
    const ResourceManager& R = engine.resources();
    const auto& recs = engine.records();
    const ResourcePool& docs = R.pool(R_DOCTOR);
    const ResourcePool& nurses = R.pool(R_NURSE);
    std::vector<StaffInstance> staff;
    staff.reserve(docs.capacity() + nurses.capacity());
    if ((int)staffPos.size() != docs.capacity()) {
        staffPos.clear();
        for (const Vec2& p : L.doctorStations) staffPos.push_back({p.x, p.z});
    }
    const float a = 1.f - std::exp(-(float)dt * 4.f);
    for (int d = 0; d < docs.capacity(); ++d) {
        const int32_t occ = docs.occupant(d);
        glm::vec2 target(L.doctorStations[d].x, L.doctorStations[d].z);
        if (occ >= 0 && recs[occ].erBed >= 0) {
            const Vec2 b = L.erBeds[recs[occ].erBed];
            target = {b.x + 1.1f, b.z - 0.3f};
        }
        staffPos[d] = glm::mix(staffPos[d], target, a);
        staff.push_back({staffPos[d].x, staffPos[d].y, occ >= 0 ? 0u : 1u});
    }
    for (int k = 0; k < nurses.capacity(); ++k)
        staff.push_back({L.nursePositions[k].x, L.nursePositions[k].z, nurses.occupant(k) >= 0 ? 2u : 3u});
    renderer_.uploadStaff(staff);
}

// -------------------------------------------------------------------- input ----
void App::handleInput(double dt) {
    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard) {
        const bool fast = ImGui::IsKeyDown(ImGuiKey_LeftShift);
        glm::vec3 d(0.f);
        if (ImGui::IsKeyDown(ImGuiKey_W)) d.z += 1;
        if (ImGui::IsKeyDown(ImGuiKey_S)) d.z -= 1;
        if (ImGui::IsKeyDown(ImGuiKey_D)) d.x += 1;
        if (ImGui::IsKeyDown(ImGuiKey_A)) d.x -= 1;
        if (ImGui::IsKeyDown(ImGuiKey_E)) d.y += 1;
        if (ImGui::IsKeyDown(ImGuiKey_Q)) d.y -= 1;
        if (cam.mode == CameraMode::Free) cam.moveFree(d, (float)dt, fast);
        else if (d.x != 0 || d.z != 0) cam.panKeys({d.x, d.z}, (float)dt, fast);
        if (ImGui::IsKeyPressed(ImGuiKey_Space)) running = !running;
        if (ImGui::IsKeyPressed(ImGuiKey_1)) cam.setMode(CameraMode::TopDown);
        if (ImGui::IsKeyPressed(ImGuiKey_2)) cam.setMode(CameraMode::Orbit);
        if (ImGui::IsKeyPressed(ImGuiKey_3)) cam.setMode(CameraMode::Free);
        if (ImGui::IsKeyPressed(ImGuiKey_4) && renderSettings.selected >= 0) cam.setMode(CameraMode::Follow);
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            renderSettings.selected = -1;
            if (cam.mode == CameraMode::Follow) cam.setMode(CameraMode::Orbit);
        }
    }
    double mx, my;
    glfwGetCursorPos(window_, &mx, &my);
    int ww, wh;
    glfwGetWindowSize(window_, &ww, &wh);
    const double sx = ww > 0 ? (double)width / ww : 1.0, sy = wh > 0 ? (double)height / wh : 1.0;
    if (!io.WantCaptureMouse) {
        for (int b = 0; b < 3; ++b)
            if (ImGui::IsMouseClicked(b)) {
                pressButton_ = b;
                pressX_ = lastX_ = mx;
                pressY_ = lastY_ = my;
                dragging_ = false;
            }
        if (io.MouseWheel != 0.f) cam.zoom(io.MouseWheel);
    }
    if (pressButton_ >= 0) {
        const double dx = mx - lastX_, dy = my - lastY_;
        if (std::abs(mx - pressX_) + std::abs(my - pressY_) > 4) dragging_ = true;
        if (dragging_) {
            if (pressButton_ == 1) cam.rotate((float)dx, (float)dy);
            else cam.pan((float)dx, (float)dy);
        }
        lastX_ = mx;
        lastY_ = my;
        if (ImGui::IsMouseReleased(pressButton_)) {
            if (!dragging_ && pressButton_ == 0) pick(mx * sx, my * sy);
            pressButton_ = -1;
        }
    }
}

void App::pick(double mx, double my) {
    const Ray r = cam.screenRay((float)mx, (float)my, width, height);
    if (r.dir.y >= -1e-4f) return;
    const float t = (0.9f - r.origin.y) / r.dir.y;
    const glm::vec3 p = r.origin + r.dir * t;
    if (interopActive) engine.backend()->readRenderData(renderData);
    int best = -1;
    float bestD = 1.6f * 1.6f;
    for (int i = 0; i < (int)renderData.size(); ++i) {
        const uint32_t st = renderData[i].packed & 0xFF;
        if (st == ST_NOT_ARRIVED || st == ST_DISCHARGED) continue;
        const float dx = renderData[i].x - p.x, dz = renderData[i].z - p.z, d = dx * dx + dz * dz;
        if (d < bestD) {
            bestD = d;
            best = i;
        }
    }
    selectPatient(best);
}

void App::selectPatient(int id) {
    renderSettings.selected = id;
    if (id < 0 && cam.mode == CameraMode::Follow) cam.setMode(CameraMode::Orbit);
}

// ------------------------------------------------------------------- jobs ----
void App::startBenchmark(bool quick, bool includeScenario) {
    if (job.busy()) return;
    wasRunningBeforeJob_ = running;
    running = false;
    jobWasBusy_ = true;
    job.start("Benchmark", [this, quick, includeScenario](BackgroundJob& j) {
        BenchmarkOptions o = quick ? BenchmarkOptions::quick() : BenchmarkOptions();
        o.outDir = "benchmarks/results_gui";
        o.progress = [&j](const std::string& m, float f) { j.setProgress(m, f); };
        o.cancel = &j.cancel;
        auto k = Benchmark::runKernelSuite(o);
        Benchmark::writeKernelCsv(o.outDir + "/kernel_benchmark.csv", k);
        std::vector<ScenarioBenchRow> s;
        if (includeScenario && !j.cancel) {
            s = Benchmark::runScenarioSuite(o);
            Benchmark::writeScenarioCsv(o.outDir + "/scenario_benchmark.csv", s);
        }
        std::lock_guard<std::mutex> l(j.resultMutex);
        kernelRows = std::move(k);
        if (includeScenario) scenarioRows = std::move(s);
        benchMessage = (j.cancel ? "Cancelled. Partial results saved to " : "Saved CSV to ") + o.outDir;
        j.setProgress("done", 1.f);
    });
}

void App::startComparison() {
    if (job.busy()) return;
    wasRunningBeforeJob_ = running;
    running = false;
    jobWasBusy_ = true;
    const ScenarioConfig a = cfg, b = cfgB;
    const BackendKind k = gpu.available ? compareBackend : BackendKind::Cpu1;
    job.start("Configuration comparison", [this, a, b, k](BackgroundJob& j) {
        j.setProgress("running configuration A", 0.1f);
        ConfigRunResult ra = runScenarioHeadless(a, k);
        j.setProgress("running configuration B", 0.55f);
        ConfigRunResult rb = runScenarioHeadless(b, k);
        std::lock_guard<std::mutex> l(j.resultMutex);
        compareA = ra;
        compareB = rb;
        compareValid = ra.ok && rb.ok;
        j.setProgress("done", 1.f);
    });
}

void App::startLiveSpeedup() {
    if (job.busy() || !gpu.available) return;
    std::vector<PatientAoS> snap;
    engine.backend()->readHotState(snap);
    const UpdateParams p = engine.config().model;
    const RouteTable rt = engine.layout().routes();
    int active = 0;
    for (const auto& q : snap) active += inMask(kActiveMask, q.state);
    wasRunningBeforeJob_ = running;
    running = false;
    jobWasBusy_ = true;
    job.start("Live speedup", [this, snap = std::move(snap), p, rt, active](BackgroundJob& j) {
        const int steps = 200;
        LiveSpeedup ls;
        ls.patients = (int)snap.size();
        ls.active = active;
        ls.steps = steps;
        j.setProgress("CPU single-thread", 0.1f);
        const CpuRunResult c1 = runCpuReference(snap, p, rt, steps, 1);
        j.setProgress("CPU multi-thread", 0.4f);
        const CpuRunResult cm = runCpuReference(snap, p, rt, steps, std::max(2, (int)std::thread::hardware_concurrency()));
        j.setProgress("CUDA optimized", 0.7f);
        GpuRunConfig gc;
        gc.steps = steps;
        const GpuRunResult g = runGpuKernelBenchmark(snap, {}, p, rt, gc);
        ls.cpu1Ms = c1.ms;
        ls.cpuMtMs = cm.ms;
        if (g.ok) {
            ls.gpuKernelMs = g.kernelMs;
            ls.gpuE2eMs = g.endToEndMs;
            const StateDiff d = compareStates(c1.finalState, g.finalState);
            ls.identical = d.intMismatches == 0 && d.maxAbsDiff == 0 && g.eventHash == c1.eventHash;
            ls.valid = true;
        } else {
            ls.error = g.error;
        }
        std::lock_guard<std::mutex> l(j.resultMutex);
        liveSpeedup = ls;
        j.setProgress("done", 1.f);
    });
}

}  // namespace hs

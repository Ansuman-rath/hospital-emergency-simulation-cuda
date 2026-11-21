// App.h - interactive application: window, main loop, background jobs.
#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "Benchmark.h"
#include "Camera.h"
#include "CudaEngine.h"
#include "Renderer.h"
#include "SimulationEngine.h"

struct GLFWwindow;

namespace hs {

// One background job at a time (benchmark / comparison / live speedup); the live
// simulation is paused while it runs so measurements are not disturbed.
class BackgroundJob {
public:
    ~BackgroundJob() { join(); }
    bool busy() const { return running_.load(); }
    void start(const std::string& name, std::function<void(BackgroundJob&)> fn);
    void join() {
        if (th_.joinable()) th_.join();
    }
    void setProgress(const std::string& msg, float f) {
        std::lock_guard<std::mutex> l(m_);
        msg_ = msg;
        progress_ = f;
    }
    void getProgress(std::string& msg, float& f) {
        std::lock_guard<std::mutex> l(m_);
        msg = msg_;
        f = progress_;
    }
    std::string name() {
        std::lock_guard<std::mutex> l(m_);
        return name_;
    }
    std::atomic<bool> cancel{false};
    std::mutex resultMutex;

private:
    std::thread th_;
    std::atomic<bool> running_{false};
    std::mutex m_;
    std::string msg_, name_;
    float progress_ = 0;
};

struct LiveSpeedup {
    bool valid = false;
    int patients = 0, active = 0, steps = 0;
    double cpu1Ms = 0, cpuMtMs = 0, gpuKernelMs = 0, gpuE2eMs = 0;
    bool identical = false;
    std::string error;
};

class App {
public:
    int run();

    // ---- state shared with the dashboard -------------------------------------------
    ScenarioConfig cfg;  // configuration used at the next reset
    BackendKind backendKind = BackendKind::CudaOptimized;
    int blockSize = 256;
    bool running = false;
    float simSpeed = 10.f;  // sim-minutes per real second
    bool useInterop = false;
    bool interopActive = false;
    std::string backendStatus;
    GpuDeviceInfo gpu;

    SimulationEngine engine;
    LiveMetrics metrics;
    double metricsAge = 1e9;

    // performance (live)
    double frameMs = 0, advanceMs = 0, uploadMs = 0;
    int stepsLastFrame = 0;
    bool lagging = false;
    double stepsPerSecond = 0;

    Camera cam;
    RenderSettings renderSettings;
    bool showLabels = true;
    std::vector<RenderInstance> renderData;
    std::vector<glm::vec2> staffPos;

    // background work + results
    BackgroundJob job;
    std::vector<KernelBenchRow> kernelRows;
    std::vector<ScenarioBenchRow> scenarioRows;
    std::string benchMessage;
    ScenarioConfig cfgB;
    BackendKind compareBackend = BackendKind::CudaOptimized;
    ConfigRunResult compareA, compareB;
    bool compareValid = false;
    LiveSpeedup liveSpeedup;

    // actions
    void resetSimulation();
    void applyPreset(ScenarioType t);
    void startBenchmark(bool quick, bool includeScenario);
    void startComparison();
    void startLiveSpeedup();
    void selectPatient(int id);
    void stepOnce() {
        engine.step();
        dirtyRender_ = true;
    }
    bool gpuAvailable() const { return gpu.available; }

    int width = 1600, height = 900;

    // automated screenshot mode (documentation / smoke test of the renderer)
    std::string screenshotPath;
    int screenshotScenario = -1;
    double screenshotSimMinutes = 0;
    int screenshotCamera = -1;
    int screenshotTab = 0;
    int screenshotPatients = 0;

private:
    void saveScreenshot(const std::string& path);
    void advanceSimulation(double realDt);
    void updateRenderData(bool force);
    void updateStaff(double dt);
    void handleInput(double dt);
    void pick(double mx, double my);
    GLFWwindow* window_ = nullptr;
    Renderer renderer_;
    bool dirtyRender_ = true;
    double simAccum_ = 0;
    bool dragging_ = false, wasRunningBeforeJob_ = false, jobWasBusy_ = false;
    double pressX_ = 0, pressY_ = 0, lastX_ = 0, lastY_ = 0;
    int pressButton_ = -1;
};

}  // namespace hs

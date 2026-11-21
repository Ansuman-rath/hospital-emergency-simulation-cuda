// CudaEngine.cu - CUDA implementation of PatientBackend.
//
//  NaiveAoS     : host keeps the authoritative AoS copy. Every step copies the whole
//                 array H2D, runs 4 separate kernels (+clear), copies the whole array
//                 and a per-patient event-bit array back D2H, and scans on the CPU.
//  OptimizedSoA : patient data stays resident in GPU global memory (SoA) across
//                 steps. Per step: one fused kernel; the GPU compacts events into a
//                 short list; only the event count + events are copied back. CPU
//                 decisions are sent as a small command list (pinned memory) and
//                 scattered by a kernel.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <cuda_gl_interop.h>

#include <algorithm>
#include <chrono>

#include "CudaEngine.h"
#include "MemoryManager.cuh"
#include "PatientKernels.cuh"
#include "PatientUpdate.h"

namespace hs {

bool cudaBuildEnabled() { return true; }
void setCudaEventTiming(bool enabled) { gpu::g_eventTimingEnabled = enabled; }

const char* cudaVariantName(CudaVariant v) {
    switch (v) {
        case CudaVariant::NaiveAoS: return "CUDA naive (AoS, split kernels, full copies)";
        case CudaVariant::OptimizedSoA: return "CUDA optimized (SoA, fused, compacted events)";
        case CudaVariant::OptimizedGraph: return "CUDA optimized + CUDA Graph";
    }
    return "?";
}

GpuDeviceInfo queryGpu() {
    GpuDeviceInfo info;
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count == 0) {
        info.error = e != cudaSuccess ? cudaGetErrorString(e) : "no CUDA device found";
        cudaGetLastError();
        return info;
    }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
        info.error = "cudaGetDeviceProperties failed";
        return info;
    }
    info.available = true;
    info.name = p.name;
    info.ccMajor = p.major;
    info.ccMinor = p.minor;
    info.smCount = p.multiProcessorCount;
    info.totalMemBytes = p.totalGlobalMem;
    info.maxThreadsPerBlock = p.maxThreadsPerBlock;
    info.warpSize = p.warpSize;
    info.sharedMemPerBlock = p.sharedMemPerBlock;
    info.constMem = p.totalConstMem;
    int busWidth = 0, l2 = 0;
    cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, 0);
    cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0);
    info.memBusWidth = busWidth;
    info.l2CacheBytes = l2;
    cudaDriverGetVersion(&info.driverVersion);
    cudaRuntimeGetVersion(&info.runtimeVersion);
    return info;
}

namespace {

using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

class CudaBackend final : public PatientBackend {
public:
    CudaBackend(CudaVariant v, int bs) : variant_(v), bs_(std::max(32, (bs / 32) * 32)) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }
    ~CudaBackend() override {
        destroyGraph();
        unregisterRenderBuffer();
        if (stream_) cudaStreamDestroy(stream_);
    }
    std::string name() const override { return std::string(cudaVariantName(variant_)) + ", block " + std::to_string(bs_); }
    bool isGpu() const override { return true; }

    void init(const std::vector<PatientAoS>& init, const UpdateParams& params, const RouteTable& routes) override {
        destroyGraph();
        lastEvents_ = 0;
        n_ = (int)init.size();
        params_ = params;
        routes_ = routes;
        timings_ = {};
        gpu::setConstants(this, params_, routes_);
        events_.resize(std::max(1, n_));
        events_.zero();  // speculative prefix copies may read unused slots: keep them defined
        hEvents_.resize(std::max(1, n_));
        count_.resize(2);  // [0] event count, [1] stopAt (multi-step batching)
        hCount_.resize(2);
        batchHint_ = 8;
        partials_.resize(gpu::reduceBlockCount(n_, bs_));
        hPartials_.resize(gpu::reduceBlockCount(n_, bs_));
        if (variant_ == CudaVariant::NaiveAoS) {
            aos_.resize(std::max(1, n_));
            hAos_.resize(std::max(1, n_));
            bits_.resize(std::max(1, n_));
            hBits_.resize(std::max(1, n_));
            std::copy(init.begin(), init.end(), hAos_.get());
        } else {
            soa_.allocate(n_);
            soa_.upload(init.data(), n_, stream_);
            render_.resize(std::max(1, n_));
            hRender_.resize(std::max(1, n_));
            timings_.bytesH2D += soa_.bytes();
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void applyCommands(const std::vector<PatientCommand>& cmds) override {
        auto t0 = Clock::now();
        const int nc = (int)cmds.size();
        if (variant_ == CudaVariant::NaiveAoS) {
            // Naive: CPU edits its own copy; the whole array is uploaded next step.
            for (const PatientCommand& c : cmds) applyCommand(hAos_[c.id], c);
        } else if (nc > 0) {
            hCmds_.resize(nc);
            cmds_.resize(nc);
            std::copy(cmds.begin(), cmds.end(), hCmds_.get());
            cmdTimer_.start(stream_);
            CUDA_CHECK(cudaMemcpyAsync(cmds_.get(), hCmds_.get(), nc * sizeof(PatientCommand), cudaMemcpyHostToDevice,
                                       stream_));
            cmdTimer_.stop(stream_);
            gpu::launchApplyCommandsSoA(soa_.view(), n_, cmds_.get(), nc, bs_, stream_);
            pendingCmdTiming_ = true;
            timings_.bytesH2D += (uint64_t)nc * sizeof(PatientCommand);
        }
        timings_.totalCommandMs += msSince(t0);
    }

    void step(std::vector<SimEvent>& events) override {
        auto t0 = Clock::now();
        gpu::setConstants(this, params_, routes_);
        events.clear();
        double h2d = 0, d2h = 0, kern = 0;
        if (n_ > 0) {
            if (variant_ == CudaVariant::NaiveAoS) stepNaive(events, h2d, kern, d2h);
            else stepOptimized(events, h2d, kern, d2h);
        }
        const double wall = msSince(t0);
        timings_.stepWallMs = wall;
        timings_.computeMs = kern;
        timings_.h2dMs = h2d;
        timings_.d2hMs = d2h;
        timings_.totalStepWallMs += wall;
        timings_.totalComputeMs += kern;
        timings_.totalH2DMs += h2d;
        timings_.totalD2HMs += d2h;
        timings_.steps++;
        timings_.batches++;
    }

    // Event-driven multi-step batching (OptimizedSoA): enqueue K fused kernels back to back
    // with ONE host synchronisation. Kernel k exits immediately if an earlier kernel of the
    // batch produced events (device-side atomicMin on stopAt), so the CPU controller still
    // reacts at exactly the step where the event happened -> results stay identical.
    int stepMulti(int maxSteps, std::vector<SimEvent>& events) override {
        if (variant_ != CudaVariant::OptimizedSoA || n_ == 0 || maxSteps <= 1)
            return PatientBackend::stepMulti(maxSteps, events);
        auto t0 = Clock::now();
        gpu::setConstants(this, params_, routes_);
        const int K = std::max(1, std::min(maxSteps, batchHint_));
        tKernel_.start(stream_);
        CUDA_CHECK(cudaMemsetAsync(count_.get(), 0, sizeof(int), stream_));
        CUDA_CHECK(cudaMemsetAsync(count_.get() + 1, 0x7F, sizeof(int), stream_));  // stopAt = 0x7F7F7F7F
        for (int k = 0; k < K; ++k)
            gpu::launchFusedSoA(soa_.view(), n_, events_.get(), count_.get(), bs_, stream_, k, count_.get() + 1);
        tKernel_.stop(stream_);
        const int spec = specBucket();
        tD2H_.start(stream_);
        CUDA_CHECK(cudaMemcpyAsync(hCount_.get(), count_.get(), 2 * sizeof(int), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaMemcpyAsync(hEvents_.get(), events_.get(), spec * sizeof(SimEvent), cudaMemcpyDeviceToHost,
                                   stream_));
        tD2H_.stop(stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        const int nev = hCount_[0], stopAt = hCount_[1];
        if (nev > spec) {
            CUDA_CHECK(cudaMemcpyAsync(hEvents_.get() + spec, events_.get() + spec, (nev - spec) * sizeof(SimEvent),
                                       cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
        lastEvents_ = nev;
        const int executed = stopAt == 0x7F7F7F7F ? K : stopAt + 1;
        // adaptive batch length: grow while batches run to completion, shrink toward the observed gap
        batchHint_ = executed == K ? std::min(512, K * 2) : std::max(2, executed * 2);
        events.assign(hEvents_.get(), hEvents_.get() + nev);
        std::sort(events.begin(), events.end(), [](const SimEvent& a, const SimEvent& b) { return a.id < b.id; });
        const double kern = tKernel_.ms(), d2h = tD2H_.ms(), h2d = pendingCmdTiming_ ? cmdTimer_.ms() : 0.0;
        pendingCmdTiming_ = false;
        const double wall = msSince(t0);
        timings_.stepWallMs = wall / executed;
        timings_.computeMs = kern / executed;
        timings_.h2dMs = h2d;
        timings_.d2hMs = d2h;
        timings_.totalStepWallMs += wall;
        timings_.totalComputeMs += kern;
        timings_.totalH2DMs += h2d;
        timings_.totalD2HMs += d2h;
        timings_.steps += executed;
        timings_.batches++;
        timings_.bytesD2H += 2 * sizeof(int) + (uint64_t)std::max(nev, spec) * sizeof(SimEvent);
        return executed;
    }
    void readRenderData(std::vector<RenderInstance>& out) override {
        out.resize(n_);
        if (n_ == 0) return;
        if (variant_ == CudaVariant::NaiveAoS) {
            for (int i = 0; i < n_; ++i) {
                const PatientAoS& q = hAos_[i];
                out[i] = {q.x, q.z, packRender(q.state, q.priority, q.flags)};
            }
            return;
        }
        gpu::launchPackRenderSoA(soa_.view(), n_, render_.get(), bs_, stream_);
        CUDA_CHECK(cudaMemcpyAsync(hRender_.get(), render_.get(), n_ * sizeof(RenderInstance),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        std::copy(hRender_.get(), hRender_.get() + n_, out.begin());
    }

    void readHotState(std::vector<PatientAoS>& out) override {
        out.resize(n_);
        if (n_ == 0) return;
        if (variant_ == CudaVariant::NaiveAoS) std::copy(hAos_.get(), hAos_.get() + n_, out.begin());
        else soa_.download(out.data(), n_, stream_);
    }

    bool readPatient(int id, PatientAoS& out) override {
        if (id < 0 || id >= n_) return false;
        if (variant_ == CudaVariant::NaiveAoS) {
            out = hAos_[id];
            return true;
        }
        const PatientSoA& v = soa_.view();
        float f[8];
        int32_t iv[5];
        float* fp[8] = {v.x, v.z, v.tx, v.tz, v.speed, v.waitingTime, v.serviceRemaining, v.health};
        int32_t* ip[5] = {v.state, v.priority, v.route, v.waypoint, v.flags};
        for (int k = 0; k < 8; ++k)
            CUDA_CHECK(cudaMemcpyAsync(&f[k], fp[k] + id, 4, cudaMemcpyDeviceToHost, stream_));
        for (int k = 0; k < 5; ++k)
            CUDA_CHECK(cudaMemcpyAsync(&iv[k], ip[k] + id, 4, cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        out = {f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], iv[0], iv[1], iv[2], iv[3], iv[4]};
        return true;
    }
    HotReduction reduce() override {
        HotReduction r;
        if (n_ == 0) return r;
        const int nb = gpu::reduceBlockCount(n_, bs_);
        if (variant_ == CudaVariant::NaiveAoS) {
            // naive keeps the authoritative copy on the host (already downloaded)
            for (int i = 0; i < n_; ++i) {
                const PatientAoS& q = hAos_[i];
                if (q.state == ST_NOT_ARRIVED) continue;
                r.sumWait += q.waitingTime;
                r.maxWait = std::max(r.maxWait, q.waitingTime);
                if (inMask(kActiveMask, q.state)) {
                    r.sumHealth += q.health;
                    r.activeCount++;
                }
            }
            return r;
        }
        gpu::launchReduceSoA(soa_.view(), n_, partials_.get(), bs_, stream_);
        CUDA_CHECK(cudaMemcpyAsync(hPartials_.get(), partials_.get(), nb * sizeof(float4), cudaMemcpyDeviceToHost,
                                   stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        for (int b = 0; b < nb; ++b) {
            const float4 v = hPartials_[b];
            r.sumWait += v.x;
            r.maxWait = std::max(r.maxWait, v.y);
            r.sumHealth += v.z;
            r.activeCount += (int)v.w;
        }
        return r;
    }

    // ---- CUDA-OpenGL interop (optional): the pack kernel writes straight into the VBO.
    bool supportsInterop() const override { return variant_ != CudaVariant::NaiveAoS; }
    bool registerRenderBuffer(unsigned int glBuffer) override {
        unregisterRenderBuffer();
        if (!supportsInterop()) return false;
        cudaError_t e = cudaGraphicsGLRegisterBuffer(&glRes_, glBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
        if (e != cudaSuccess) {
            cudaGetLastError();
            glRes_ = nullptr;
            return false;
        }
        return true;
    }
    bool writeRenderBufferInterop() override {
        if (!glRes_ || n_ == 0) return false;
        if (cudaGraphicsMapResources(1, &glRes_, stream_) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        void* ptr = nullptr;
        size_t size = 0;
        bool ok = cudaGraphicsResourceGetMappedPointer(&ptr, &size, glRes_) == cudaSuccess &&
                  size >= (size_t)n_ * sizeof(RenderInstance);
        if (ok) gpu::launchPackRenderSoA(soa_.view(), n_, (RenderInstance*)ptr, bs_, stream_);
        cudaGraphicsUnmapResources(1, &glRes_, stream_);
        cudaGetLastError();
        return ok;
    }
    void unregisterRenderBuffer() override {
        if (glRes_) cudaGraphicsUnregisterResource(glRes_);
        glRes_ = nullptr;
    }

private:
    void stepNaive(std::vector<SimEvent>& events, double& h2d, double& kern, double& d2h) {
        const size_t bytes = (size_t)n_ * sizeof(PatientAoS);
        tH2D_.start(stream_);
        CUDA_CHECK(cudaMemcpyAsync(aos_.get(), hAos_.get(), bytes, cudaMemcpyHostToDevice, stream_));
        tH2D_.stop(stream_);
        tKernel_.start(stream_);
        gpu::launchSplitAoS(aos_.get(), bits_.get(), n_, bs_, stream_);
        tKernel_.stop(stream_);
        tD2H_.start(stream_);
        CUDA_CHECK(cudaMemcpyAsync(hAos_.get(), aos_.get(), bytes, cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaMemcpyAsync(hBits_.get(), bits_.get(), n_ * 4, cudaMemcpyDeviceToHost, stream_));
        tD2H_.stop(stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        for (int i = 0; i < n_; ++i)
            if (hBits_[i]) events.push_back({i, hBits_[i], hAos_[i].priority, hAos_[i].waitingTime});
        h2d = tH2D_.ms();
        kern = tKernel_.ms();
        d2h = tD2H_.ms();
        timings_.bytesH2D += bytes;
        timings_.bytesD2H += bytes + (uint64_t)n_ * 4;
    }

    // Speculative event-prefix size, rounded up to a power of two so that the CUDA graph
    // (which bakes in the copy size) rarely needs to be re-instantiated.
    int specBucket() const {
        int s = 1024;
        while (s < 2 * lastEvents_) s <<= 1;
        return std::min(n_, s);
    }

    // Records memset + fused kernel + both D2H copies as ONE CUDA graph; replaying it costs a
    // single launch instead of four API submissions per step.
    void ensureGraph(int spec) {
        if (graphExec_ && graphSpec_ == spec) return;
        destroyGraph();
        cudaGraph_t g = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        CUDA_CHECK(cudaMemsetAsync(count_.get(), 0, sizeof(int), stream_));
        gpu::launchFusedSoA(soa_.view(), n_, events_.get(), count_.get(), bs_, stream_);
        CUDA_CHECK(cudaMemcpyAsync(hCount_.get(), count_.get(), sizeof(int), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaMemcpyAsync(hEvents_.get(), events_.get(), spec * sizeof(SimEvent), cudaMemcpyDeviceToHost,
                                   stream_));
        CUDA_CHECK(cudaStreamEndCapture(stream_, &g));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec_, g, 0));
        cudaGraphDestroy(g);
        graphSpec_ = spec;
        graphBuilds_++;
    }
    void destroyGraph() {
        if (graphExec_) cudaGraphExecDestroy(graphExec_);
        graphExec_ = nullptr;
        graphSpec_ = -1;
    }

    void stepOptimized(std::vector<SimEvent>& events, double& h2d, double& kern, double& d2h) {
        // One synchronisation per step: copy the event count AND a speculative prefix of
        // the event buffer in the same async chain; only if more events than expected
        // were produced is a second copy needed.
        const int spec = specBucket();
        if (variant_ == CudaVariant::OptimizedGraph) {
            ensureGraph(spec);
            tKernel_.start(stream_);  // graph time = memset + kernel + event copies
            CUDA_CHECK(cudaGraphLaunch(graphExec_, stream_));
            tKernel_.stop(stream_);
            tD2H_.start(stream_);
            tD2H_.stop(stream_);
        } else {
            tKernel_.start(stream_);
            CUDA_CHECK(cudaMemsetAsync(count_.get(), 0, sizeof(int), stream_));
            gpu::launchFusedSoA(soa_.view(), n_, events_.get(), count_.get(), bs_, stream_);
            tKernel_.stop(stream_);
            tD2H_.start(stream_);
            CUDA_CHECK(cudaMemcpyAsync(hCount_.get(), count_.get(), sizeof(int), cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaMemcpyAsync(hEvents_.get(), events_.get(), spec * sizeof(SimEvent),
                                       cudaMemcpyDeviceToHost, stream_));
            tD2H_.stop(stream_);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        const int nev = hCount_[0];
        if (nev > spec) {
            CUDA_CHECK(cudaMemcpyAsync(hEvents_.get() + spec, events_.get() + spec, (nev - spec) * sizeof(SimEvent),
                                       cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
        lastEvents_ = nev;
        if (pendingCmdTiming_) {
            h2d = cmdTimer_.ms();
            pendingCmdTiming_ = false;
        }
        events.assign(hEvents_.get(), hEvents_.get() + nev);
        // Atomic append order is non-deterministic -> sort by id for a deterministic CPU controller.
        std::sort(events.begin(), events.end(), [](const SimEvent& a, const SimEvent& b) { return a.id < b.id; });
        kern = tKernel_.ms();
        d2h = tD2H_.ms();
        timings_.bytesD2H += sizeof(int) + (uint64_t)std::max(nev, spec) * sizeof(SimEvent);
    }

    CudaVariant variant_;
    int bs_;
    cudaStream_t stream_ = nullptr;
    UpdateParams params_;
    RouteTable routes_{};
    // optimized
    gpu::DeviceSoA soa_;
    gpu::DeviceBuffer<PatientCommand> cmds_;
    gpu::PinnedBuffer<PatientCommand> hCmds_;
    gpu::DeviceBuffer<RenderInstance> render_;
    gpu::PinnedBuffer<RenderInstance> hRender_;
    // naive
    gpu::DeviceBuffer<PatientAoS> aos_;
    gpu::PinnedBuffer<PatientAoS> hAos_;
    gpu::DeviceBuffer<uint32_t> bits_;
    gpu::PinnedBuffer<uint32_t> hBits_;
    // shared
    gpu::DeviceBuffer<SimEvent> events_;
    gpu::PinnedBuffer<SimEvent> hEvents_;
    gpu::DeviceBuffer<int> count_;
    gpu::PinnedBuffer<int> hCount_;
    gpu::DeviceBuffer<float4> partials_;
    gpu::PinnedBuffer<float4> hPartials_;
    gpu::EventTimer tH2D_, tKernel_, tD2H_, cmdTimer_;
    bool pendingCmdTiming_ = false;
    int lastEvents_ = 0;
    int batchHint_ = 8;
    cudaGraphExec_t graphExec_ = nullptr;
    int graphSpec_ = -1;
    int graphBuilds_ = 0;
    cudaGraphicsResource* glRes_ = nullptr;
};

}  // namespace

std::unique_ptr<PatientBackend> createCudaBackend(CudaVariant v, int blockSize, std::string* error) {
    try {
        GpuDeviceInfo info = queryGpu();
        if (!info.available) {
            if (error) *error = "CUDA unavailable: " + info.error;
            return nullptr;
        }
        return std::make_unique<CudaBackend>(v, blockSize);
    } catch (const std::exception& ex) {
        if (error) *error = ex.what();
        return nullptr;
    }
}

}  // namespace hs

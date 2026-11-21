// CudaBenchmark.cu - kernel-level GPU experiments (AoS vs SoA, split vs fused,
// block size, active-list processing). Two measurements per configuration:
//   kernelMs   : CUDA events around `steps` back-to-back launches, data resident on
//                the GPU, no host interaction (pure kernel throughput incl. launch gaps)
//   endToEndMs : wall clock for the realistic hybrid loop: initial upload, every step
//                kernel + the transfer the CPU controller needs, final download.
#include <algorithm>
#include <chrono>

#include "CudaEngine.h"
#include "MemoryManager.cuh"
#include "PatientKernels.cuh"

namespace hs {

const char* gpuKernelVariantName(GpuKernelVariant v) {
    switch (v) {
        case GpuKernelVariant::NaiveAoS: return "CUDA naive (AoS split)";
        case GpuKernelVariant::AoSFused: return "CUDA AoS fused";
        case GpuKernelVariant::SoASplit: return "CUDA SoA split";
        case GpuKernelVariant::SoAFused: return "CUDA optimized (SoA fused)";
        case GpuKernelVariant::SoAFusedActive: return "CUDA SoA fused active-list";
        default: return "?";
    }
}

GpuRunResult runGpuKernelBenchmark(const std::vector<PatientAoS>& init, const std::vector<int32_t>& activeList,
                                   const UpdateParams& params, const RouteTable& routes, const GpuRunConfig& cfg) {
    using namespace gpu;
    GpuRunResult res;
    cudaStream_t s = nullptr;
    try {
        const int n = (int)init.size();
        const int bs = std::max(32, (cfg.blockSize / 32) * 32);
        const GpuKernelVariant v = cfg.variant;
        const bool aos = v == GpuKernelVariant::NaiveAoS || v == GpuKernelVariant::AoSFused;
        const int nActive = (int)activeList.size();
        setConstants(nullptr, params, routes);  // owner=nullptr: live engines re-upload afterwards
        CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

        DeviceBuffer<PatientAoS> dAos;
        PinnedBuffer<PatientAoS> hAos;
        DeviceSoA soa;
        DeviceBuffer<uint32_t> bits(std::max(1, n));
        PinnedBuffer<uint32_t> hBits;
        DeviceBuffer<SimEvent> ev(std::max(1, n));
        PinnedBuffer<SimEvent> hEv;
        DeviceBuffer<int> cnt(1);
        PinnedBuffer<int> hCnt;
        DeviceBuffer<int32_t> idx(std::max(1, nActive));
        ev.zero();  // speculative prefix copies may read unused slots: keep them defined
        hEv.resize(std::max(1, n));
        hCnt.resize(1);
        if (aos) {
            dAos.resize(std::max(1, n));
            hAos.resize(std::max(1, n));
            hBits.resize(std::max(1, n));
        } else {
            soa.allocate(n);
        }
        if (v == GpuKernelVariant::SoAFusedActive && nActive > 0)
            CUDA_CHECK(cudaMemcpy(idx.get(), activeList.data(), nActive * 4, cudaMemcpyHostToDevice));

        auto upload = [&]() {
            if (aos) {
                std::copy(init.begin(), init.end(), hAos.get());
                CUDA_CHECK(cudaMemcpyAsync(dAos.get(), hAos.get(), (size_t)n * sizeof(PatientAoS),
                                           cudaMemcpyHostToDevice, s));
            } else {
                soa.upload(init.data(), n, s);
            }
        };
        auto launch = [&]() {
            if (v != GpuKernelVariant::NaiveAoS) CUDA_CHECK(cudaMemsetAsync(cnt.get(), 0, sizeof(int), s));
            switch (v) {
                case GpuKernelVariant::NaiveAoS: launchSplitAoS(dAos.get(), bits.get(), n, bs, s); break;
                case GpuKernelVariant::AoSFused: launchFusedAoS(dAos.get(), n, ev.get(), cnt.get(), bs, s); break;
                case GpuKernelVariant::SoASplit:
                    launchSplitSoA(soa.view(), bits.get(), n, ev.get(), cnt.get(), bs, s);
                    break;
                case GpuKernelVariant::SoAFused: launchFusedSoA(soa.view(), n, ev.get(), cnt.get(), bs, s); break;
                case GpuKernelVariant::SoAFusedActive:
                    launchFusedSoAActive(soa.view(), idx.get(), nActive, ev.get(), cnt.get(), bs, s);
                    break;
                default: break;
            }
        };

        // Warm-up (module load, caches, clocks).
        upload();
        for (int i = 0; i < 3; ++i) launch();
        CUDA_CHECK(cudaStreamSynchronize(s));

        // ---- kernel-only run
        upload();
        CUDA_CHECK(cudaStreamSynchronize(s));
        EventTimer kt;
        kt.start(s);
        for (int st = 0; st < cfg.steps; ++st) launch();
        kt.stop(s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        res.kernelMs = kt.ms();

        // ---- end-to-end run
        EventTimer th, td;
        double h2d = 0, d2h = 0;
        uint64_t hash = 1469598103934665603ull, events = 0;
        int lastEv = 0;
        std::vector<SimEvent> stepEv;
        stepEv.reserve(1024);
        const size_t aosBytes = (size_t)n * sizeof(PatientAoS);
        auto t0 = std::chrono::steady_clock::now();
        th.start(s);
        upload();
        th.stop(s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        h2d += th.ms();
        for (int st = 0; st < cfg.steps; ++st) {
            stepEv.clear();
            if (v == GpuKernelVariant::NaiveAoS) {
                if (st > 0) {  // naive: host copy is authoritative, re-upload everything
                    th.start(s);
                    CUDA_CHECK(cudaMemcpyAsync(dAos.get(), hAos.get(), aosBytes, cudaMemcpyHostToDevice, s));
                    th.stop(s);
                }
                launch();
                td.start(s);
                CUDA_CHECK(cudaMemcpyAsync(hAos.get(), dAos.get(), aosBytes, cudaMemcpyDeviceToHost, s));
                CUDA_CHECK(cudaMemcpyAsync(hBits.get(), bits.get(), (size_t)n * 4, cudaMemcpyDeviceToHost, s));
                td.stop(s);
                CUDA_CHECK(cudaStreamSynchronize(s));
                if (st > 0) h2d += th.ms();
                d2h += td.ms();
                for (int i = 0; i < n; ++i)
                    if (hBits[i]) stepEv.push_back({i, hBits[i], hAos[i].priority, hAos[i].waitingTime});
            } else {
                launch();
                // same single-sync speculative event download as the live CUDA backend
                const int spec = std::min(std::max(1, n), std::max(1024, 2 * lastEv));
                td.start(s);
                CUDA_CHECK(cudaMemcpyAsync(hCnt.get(), cnt.get(), sizeof(int), cudaMemcpyDeviceToHost, s));
                CUDA_CHECK(cudaMemcpyAsync(hEv.get(), ev.get(), spec * sizeof(SimEvent), cudaMemcpyDeviceToHost, s));
                td.stop(s);
                CUDA_CHECK(cudaStreamSynchronize(s));
                d2h += td.ms();
                const int ne = hCnt[0];
                if (ne > spec) {
                    td.start(s);
                    CUDA_CHECK(cudaMemcpyAsync(hEv.get() + spec, ev.get() + spec, (ne - spec) * sizeof(SimEvent),
                                               cudaMemcpyDeviceToHost, s));
                    td.stop(s);
                    CUDA_CHECK(cudaStreamSynchronize(s));
                    d2h += td.ms();
                }
                lastEv = ne;
                stepEv.assign(hEv.get(), hEv.get() + ne);
                std::sort(stepEv.begin(), stepEv.end(),
                          [](const SimEvent& a, const SimEvent& b) { return a.id < b.id; });
            }
            events += stepEv.size();
            hash = hashEvents(hash, (uint64_t)st, stepEv.data(), stepEv.size());
        }
        res.finalState.resize(n);
        td.start(s);
        if (aos) {
            CUDA_CHECK(cudaMemcpyAsync(hAos.get(), dAos.get(), aosBytes, cudaMemcpyDeviceToHost, s));
            td.stop(s);
            CUDA_CHECK(cudaStreamSynchronize(s));
            std::copy(hAos.get(), hAos.get() + n, res.finalState.begin());
        } else {
            soa.download(res.finalState.data(), n, s);
            td.stop(s);
            CUDA_CHECK(cudaStreamSynchronize(s));
        }
        d2h += td.ms();
        res.endToEndMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        res.h2dMs = h2d;
        res.d2hMs = d2h;
        res.events = events;
        res.eventHash = hash;
        res.ok = true;
    } catch (const std::exception& ex) {
        res.ok = false;
        res.error = ex.what();
    }
    if (s) cudaStreamDestroy(s);
    return res;
}

}  // namespace hs

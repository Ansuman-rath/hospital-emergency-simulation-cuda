// CpuBackend.cpp - CPU baseline for the per-patient update.
// Uses SoA std::vectors and exactly the same HS_HD update functions as the GPU.
#include <algorithm>
#include <chrono>
#include <cmath>

#include "PatientBackend.h"
#include "PatientUpdate.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hs {
namespace {

using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

class CpuBackend final : public PatientBackend {
public:
    explicit CpuBackend(int threads) : threads_(std::max(1, threads)) {
#ifndef _OPENMP
        threads_ = 1;
#endif
    }
    std::string name() const override {
        return threads_ > 1 ? "CPU (" + std::to_string(threads_) + " threads)" : "CPU (1 thread)";
    }

    void init(const std::vector<PatientAoS>& init, const UpdateParams& params, const RouteTable& routes) override {
        n_ = (int)init.size();
        params_ = params;
        routes_ = routes;
        auto resizeF = [&](std::vector<float>& v) { v.resize(n_); };
        auto resizeI = [&](std::vector<int32_t>& v) { v.resize(n_); };
        resizeF(x_); resizeF(z_); resizeF(tx_); resizeF(tz_); resizeF(speed_); resizeF(wait_);
        resizeF(svc_); resizeF(health_);
        resizeI(state_); resizeI(prio_); resizeI(route_); resizeI(wp_); resizeI(flags_);
        for (int i = 0; i < n_; ++i) store(i, init[i]);
        timings_ = {};
        threadEvents_.assign(threads_, {});
    }

    void applyCommands(const std::vector<PatientCommand>& cmds) override {
        auto t0 = Clock::now();
        for (const PatientCommand& c : cmds) {
            PatientAoS q = load(c.id);
            applyCommand(q, c);
            store(c.id, q);
        }
        timings_.totalCommandMs += msSince(t0);
    }

    void step(std::vector<SimEvent>& events) override {
        auto t0 = Clock::now();
        events.clear();
        if (threads_ == 1) {
            for (int i = 0; i < n_; ++i) updateOne(i, events);
        } else {
#ifdef _OPENMP
            // Static schedule: thread t gets a contiguous id range, so concatenating
            // per-thread buffers in thread order yields events sorted by id.
#pragma omp parallel num_threads(threads_)
            {
                const int t = omp_get_thread_num(), nt = omp_get_num_threads();
                std::vector<SimEvent>& ev = threadEvents_[t];
                ev.clear();
                const int chunk = (n_ + nt - 1) / nt;
                const int b = t * chunk, e = std::min(n_, b + chunk);
                for (int i = b; i < e; ++i) updateOne(i, ev);
            }
            for (auto& ev : threadEvents_) events.insert(events.end(), ev.begin(), ev.end());
#endif
        }
        const double ms = msSince(t0);
        timings_.stepWallMs = timings_.computeMs = ms;
        timings_.totalStepWallMs += ms;
        timings_.totalComputeMs += ms;
        timings_.steps++;
        timings_.batches++;
    }

    void readRenderData(std::vector<RenderInstance>& out) override {
        out.resize(n_);
        for (int i = 0; i < n_; ++i) out[i] = {x_[i], z_[i], packRender(state_[i], prio_[i], flags_[i])};
    }
    void readHotState(std::vector<PatientAoS>& out) override {
        out.resize(n_);
        for (int i = 0; i < n_; ++i) out[i] = load(i);
    }
    bool readPatient(int id, PatientAoS& out) override {
        if (id < 0 || id >= n_) return false;
        out = load(id);
        return true;
    }
    HotReduction reduce() override {
        HotReduction r;
        for (int i = 0; i < n_; ++i) {
            if (state_[i] == ST_NOT_ARRIVED) continue;
            r.sumWait += wait_[i];
            r.maxWait = std::max(r.maxWait, wait_[i]);
            if (inMask(kActiveMask, state_[i])) {
                r.sumHealth += health_[i];
                r.activeCount++;
            }
        }
        return r;
    }

private:
    inline void updateOne(int i, std::vector<SimEvent>& ev) {
        int32_t prio = prio_[i], wp = wp_[i], flags = flags_[i];
        const uint32_t bits = updatePatient(x_[i], z_[i], tx_[i], tz_[i], speed_[i], wait_[i], svc_[i], health_[i],
                                            state_[i], prio, route_[i], wp, flags, params_, routes_);
        prio_[i] = prio;
        wp_[i] = wp;
        flags_[i] = flags;
        if (bits) ev.push_back({i, bits, prio, wait_[i]});
    }
    PatientAoS load(int i) const {
        return {x_[i], z_[i], tx_[i], tz_[i], speed_[i], wait_[i], svc_[i], health_[i],
                state_[i], prio_[i], route_[i], wp_[i], flags_[i]};
    }
    void store(int i, const PatientAoS& q) {
        x_[i] = q.x; z_[i] = q.z; tx_[i] = q.tx; tz_[i] = q.tz; speed_[i] = q.speed;
        wait_[i] = q.waitingTime; svc_[i] = q.serviceRemaining; health_[i] = q.health;
        state_[i] = q.state; prio_[i] = q.priority; route_[i] = q.route; wp_[i] = q.waypoint; flags_[i] = q.flags;
    }

    int threads_;
    UpdateParams params_;
    RouteTable routes_{};
    std::vector<float> x_, z_, tx_, tz_, speed_, wait_, svc_, health_;
    std::vector<int32_t> state_, prio_, route_, wp_, flags_;
    std::vector<std::vector<SimEvent>> threadEvents_;
};

}  // namespace

std::unique_ptr<PatientBackend> createCpuBackend(int threads) { return std::make_unique<CpuBackend>(threads); }

}  // namespace hs

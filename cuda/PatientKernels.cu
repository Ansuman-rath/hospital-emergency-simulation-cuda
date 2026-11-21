// PatientKernels.cu - CUDA kernels for the per-patient simulation update.
//
// Thread mapping: one thread per patient,
//     int id = blockIdx.x * blockDim.x + threadIdx.x;   if (id < n) { ... }
// All per-patient math is in PatientUpdate.h and shared with the CPU baseline.
#include <mutex>

#include "MemoryManager.cuh"
#include "PatientKernels.cuh"
#include "PatientUpdate.h"

namespace hs {
namespace gpu {

__constant__ UpdateParams c_params;
__constant__ RouteTable c_routes;

static std::mutex g_constMutex;
static const void* g_constOwner = nullptr;

void setConstants(const void* owner, const UpdateParams& params, const RouteTable& routes) {
    std::lock_guard<std::mutex> lock(g_constMutex);
    if (g_constOwner == owner && owner != nullptr) return;
    CUDA_CHECK(cudaMemcpyToSymbol(c_params, &params, sizeof(UpdateParams)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_routes, &routes, sizeof(RouteTable)));
    g_constOwner = owner;
}

// ------------------------------------------------------------------------------------
// Warp-aggregated stream compaction: instead of one atomicAdd per event, the warp
// votes (__ballot_sync), one leader lane reserves space for all events of the warp,
// and every lane writes to base + (number of lower lanes with an event).
// All 32 lanes must reach this function (no early return in the kernels).
// stopAt (optional, multi-step batching): the leader also records the earliest step of
// the batch that produced an event, so that later kernels of the batch can skip.
__device__ __forceinline__ void appendEventWarp(SimEvent* out, int* count, bool has, const SimEvent& e,
                                                int* stopAt = nullptr, int step = 0) {
    const unsigned full = 0xffffffffu;
    const unsigned mask = __ballot_sync(full, has);
    if (mask == 0u) return;
    const int lane = threadIdx.x & 31;
    const int leader = __ffs(mask) - 1;
    int base = 0;
    if (lane == leader) {
        base = atomicAdd(count, __popc(mask));
        if (stopAt) atomicMin(stopAt, step);
    }
    base = __shfl_sync(full, base, leader);
    if (has) out[base + __popc(mask & ((1u << lane) - 1u))] = e;
}

// ----------------------------------------------------------- naive split (AoS) -----
// Each kernel re-reads and re-writes the 52-byte structure: thread i touches
// bytes [52*i, 52*i+52), i.e. strided, poorly coalesced access.
__global__ void k_clear_bits(uint32_t* bits, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) bits[id] = 0u;
}

__global__ void k_move_aos(PatientAoS* p, uint32_t* bits, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        PatientAoS q = p[id];
        bits[id] |= moveStep(q.x, q.z, q.tx, q.tz, q.speed, q.route, q.waypoint, q.flags, c_params.dt, c_routes);
        p[id] = q;
    }
}
__global__ void k_wait_aos(PatientAoS* p, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        PatientAoS q = p[id];
        waitStep(q.state, q.waitingTime, c_params.dt);
        p[id] = q;
    }
}
__global__ void k_service_aos(PatientAoS* p, uint32_t* bits, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        PatientAoS q = p[id];
        bits[id] |= serviceStep(q.state, q.flags, q.serviceRemaining, c_params.dt);
        p[id] = q;
    }
}
__global__ void k_health_aos(PatientAoS* p, uint32_t* bits, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        PatientAoS q = p[id];
        bits[id] |= healthStep(q.state, q.flags, q.priority, q.health, c_params);
        p[id] = q;
    }
}

void launchSplitAoS(PatientAoS* p, uint32_t* bits, int n, int bs, cudaStream_t s) {
    if (n <= 0) return;
    const int g = gridFor(n, bs);
    k_clear_bits<<<g, bs, 0, s>>>(bits, n);
    k_move_aos<<<g, bs, 0, s>>>(p, bits, n);
    k_wait_aos<<<g, bs, 0, s>>>(p, n);
    k_service_aos<<<g, bs, 0, s>>>(p, bits, n);
    k_health_aos<<<g, bs, 0, s>>>(p, bits, n);
    CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------- fused (AoS) -----
__global__ void k_fused_aos(PatientAoS* p, int n, SimEvent* ev, int* evCount) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bits = 0u;
    SimEvent e{};
    if (id < n) {
        PatientAoS q = p[id];
        bits = updatePatientAoS(q, c_params, c_routes);
        if (inMask(kActiveMask, q.state)) p[id] = q;
        e = {id, bits, q.priority, q.waitingTime};
    }
    appendEventWarp(ev, evCount, bits != 0u, e);
}

void launchFusedAoS(PatientAoS* p, int n, SimEvent* ev, int* evCount, int bs, cudaStream_t s) {
    if (n <= 0) return;
    k_fused_aos<<<gridFor(n, bs), bs, 0, s>>>(p, n, ev, evCount);
    CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------- split (SoA) -----
__global__ void k_move_soa(PatientSoA p, uint32_t* bits, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        int32_t flags = p.flags[id];
        uint32_t b = 0u;
        if (flags & FLAG_MOVING) {
            float x = p.x[id], z = p.z[id];
            int32_t wp = p.waypoint[id];
            b = moveStep(x, z, p.tx[id], p.tz[id], p.speed[id], p.route[id], wp, flags, c_params.dt, c_routes);
            p.x[id] = x;
            p.z[id] = z;
            p.waypoint[id] = wp;
            p.flags[id] = flags;
        }
        bits[id] = b;
    }
}
__global__ void k_wait_soa(PatientSoA p, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        const int32_t st = p.state[id];
        if (inMask(kWaitMask, st)) {
            float w = p.waitingTime[id];
            waitStep(st, w, c_params.dt);
            p.waitingTime[id] = w;
        }
    }
}
__global__ void k_service_soa(PatientSoA p, uint32_t* bits, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        const int32_t st = p.state[id];
        if (inMask(kServiceMask, st)) {
            float r = p.serviceRemaining[id];
            bits[id] |= serviceStep(st, p.flags[id], r, c_params.dt);
            p.serviceRemaining[id] = r;
        }
    }
}
__global__ void k_health_soa(PatientSoA p, uint32_t* bits, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        const int32_t st = p.state[id];
        if (inMask(kDeteriorateMask | kRecoverMask, st)) {
            float h = p.health[id];
            int32_t pr = p.priority[id];
            bits[id] |= healthStep(st, p.flags[id], pr, h, c_params);
            p.health[id] = h;
            p.priority[id] = pr;
        }
    }
}
__global__ void k_compact_soa(PatientSoA p, const uint32_t* bits, int n, SimEvent* ev, int* evCount) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = 0u;
    SimEvent e{};
    if (id < n) {
        b = bits[id];
        if (b) e = {id, b, p.priority[id], p.waitingTime[id]};
    }
    appendEventWarp(ev, evCount, b != 0u, e);
}

void launchSplitSoA(const PatientSoA& p, uint32_t* bits, int n, SimEvent* ev, int* evCount, int bs,
                    cudaStream_t s) {
    if (n <= 0) return;
    const int g = gridFor(n, bs);
    k_move_soa<<<g, bs, 0, s>>>(p, bits, n);
    k_wait_soa<<<g, bs, 0, s>>>(p, n);
    k_service_soa<<<g, bs, 0, s>>>(p, bits, n);
    k_health_soa<<<g, bs, 0, s>>>(p, bits, n);
    k_compact_soa<<<g, bs, 0, s>>>(p, bits, n, ev, evCount);
    CUDA_CHECK_LAUNCH();
}

// --------------------------------------------------------------- fused (SoA) -----
// Optimized kernel: neighbouring threads read neighbouring 4-byte words of each
// field array -> each warp load is one fully used 128-byte transaction.
// SELECTIVE loads/stores: a field is only read (and written back) if the patient's
// state needs it - e.g. positions/route only when moving, the service timer only in a
// service state. Inactive patients cost one 4-byte load. This matters once the arrays
// no longer fit in L2 and the kernel becomes DRAM-bandwidth bound (measured at N = 1M).
// The guards mirror the conditions inside moveStep/waitStep/serviceStep/healthStep, so
// the result is identical to updatePatient() (verified bit-exact against the CPU).
__device__ __forceinline__ uint32_t fusedSoAOne(const PatientSoA& p, int id, SimEvent& e) {
    const int32_t state = p.state[id];
    if (!inMask(kActiveMask, state)) return 0u;
    const int32_t flags0 = p.flags[id];
    int32_t flags = flags0;
    uint32_t bits = 0u;
    if (flags & FLAG_MOVING) {
        float x = p.x[id], z = p.z[id];
        int32_t wp = p.waypoint[id];
        bits |= moveStep(x, z, p.tx[id], p.tz[id], p.speed[id], p.route[id], wp, flags, c_params.dt, c_routes);
        p.x[id] = x;
        p.z[id] = z;
        p.waypoint[id] = wp;
        if (flags != flags0) p.flags[id] = flags;
    }
    float w = 0.f;
    bool haveW = false;
    if (inMask(kWaitMask, state)) {
        w = p.waitingTime[id];
        waitStep(state, w, c_params.dt);
        p.waitingTime[id] = w;
        haveW = true;
    }
    if (inMask(kServiceMask, state) && (flags & FLAG_MOVING) == 0) {
        float r = p.serviceRemaining[id];
        if (r > 0.0f) {
            bits |= serviceStep(state, flags, r, c_params.dt);
            p.serviceRemaining[id] = r;
        }
    }
    int32_t pr = 0;
    bool havePr = false;
    if (inMask(kDeteriorateMask, state) || (inMask(kRecoverMask, state) && (flags & FLAG_MOVING) == 0)) {
        float h = p.health[id];
        pr = p.priority[id];
        havePr = true;
        const int32_t pr0 = pr;
        bits |= healthStep(state, flags, pr, h, c_params);
        p.health[id] = h;
        if (pr != pr0) p.priority[id] = pr;
    }
    if (bits) {  // rare: fetch the remaining event fields only for patients that emit an event
        if (!havePr) pr = p.priority[id];
        if (!haveW) w = p.waitingTime[id];
        e = {id, bits, pr, w};
    }
    return bits;
}
__global__ void k_fused_soa(PatientSoA p, int n, SimEvent* ev, int* evCount, int step, int* stopAt) {
    // Multi-step batching: if an EARLIER step of this batch produced events, the CPU must
    // react before the simulation may continue -> the whole grid exits (uniform branch).
    if (stopAt && *stopAt < step) return;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bits = 0u;
    SimEvent e{};
    if (id < n) bits = fusedSoAOne(p, id, e);
    appendEventWarp(ev, evCount, bits != 0u, e, stopAt, step);
}

__global__ void k_fused_soa_active(PatientSoA p, const int32_t* idx, int count, SimEvent* ev, int* evCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bits = 0u;
    SimEvent e{};
    if (i < count) bits = fusedSoAOne(p, idx[i], e);
    appendEventWarp(ev, evCount, bits != 0u, e);
}

void launchFusedSoA(const PatientSoA& p, int n, SimEvent* ev, int* evCount, int bs, cudaStream_t s, int step,
                    int* stopAt) {
    if (n <= 0) return;
    k_fused_soa<<<gridFor(n, bs), bs, 0, s>>>(p, n, ev, evCount, step, stopAt);
    CUDA_CHECK_LAUNCH();
}

void launchFusedSoAActive(const PatientSoA& p, const int32_t* idx, int count, SimEvent* ev, int* evCount, int bs,
                          cudaStream_t s) {
    if (count <= 0) return;
    k_fused_soa_active<<<gridFor(count, bs), bs, 0, s>>>(p, idx, count, ev, evCount);
    CUDA_CHECK_LAUNCH();
}

// ------------------------------------------------------------- commands (scatter) ----
__global__ void k_apply_cmds_soa(PatientSoA p, int n, const PatientCommand* cmds, int nc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nc) {
        const PatientCommand c = cmds[i];
        const int id = c.id;
        if (id < 0 || id >= n) return;  // bounds check on a CPU-provided index
        p.state[id] = c.state;
        p.priority[id] = c.priority;
        p.flags[id] = c.flags;
        p.serviceRemaining[id] = c.serviceRemaining;
        if (c.mask & CMD_ROUTE) {
            p.route[id] = c.route;
            p.waypoint[id] = 0;
            p.tx[id] = c.tx;
            p.tz[id] = c.tz;
        }
        if (c.mask & CMD_INIT) {
            p.x[id] = c.x;
            p.z[id] = c.z;
            p.health[id] = c.health;
            p.speed[id] = c.speed;
            p.waitingTime[id] = 0.0f;
        }
    }
}
__global__ void k_apply_cmds_aos(PatientAoS* p, int n, const PatientCommand* cmds, int nc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nc) {
        const PatientCommand c = cmds[i];
        if (c.id < 0 || c.id >= n) return;
        PatientAoS q = p[c.id];
        applyCommand(q, c);
        p[c.id] = q;
    }
}
void launchApplyCommandsSoA(const PatientSoA& p, int n, const PatientCommand* c, int nc, int bs, cudaStream_t s) {
    if (nc <= 0) return;
    k_apply_cmds_soa<<<gridFor(nc, bs), bs, 0, s>>>(p, n, c, nc);
    CUDA_CHECK_LAUNCH();
}
void launchApplyCommandsAoS(PatientAoS* p, int n, const PatientCommand* c, int nc, int bs, cudaStream_t s) {
    if (nc <= 0) return;
    k_apply_cmds_aos<<<gridFor(nc, bs), bs, 0, s>>>(p, n, c, nc);
    CUDA_CHECK_LAUNCH();
}

// ------------------------------------------------------------------ render pack ----
__global__ void k_pack_render_soa(PatientSoA p, int n, RenderInstance* out) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) out[id] = {p.x[id], p.z[id], packRender(p.state[id], p.priority[id], p.flags[id])};
}
void launchPackRenderSoA(const PatientSoA& p, int n, RenderInstance* out, int bs, cudaStream_t s) {
    if (n <= 0) return;
    k_pack_render_soa<<<gridFor(n, bs), bs, 0, s>>>(p, n, out);
    CUDA_CHECK_LAUNCH();
}

// -------------------------------------------------------------------- reduction ----
// Stage 1 (GPU): grid-stride accumulation in registers, then a shared-memory tree
// reduction inside each block (__syncthreads between levels) and warp shuffles for
// the last 32 values. Stage 2 (host): partials summed in block order -> deterministic.
constexpr int kReduceThreads = 256;

__device__ __forceinline__ void blockReduce4(float& sum, float& mx, float& hs, float& cnt, float4* out) {
    __shared__ float sSum[kReduceThreads], sMax[kReduceThreads], sH[kReduceThreads], sC[kReduceThreads];
    const int t = threadIdx.x;
    sSum[t] = sum; sMax[t] = mx; sH[t] = hs; sC[t] = cnt;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (t < stride) {
            sSum[t] += sSum[t + stride];
            sMax[t] = fmaxf(sMax[t], sMax[t + stride]);
            sH[t] += sH[t + stride];
            sC[t] += sC[t + stride];
        }
        __syncthreads();
    }
    if (t < 32) {
        sum = sSum[t]; mx = sMax[t]; hs = sH[t]; cnt = sC[t];
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, off);
            mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
            hs += __shfl_down_sync(0xffffffffu, hs, off);
            cnt += __shfl_down_sync(0xffffffffu, cnt, off);
        }
        if (t == 0) out[blockIdx.x] = make_float4(sum, mx, hs, cnt);
    }
}

__global__ void k_reduce_soa(PatientSoA p, int n, float4* partials) {
    float sum = 0.f, mx = 0.f, hs = 0.f, cnt = 0.f;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < n; id += gridDim.x * blockDim.x) {
        const int32_t st = p.state[id];
        if (st == ST_NOT_ARRIVED) continue;
        const float w = p.waitingTime[id];
        sum += w;
        mx = fmaxf(mx, w);
        if (inMask(kActiveMask, st)) {
            hs += p.health[id];
            cnt += 1.f;
        }
    }
    blockReduce4(sum, mx, hs, cnt, partials);
}
__global__ void k_reduce_aos(const PatientAoS* p, int n, float4* partials) {
    float sum = 0.f, mx = 0.f, hs = 0.f, cnt = 0.f;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < n; id += gridDim.x * blockDim.x) {
        const PatientAoS q = p[id];
        if (q.state == ST_NOT_ARRIVED) continue;
        sum += q.waitingTime;
        mx = fmaxf(mx, q.waitingTime);
        if (inMask(kActiveMask, q.state)) {
            hs += q.health;
            cnt += 1.f;
        }
    }
    blockReduce4(sum, mx, hs, cnt, partials);
}

int reduceBlockCount(int n, int /*bs*/) {
    int b = gridFor(n, kReduceThreads);
    return b < 1 ? 1 : (b > 512 ? 512 : b);
}
void launchReduceSoA(const PatientSoA& p, int n, float4* partials, int bs, cudaStream_t s) {
    k_reduce_soa<<<reduceBlockCount(n, bs), kReduceThreads, 0, s>>>(p, n, partials);
    CUDA_CHECK_LAUNCH();
}
void launchReduceAoS(const PatientAoS* p, int n, float4* partials, int bs, cudaStream_t s) {
    k_reduce_aos<<<reduceBlockCount(n, bs), kReduceThreads, 0, s>>>(p, n, partials);
    CUDA_CHECK_LAUNCH();
}

}  // namespace gpu
}  // namespace hs

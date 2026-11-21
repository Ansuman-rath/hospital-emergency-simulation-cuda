// PatientKernels.cuh - host-side launch wrappers for the patient kernels.
#pragma once
#include <cuda_runtime.h>

#include "Patient.h"

namespace hs {
namespace gpu {

// Route table and model parameters live in __constant__ memory (broadcast reads,
// cached). `owner` identifies which engine uploaded them so different engines
// (live simulation vs benchmark) re-upload when they take over.
void setConstants(const void* owner, const UpdateParams& params, const RouteTable& routes);

inline int gridFor(int n, int bs) { return (n + bs - 1) / bs; }

// Naive split kernels (one per operation) - AoS
void launchSplitAoS(PatientAoS* p, uint32_t* bits, int n, int bs, cudaStream_t s);
// Fused kernel - AoS (+ warp-aggregated compaction)
void launchFusedAoS(PatientAoS* p, int n, SimEvent* ev, int* evCount, int bs, cudaStream_t s);
// Split kernels - SoA, followed by a compaction kernel
void launchSplitSoA(const PatientSoA& p, uint32_t* bits, int n, SimEvent* ev, int* evCount, int bs, cudaStream_t s);
// Fused kernel - SoA (+ warp-aggregated compaction)   <- optimized path
// step/stopAt: optional multi-step batching (see k_fused_soa)
void launchFusedSoA(const PatientSoA& p, int n, SimEvent* ev, int* evCount, int bs, cudaStream_t s, int step = 0,
                    int* stopAt = nullptr);
// Fused kernel over an explicit active index list
void launchFusedSoAActive(const PatientSoA& p, const int32_t* idx, int count, SimEvent* ev, int* evCount, int bs,
                          cudaStream_t s);

// CPU commands -> patient arrays (scatter, one thread per command)
void launchApplyCommandsSoA(const PatientSoA& p, int n, const PatientCommand* c, int nc, int bs, cudaStream_t s);
void launchApplyCommandsAoS(PatientAoS* p, int n, const PatientCommand* c, int nc, int bs, cudaStream_t s);

// Render packing
void launchPackRenderSoA(const PatientSoA& p, int n, RenderInstance* out, int bs, cudaStream_t s);

// Two-stage deterministic reduction: per-block partials (shared memory tree), host sums blocks in order.
int reduceBlockCount(int n, int bs);
void launchReduceSoA(const PatientSoA& p, int n, float4* partials, int bs, cudaStream_t s);
void launchReduceAoS(const PatientAoS* p, int n, float4* partials, int bs, cudaStream_t s);

}  // namespace gpu
}  // namespace hs

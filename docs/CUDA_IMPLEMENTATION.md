# CUDA Implementation

Hardware used for development and measurements: **NVIDIA GeForce RTX 5060 Laptop GPU**
(Blackwell, compute capability 12.0, 26 SMs, 8 GB GDDR7 on a 128-bit bus, 32 MB L2),
CUDA 13.5, driver 621.01 (WDDM), MSVC 19.44, Windows 11.
Code is compiled for the native architecture (`CMAKE_CUDA_ARCHITECTURES=native` → `sm_120`).

## 1. What runs on the GPU and what does not

| Computation | Device | Reason |
|---|---|---|
| Patient movement along route waypoints | **GPU** | independent per patient, O(N) every step |
| Waiting-time accumulation | **GPU** | independent per patient |
| Treatment / service progress, completion detection | **GPU** | independent per patient |
| Deterioration and priority escalation (synthetic health score) | **GPU** | independent per patient |
| Event detection + stream compaction | **GPU** | turns an N-sized result into a short list |
| Applying CPU decisions (scatter of commands) | **GPU** | keeps patient data resident on the device |
| Waiting-time / health reductions for the dashboard | **GPU** | O(N) reduction, shared-memory tree |
| Render-instance packing (+ optional CUDA-GL interop) | **GPU** | O(N) per frame, avoids a host round trip |
| Patient generation, arrival schedule, priority assignment | CPU | done once, sequential RNG stream |
| Queue ordering (priority + FIFO), escalation re-keying | CPU | ordered data structure shared by all patients |
| Resource allocation (doctors, beds, nurses, lab) | CPU | contention for shared resources; must be race-free and deterministic |
| State-machine transitions on events | CPU | depend on queues/resources |
| Metrics bookkeeping, configuration, rendering control | CPU | control logic |

Resource allocation is deliberately **not** forced onto the GPU. With hundreds of patients
competing for a few dozen beds, the work is small and sequentially dependent. A GPU version
would need atomics plus a deterministic conflict-resolution pass (e.g. sort by priority, then
prefix-sum ranks against free slots) and would gain nothing at realistic resource counts.

## 2. Thread hierarchy

```
grid  = ceil(N / blockDim) blocks          (1-D)
block = blockDim threads (32 … 1024, default 256 = 8 warps)
thread i  ⇔  patient  id = blockIdx.x * blockDim.x + threadIdx.x ,  guarded by  if (id < n)
```

* The bounds check never uses an early `return` before warp-collective operations. Threads with
  `id >= n` set `bits = 0` and still take part in `__ballot_sync` (warp-aggregated compaction
  requires all 32 lanes). Block sizes are rounded to a multiple of 32.
* The scatter kernel uses one thread per *command* and validates the CPU-provided patient id
  (`0 <= id < n`).
* The reduction kernel uses a capped grid (≤ 512 blocks × 256 threads) with a grid-stride loop.

## 3. Kernels

| Kernel | Variant | Work per thread |
|---|---|---|
| `k_clear_bits`, `k_move_aos`, `k_wait_aos`, `k_service_aos`, `k_health_aos` | **naive** | one operation, re-reads/re-writes the 52-byte struct each time |
| `k_fused_aos` | AoS fused | all four operations in registers, one load + one store of the struct |
| `k_move_soa` … `k_health_soa` + `k_compact_soa` | SoA split | each kernel loads only the fields it needs |
| `k_fused_soa` | **optimized** | loads `state` first, so inactive patients cost one 4-byte load. Then **selective** loads and stores: positions/route only if moving, timer only in service states, health/priority only where they change. Fused update, warp-aggregated event append, optional multi-step early exit |
| `k_fused_soa_active` | experiment | same, over an explicit index list of active patients |
| `k_apply_cmds_soa/aos` | both | scatter CPU decisions into the arrays |
| `k_pack_render_soa` | both | `{x, z, state | priority<<8 | flags<<16}` for instanced rendering |
| `k_reduce_soa/aos` | both | Σ waiting, max waiting, Σ health, active count |

All per-patient arithmetic is in `include/PatientUpdate.h` as `__host__ __device__` functions:
`moveStep`, `waitStep`, `serviceStep`, `healthStep`, `updatePatient`. The CPU baseline calls the
**same functions**, so the computations are equivalent by construction.

## 4. Memory hierarchy

| Memory | Used for | Notes |
|---|---|---|
| **Registers** | all per-patient working values inside the fused kernel | loaded once, updated in registers, stored once |
| **Global memory** | patient SoA arrays (13 × N × 4 B), event buffer, command buffer, render buffer | resident across steps; one `cudaMalloc` per field (256-B aligned) |
| **Constant memory** | `UpdateParams` (dt, decay/escalation thresholds) and the 81-route waypoint table (≈2.6 KB) | read-only, cached; route reads are divergent (different routes per thread), so they partially serialise. The table is tiny and stays in the constant cache |
| **Shared memory** | block-level tree reduction in `k_reduce_*` (4 × 256 floats) | `__syncthreads()` between levels, warp shuffles (`__shfl_down_sync`) for the final 32 |
| **Pinned (page-locked) host memory** | command upload, event download, render download, SoA staging | enables asynchronous DMA with `cudaMemcpyAsync` on a non-blocking stream |
| L1/L2 caches | implicit | the 32 MB L2 holds the whole SoA for N ≲ 600k, which matters for the repeated-step benchmarks |

### Coalescing: AoS vs SoA

*AoS:* thread *i* reads `p[i].x` at byte `52·i`. A warp's 32 loads of one field span 32 × 52 B =
1664 B, i.e. ~13 × 128-byte segments, of which only 4 bytes per 52 are useful. All 13 fields are
eventually touched, so the cache lines are reused from L1/L2, but every *instruction* is a scattered
access.

*SoA:* thread *i* reads `x[i]` at byte `4·i`. A warp's 32 loads hit exactly one 128-byte segment
(fully coalesced). The fused SoA kernel also skips the 12 other fields of inactive patients,
which AoS cannot (the struct is loaded as a whole).

Whether this matters in practice was **measured**, not assumed; see `BENCHMARKS.md`.

## 5. Synchronisation

| Mechanism | Where | Purpose |
|---|---|---|
| Kernel boundaries (same stream) | between the split kernels and between steps | global barrier: step *k+1* sees all writes of step *k* |
| `__syncthreads()` | reduction tree | block-level barrier on shared memory |
| `__ballot_sync`, `__shfl_sync`, `__popc` | event compaction | warp-level vote + broadcast, no shared memory |
| `atomicAdd` (one per warp) | event count | reserve output slots |
| `atomicMin` | multi-step batching | earliest step in the batch that produced an event |
| `cudaStreamSynchronize` | end of a step / batch | host needs the events before the CPU controller can run |
| CUDA events | timing | kernel and transfer time on the GPU timeline |

There are **no data races**:

* Each patient is written only by its own thread.
* Event slots are reserved atomically.
* The scatter kernel receives at most one command per patient per step (the controller
  de-duplicates with a dirty flag).
* Resource allocation runs on the CPU only.

`atomicAdd` makes the order of events in the buffer nondeterministic, so the host sorts the
(small) event list by id before use.

## 6. Optimizations implemented (and what was measured)

1. **SoA layout** instead of AoS: coalesced accesses, skip loads for inactive patients, and
   **selective field loads** in the fused kernel (added after measuring that the first fused kernel was DRAM-bound at 1M).
2. **Kernel fusion:** 1 kernel instead of 4 (+clear). Fewer launches, and each field is read/written
   once per step instead of once per operation.
3. **Device-resident data:** no full-array copy per step. The naive variant copies 52·N bytes in
   each direction every step.
4. **Event compaction on the GPU** with warp-aggregated atomics: one `atomicAdd` per warp
   instead of per event, and the host receives `count + events` instead of N flags.
5. **Single synchronisation per step:** the event count and a speculative, power-of-two-sized
   prefix of the event buffer are copied in one async chain. A second copy is needed only on overflow.
6. **Pinned memory + non-blocking stream** for every per-step transfer.
7. **Event-driven multi-step batching:** K kernels per host synchronisation, with on-device early exit.
8. **CUDA Graph variant:** memset + kernel + 2 copies replayed as one graph launch. Measured; the gain
   on this WDDM laptop is small because the floor is synchronisation latency, not launch cost.
9. **CUDA-OpenGL interop (optional):** the render-pack kernel writes into the mapped VBO.
10. **Block-size sweep and ordering experiments** (zone/state-sorted vs random order; active-index
    list vs full range).

Each optimization has a measured result in `BENCHMARKS.md`, including the ones that did **not** help.

## 7. Error handling and resource management

* `CUDA_CHECK(expr)` wraps every runtime call and throws `std::runtime_error` with the error name,
  expression, file and line. `CUDA_CHECK_LAUNCH()` checks `cudaGetLastError()` after launches.
* `DeviceBuffer<T>`, `PinnedBuffer<T>`, `DeviceSoA`, `EventTimer` are RAII wrappers, so
  `cudaFree`/`cudaFreeHost`/`cudaEventDestroy` run on every path. Graph executables and GL
  registrations are released in the backend destructor.
* `createCudaBackend()` returns `nullptr` with a message if no device exists. The GUI then
  falls back to the CPU backend **and says so** in the status line.

## 8. Profiling

See `docs/BENCHMARKS.md` §Profiling for Nsight Systems / Nsight Compute commands and the
measured occupancy, memory throughput and timeline observations.

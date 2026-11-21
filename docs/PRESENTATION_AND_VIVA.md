# PAGP Presentation (5–10 min) and Viva Preparation

Numbers quoted here come from `benchmarks/results/*.csv` (RTX 5060 Laptop, Core Ultra 7 255HX).
See `BENCHMARKS.md` for the full tables. Re-run `hospital_sim --benchmark` on the examination
machine if it differs.

## Part A – Presentation outline

**Slide 1 – Problem (45 s).** Emergency departments face surges (accidents, ambulance waves, mass-casualty
events). Testing staffing and bed configurations in a real hospital is impossible, so we simulate them.
Simulating many patients means repeating, every 3 simulated seconds, movement, waiting-time, treatment-progress
and deterioration updates for every patient. These are independent per patient: a data-parallel workload.

**Slide 2 – System (60 s).** Hybrid architecture diagram (`ARCHITECTURE.md` §1, §4). CPU = controller
(arrivals, queues, allocation, state machine). GPU = per-patient updates, event compaction, reductions,
render packing. OpenGL = 3D view; ImGui = dashboard.

**Slide 3 – Why not everything on the GPU (45 s).** Allocation is a contention problem: 400 patients, 16 ER beds,
8 doctors. It is sequentially dependent and needs deterministic priority order. A GPU version needs atomics plus a
conflict-resolution pass and gains nothing at these sizes. We keep it serial on the CPU, which makes it race-free
by construction.

**Slide 4 – CUDA mapping (75 s).** One thread per patient: `id = blockIdx.x*blockDim.x + threadIdx.x`, `if (id < n)`.
Fused kernel: load state, return early for inactive patients, load the remaining fields into registers, run
`updatePatient` (the same `__host__ __device__` code as the CPU), store, then append events with warp-aggregated
atomics. Memory: SoA in global memory, route table and parameters in constant memory, shared-memory tree reduction,
pinned host buffers.

**Slide 5 – Correctness (45 s).** `--fmad=false` + `/fp:precise`: GPU output is bit-identical to the CPU. Event
streams are hashed, and every benchmark row is validated. 14 tests / 447 checks, including race-free allocation,
no double booking, no discharge before treatment and waiting-time consistency.

**Slide 6 – Results: scaling (75 s).** Graphs 01 and 02. Small N: the CPU wins (launch + synchronisation
latency). The crossover vs one core is ~5k patients (kernel) and ~10k (end-to-end); vs 20 CPU threads it is ~50k.
Peak speedup is 349× kernel-only / 49× end-to-end at 500k patients. At 1M the data (52 MB) no longer fits in the
32 MB L2, so the kernel becomes DRAM-bound and the speedup drops to 109× / 36×. End-to-end is lower because of
per-step synchronisation, so we show both.

**Slide 7 – Results: optimizations (75 s).** Graph 03 (AoS vs SoA, fusion), graph 04 (block size), graph 05 (naive
vs optimized). The naive version copies the whole array both ways every step: 1.1 s of its 1.6 s at 1M is transfers.
Story worth telling: the first fused SoA kernel *lost* to the split kernels at 1M. Measuring showed bytes moved
mattered, so selective loads were added (−39%). Zone/state sorting gives another 3.9× on the kernel.

**Slide 8 – System-level lesson (60 s).** Full simulation (graph 07). Nsight Systems: kernel 2.3 µs, but
`cudaStreamSynchronize` 41 µs and ~10 µs per launch/copy call on WDDM. The coupled loop is latency-bound, so 20 CPU
threads beat the GPU there (10.8× vs 6.9× at 100k). CUDA Graphs do not help (the floor is synchronisation, not
launch). **Event-driven batching** (K steps per synchronisation, with on-device early exit) cuts synchronisations by
12× in the Normal scenario with identical results. Amdahl: the serial controller bounds the end-to-end speedup.

**Slide 9 – Demo + conclusion (60 s).** Live: mass-casualty, bottleneck detection, A/B comparison, live speedup button.

## Part B – Likely viva questions and answers

**Q1. Why is one thread per patient valid?**
Each patient's update reads and writes only that patient's fields plus read-only shared data (route table,
parameters). There are no inter-patient dependencies inside a step, so the iterations of the CPU loop are
independent and can run in any order: an embarrassingly parallel map.

**Q2. Is one thread per patient optimal?**
Not always. The work per patient is small (≈ 50–100 instructions) and memory-bound. For very small N there are
too few warps to fill 26 SMs (we need ≳ 26 × 48 warps × 32 ≈ 40k threads for full occupancy), so the GPU is
latency-bound. One could process several patients per thread (grid-stride) to amortise index math, but measurements
show the kernel is already bandwidth-limited at large N. The reduction kernel *does* use a grid-stride loop with a
capped grid.

**Q3. Explain grid, block and thread for your kernel.**
1-D grid of `ceil(N/256)` blocks of 256 threads (8 warps). Each SM runs several blocks concurrently, limited by
registers, shared memory and the 48-warp limit. Warps of 32 threads execute in SIMT. A block is the unit of
scheduling and of shared memory/`__syncthreads()` scope.

**Q4. What is coalescing and how did you achieve it?**
When the 32 threads of a warp access consecutive 4-byte words, the hardware serves the warp with one 128-byte
transaction. With SoA, `x[id]` for consecutive `id` is consecutive in memory, so every field load is coalesced.
With AoS, `p[id].x` is strided by 52 bytes, so a warp touches about 13 cache lines per field.

**Q5. Did SoA always win?**
No, and we measured it. Fused SoA beats fused AoS by 1.3–3.3× (e.g. 29.7 vs 59.6 ms at 1M). But our *first* fused
SoA kernel lost to the split SoA kernels at 1M (48.5 vs 40.3 ms): beyond the L2 cache the kernel is DRAM-bound, and
it loaded all 13 fields of every active patient while the split kernels loaded only what each state needed. Making
the fused kernel's loads selective fixed it (29.7 ms). The big AoS loss comes from re-reading the struct in each
split kernel (287 ms) and from the naive per-step transfers.

**Q6. How do you avoid race conditions?**
(1) Each thread writes only its own patient. (2) Event output slots are reserved with `atomicAdd` (one per warp);
there are no overlapping writes. (3) Shared resources (beds, doctors) are allocated serially on the CPU, so two
patients cannot obtain the same bed. `ResourcePool::acquire` additionally throws if a slot is handed out twice.
(4) The CPU de-duplicates commands so the scatter kernel never sees two commands for the same patient.

**Q7. The atomics make the event order random. Is your simulation still deterministic?**
Yes. The host sorts the event list by patient id before the controller processes it. Tests run the optimized
backend with block sizes 128 and 256 and compare with the CPU: every record is identical.

**Q8. How did you prove CPU and GPU compute the same thing?**
The per-patient code is literally the same `__host__ __device__` function. FMA contraction is disabled
(`--fmad=false`), IEEE `sqrtf` and division are the defaults, and MSVC uses `/fp:precise`. So we assert *exact*
equality of the final state (all 13 fields), the per-step event stream (hash) and the full simulation outcome.
Tolerance: max |Δ| ≤ 1e-4, and the measured Δ is 0.

**Q9. What does `--fmad=false` cost?**
FMA fuses a multiply and an add into one instruction with a single rounding. Disabling it adds a few instructions,
but this kernel is memory-bound, so the impact is negligible. It is a deliberate choice for verifiable equivalence.

**Q10. Why use constant memory for the routes?**
The table is small (≈ 2.6 KB), read-only and used by all threads. The constant cache broadcasts when a warp reads
the same address. Our accesses are partly divergent (different routes), which serialises those reads. The data is
still cached and tiny, and the alternative (`__ldg` from global) was not needed.

**Q11. Where do you use shared memory and `__syncthreads()`?**
In the reduction kernel: each block reduces 256 per-thread partial sums in a shared-memory tree with a barrier
between levels, then the last warp uses `__shfl_down_sync`. Partials are summed on the host in block order, which
makes the result deterministic (a float `atomicAdd` would not be).

**Q12. What did the naive version do wrong?**
Every step it copied the whole patient array to the GPU and back (104 bytes × N per step) plus an N-sized flag
array, launched five kernels that each re-read the whole struct, and scanned all N flags on the CPU. Transfers
dominate, so it is slower end-to-end than the CPU for most N even though its kernels are faster.

**Q13. Kernel-only vs end-to-end speedup: which is "the" speedup?**
Both are reported and labelled. Kernel-only shows the computational benefit. End-to-end shows what a user
gets, including PCIe transfers, synchronisation and launch overhead. Hiding the latter would overstate the GPU.

**Q14. Why is the GPU slower for a 300-patient scenario?**
Each step does about 300 × 60 instructions, a few microseconds of CPU work. A GPU step costs a launch plus a
device→host synchronisation (≈ 40–50 µs on Windows WDDM), so the GPU only wins once per-step work exceeds that floor.
We reduced the number of synchronisations with event-driven batching, but for tiny N the CPU is the right choice.

**Q15. What is event-driven batching and why is it correct?**
If no patient produces an event, no arrival is due and no metrics sample is due, then the CPU controller has nothing
to do. So the GPU can run several steps without the CPU. We queue K kernels; each checks the device variable
`stopAt` (set by `atomicMin` when a step produces an event) and exits if an earlier step of the batch produced
events. The CPU reacts at exactly the right step, and the test `event_batching_identical` shows identical outcomes.

**Q16. Did CUDA Graphs help? Why not more?**
Measured: up to 21% while per-step event timing was on (many API calls), and 3–9% slower in the final uninstrumented
runs. Graphs remove per-API-call submission cost, but here the cost is waiting for the GPU
and getting the result back (synchronisation latency), which graphs do not remove.

**Q17. How did you choose the block size?**
By measurement (graph 04, N = 1M). 64–512 are within 8% of each other (64 fastest: 29.5 ms; 256: 31.6 ms) because the
kernel is memory-bound and occupancy is sufficient. 32 is worst (35.1 ms; one warp per block, so the per-SM block
limit caps resident warps). We keep 256 as a robust default across GPUs.

**Q18. What is occupancy and what limited it?**
Occupancy is the number of resident warps divided by the maximum (48 per SM on sm_120). The fused kernel uses few
registers (40/thread) and no shared memory, so theoretical occupancy is 100%. Nsight Compute measures 92% achieved
occupancy for `k_fused_soa` at 1M patients. Occupancy is therefore not the limiter: the kernel runs at 89% of peak
DRAM throughput against 40% SM throughput, so it is memory-bound (`BENCHMARKS.md` §8).

**Q19. Is there branch divergence?**
Yes: patients in different states take different paths (moving vs waiting vs in treatment), and `moveStep` has a
bounded loop. The ordering experiment sorts patients by state ("zone-based partitioning") so warps become uniform:
the SoA kernel gets 3.9× faster (31.6 → 8.1 ms at 1M), because whole warps also skip the same field loads. AoS gets
13% slower, since it loads the whole struct anyway. Nsight Compute confirms it: in random order only about 15 of
32 threads per warp are active in `k_fused_soa`. In the live simulation this would need periodic re-sorting.

**Q20. What is CUDA-OpenGL interop and is your project dependent on it?**
`cudaGraphicsGLRegisterBuffer` maps the OpenGL instance VBO into CUDA, so the pack kernel writes patient instances
directly, with no device→host→device copy. It is optional (a checkbox). If the GL context is on a different GPU
(Optimus), registration fails and we fall back to `glBufferSubData`.

**Q21. How is memory freed and errors handled?**
RAII classes (`DeviceBuffer`, `PinnedBuffer`, `DeviceSoA`, `EventTimer`) free memory in destructors. Every CUDA call
is wrapped in `CUDA_CHECK`, and launches are followed by `cudaGetLastError`. compute-sanitizer memcheck reports no
errors.

**Q22. How would you scale further?**
(1) Move the remaining host work per step off the critical path (persistent kernel or GPU-side event queues).
(2) GPU-side allocation with a deterministic parallel matching for very large hospitals (sort waiting patients by
key, prefix-sum free slots, assign the k-th free slot to the k-th patient). (3) Multiple scenarios in parallel
(one per stream) for Monte-Carlo studies, the natural next use of the GPU. (4) Half precision for positions.

**Q23. What are the limitations of the hospital model?**
Synthetic parameters, simplified pathways and no staff shifts or nurse-to-bed ratios. It is not clinically validated.
Results are for relative comparisons between configurations, not real predictions.

**Q24. Amdahl's law in your project?**
If the CPU controller takes fraction *s* of a step, the end-to-end speedup is ≤ 1/s regardless of the GPU. The
scenario benchmark reports controller time separately so this bound can be read directly from the CSV.

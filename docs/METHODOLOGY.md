# Methodology

## 1. Methodology flowchart

```
 ┌────────────────────────┐
 │ Scenario configuration │  preset (normal / surge / ambulance / mass-casualty),
 │  + deterministic seed  │  patients, arrival rate, P1–P4 mix, staff and beds
 └───────────┬────────────┘
             ▼
 ┌────────────────────────┐   splitmix64: arrival times (Poisson + surge / bursts / MCI),
 │ Patient generation     │   priority, pathway, durations, initial health
 │ (CPU, once)            │
 └───────────┬────────────┘
             ▼
 ┌────────────────────────┐   hot arrays uploaded once to GPU global memory (SoA)
 │ Initial upload         │   route table + model parameters -> __constant__ memory
 └───────────┬────────────┘
             ▼
 ╔═══════════════════════════════════════════════════════════════════════════╗
 ║ SIMULATION LOOP (per step or per event-driven batch of steps)            ║
 ║  ┌──────────────────────────────────────────────┐                         ║
 ║  │ GPU: fused per-patient kernel (1 thread = 1 patient)                   ║
 ║  │   move -> wait -> treatment progress -> deterioration/escalation       ║
 ║  │   -> warp-aggregated event compaction                                  ║
 ║  └───────────────┬──────────────────────────────┘                         ║
 ║                  ▼  D2H: event count + compact event list (pinned)        ║
 ║  ┌──────────────────────────────────────────────┐                         ║
 ║  │ CPU controller (sorted by id = deterministic) │                        ║
 ║  │  state machine -> arrivals -> queues (priority, FIFO)                  ║
 ║  │  -> serial resource allocation (race-free)                             ║
 ║  └───────────────┬──────────────────────────────┘                         ║
 ║                  ▼  H2D: commands for changed patients -> scatter kernel  ║
 ║  ┌──────────────────────────────────────────────┐                         ║
 ║  │ Metrics: GPU shared-memory reduction + CPU queue statistics            ║
 ║  └───────────────┬──────────────────────────────┘                         ║
 ╚══════════════════╪════════════════════════════════════════════════════════╝
                    ├────────────► OpenGL: render-pack kernel -> instanced 3D patients
                    ├────────────► ImGui dashboard: live metrics, bottlenecks, performance
                    ▼
 ┌────────────────────────┐   kernel suite (synthetic population) + scenario suite
 │ Experiments            │   CPU 1T / CPU MT / CUDA naive / optimized / graph / batched
 │ (benchmark mode)       │   correctness check vs CPU on every run -> CSV -> graphs
 └────────────────────────┘
```

## 2. Simulation model (synthetic)

| Element | Model |
|---|---|
| Arrivals | Non-homogeneous Poisson process (thinning) with base rate λ. Surge: λ × m in a window. Ambulance surge: periodic bursts of b patients within w minutes. Mass casualty: M patients uniformly within a window. The total is fixed by the "patients" parameter. |
| Priority | Sampled from a configurable P1–P4 mix (separate mixes for bursts and MCI). Revealed at triage; ambulance P1 patients are pre-triaged and fast-tracked to the ER queue. |
| Pathway | Diagnostics with probability `pDiag[p]`; disposition ICU / ward / home with `pICU[p]`, `pWard[p]`; ICU step-down to ward with `icuStepDown`. |
| Durations | Triage, diagnostics, treatment (mean × priority scale × U(0.4, 1.6)), ICU and ward stays. |
| Resources | Exclusive pools: doctors, triage nurses (one per desk), ER beds, ICU beds, ward beds, lab slots. ER treatment needs a doctor **and** an ER bed. |
| Boarding | A patient whose ICU/ward bed is not available keeps the ER (or ICU) bed. This models the downstream-bottleneck effect. |
| Queue policy | (current priority, enqueue sequence, id). Escalation raises priority but keeps the original FIFO position within the new level. |
| Deterioration | Untreated patients lose health at a priority-dependent rate. Below a threshold a triaged patient escalates P4→P3→P2→P1. Cared-for patients recover. |
| Movement | Waypoint routes `door → corridor → corridor → door` (constant memory) followed by the patient's personal target (bed, desk, queue slot), at ~1 m/s. |
| Time | Fixed step `dt` (default 0.05 min = 3 s). |

## 3. Experimental methodology

1. **Correctness before performance.** Every benchmark row records the maximum absolute float
   difference, integer field mismatches and the event-stream hash compared with the CPU single-thread
   reference run on the same input. All published rows are bit-identical (difference 0).
2. **Deterministic inputs.** Synthetic populations and scenarios are generated from fixed seeds.
3. **Warm-up.** Every GPU configuration runs 3 warm-up steps (module load, clocks, caches) before timing.
4. **Repeats.** Each configuration is measured `repeats` times (default 3); the median is reported.
5. **Timers.** CPU uses `std::chrono::steady_clock`. GPU kernel time uses CUDA events on the same stream.
   End-to-end time uses `steady_clock` around the complete host-visible operation.
6. **Two timing definitions, always both reported:**
   * *Kernel-only:* CUDA events around `steps` back-to-back launches with data resident on the GPU.
   * *End-to-end:* wall clock for initial upload + every step's kernel(s) + the transfers the CPU controller
     needs each step + final download, including synchronisation and launch overhead.
7. **Speedup = CPU single-thread time / GPU time,** stated per timing definition. The CPU multi-thread (OpenMP)
   result is reported alongside, so the GPU is not only compared with a single core.
8. **Full-system benchmark.** The complete hybrid simulation (controller + backend) runs for a fixed simulated
   time on scaled mass-casualty scenarios. Outcomes (state, treatment start, discharge time, beds, escalations)
   are compared record by record with the CPU single-thread run.
9. **Instrumentation overhead** is itself measured (`--no-event-timing`).

## 4. Threats to validity

* Laptop GPU under Windows WDDM: launch/synchronisation latency is higher than on Linux/TCC. Clocks vary
  with power and thermal state (medians mitigate this).
* The per-patient update is memory-bound and light (tens of FLOPs). Speedups therefore reflect memory
  bandwidth (GDDR7 vs DDR5), not arithmetic throughput.
* For N ≤ ~600k the whole SoA fits in the 32 MB L2 cache, so repeated steps benefit from L2 residency.
* The synthetic population mixes states realistically, but a real ED census would differ.
* The simulation model is not clinically validated; decision-support comparisons are relative
  comparisons between configurations, not predictions.

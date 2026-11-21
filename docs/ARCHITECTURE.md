# Architecture

> This software is a **simulation and analysis tool**. Priorities, durations, deterioration
> rates and pathways are synthetic model parameters. It is not a hospital management
> system and not a clinically validated triage protocol.

## 1. Module overview

```
                ┌─────────────────────────────────────────────────────────────┐
                │                    hospital_sim.exe                         │
                │  main.cpp ─ CLI modes (--benchmark, --run, --compare, ...)  │
                │  App ─ GLFW window, main loop, background jobs              │
                │   ├── Renderer   (OpenGL 3.3: static scene + instancing)   │
                │   ├── Camera     (top-down / orbit / free-look / follow)   │
                │   └── Dashboard  (Dear ImGui panels, overlay labels)       │
                └──────────────────────────────┬──────────────────────────────┘
                                               │ uses
┌──────────────────────────────────────────────▼──────────────────────────────────────┐
│ hs_core (static library, plain C++17)                                               │
│  Scenario ─ presets & configuration     PatientGenerator ─ deterministic arrivals   │
│  Hospital ─ floor plan, routes, slots   ResourceManager ─ pools + priority queues   │
│  SimulationEngine ─ CPU controller: state machine, allocation, metrics, batching    │
│  PatientBackend (interface) ─ CpuBackend (1 thread / OpenMP)                        │
│  Benchmark ─ experiments, validation, CSV                                           │
└──────────────────────────────────────────────┬──────────────────────────────────────┘
                                               │ PatientBackend implementation
┌──────────────────────────────────────────────▼──────────────────────────────────────┐
│ hs_cuda (static library, CUDA C++)                                                  │
│  CudaEngine.cu    ─ CudaBackend: NaiveAoS / OptimizedSoA / OptimizedGraph, interop  │
│  PatientKernels.cu ─ kernels + __constant__ route table and model parameters        │
│  MemoryManager.cu  ─ RAII device/pinned buffers, device SoA, CUDA_CHECK             │
│  CudaBenchmark.cu  ─ kernel-level experiments (CUDA events + wall clock)            │
└─────────────────────────────────────────────────────────────────────────────────────┘
 shared by CPU and GPU:  include/Patient.h (data model), include/PatientUpdate.h (__host__ __device__ rules)
```

Simulation, CUDA computation, resource management, rendering and benchmarking are
separate compilation units. The renderer never touches simulation internals: it
consumes `RenderInstance {x, z, packed state|priority|flags}` arrays (or a VBO that
CUDA wrote directly through CUDA-OpenGL interop).

If CUDA is not installed, CMake builds `src/CudaStub.cpp` instead of `hs_cuda`. Every
GPU entry point then reports *unavailable*. Nothing is emulated on the CPU under a GPU label.

## 2. Data model: hot / cold split

| Data | Where | Updated by | Purpose |
|---|---|---|---|
| **Hot** (`PatientAoS` / `PatientSoA`, 13 fields, 52 B) position, target, speed, waiting time, remaining service, health, state, priority, route, waypoint, flags | GPU global memory (CUDA backends) or host vectors (CPU backend) | per-patient update every step (parallel) | movement, waiting, treatment progress, deterioration |
| **Cold** (`PatientRecord`) arrival time, pathway, durations, assigned doctor/beds, timestamps | CPU only | SimulationEngine on events | state machine, queues, allocation, metrics |

The hot data stays resident on the GPU for the whole run. The CPU sends only *commands*
(state changes for patients it touched), and the GPU returns only *events*
(patients needing a CPU decision).

## 3. Patient state machine

```
NOT_ARRIVED ──► ARRIVING ──► WAIT_TRIAGE ──► IN_TRIAGE ──┬──► WAIT_DIAG ──► IN_DIAG ──┐
     │                                                    └───────────────────────────┤
     │ (ambulance P1: pre-triaged fast track)                                         ▼
     └────────────────────────────────────────────────────────────────────────► WAIT_ER
                                                                                     │ doctor + ER bed
                                                                                     ▼
                                                                              IN_TREATMENT
                                         ┌───────────────────┬───────────────────────┤
                                         ▼                   ▼                       ▼
                                   WAIT_ICU (boards     WAIT_WARD (boards      DISCHARGING ──► DISCHARGED
                                   in ER bed)           in ER/ICU bed)               ▲
                                         │ ICU bed           │ ward bed              │
                                         ▼                   ▼                       │
                                      IN_ICU ──(step-down)──►WAIT_WARD   IN_WARD ────┘
                                         └──────────(home)───────────────────────────┘
```

The allowed transitions are listed in `tests/test_main.cpp` (`allowedTransitions`). Every
transition of a full run is checked against that table.

## 4. One simulation step (hybrid CPU-GPU)

| # | Phase | Runs on | Why there |
|---|---|---|---|
| 1 | Per-patient update: movement along waypoints, waiting time, service countdown, deterioration + escalation, event detection | **GPU** (CUDA) or CPU backend | independent per patient → one thread per patient |
| 2 | Event compaction (patients that arrived / finished service / escalated) | **GPU** (warp-aggregated atomics) | avoids copying N records back |
| 3 | Event download (count + compact list) | PCIe D2H | only what the CPU needs |
| 4 | State machine reactions to events (sorted by id) | CPU | shared dependencies (queues, resources) |
| 5 | Activate arrivals due at this time | CPU | sequential schedule |
| 6 | Resource allocation (ward → ICU → triage → diagnostics → ER) | CPU | contention for shared resources; serial = race-free and deterministic |
| 7 | Command upload + scatter kernel | PCIe H2D + **GPU** | only changed patients |
| — | Metrics reduction (sum/max waiting time, health) | **GPU** (shared-memory tree) on demand | O(N) reduction for the dashboard |
| — | Render packing (`x, z, state/priority`) | **GPU** kernel, optional CUDA-GL interop | O(N) per frame |

**Event-driven batching.** When a step produces no events, no arrival is due and no
sampling point is reached, phases 4–7 do no work. The engine therefore asks the backend
to advance *up to K steps* (`stepMulti`). The CUDA backend enqueues K kernels with one
synchronisation, and the kernels after the first event-producing step exit immediately.
Outcomes are identical (tested in `event_batching_identical`).

## 5. Determinism

* All randomness is generated on the CPU from `ScenarioConfig::seed` with splitmix64.
  Per-patient attributes come from an independent stream `hash(seed, id)`.
* Events produced by atomics on the GPU arrive in nondeterministic order. They are sorted by
  patient id before the controller sees them.
* Allocation is serial in a fixed order with a total queue order `(priority, FIFO sequence, id)`
  and lowest-free-slot resource assignment.
* CUDA is compiled with `--fmad=false`, and MSVC uses `/fp:precise`. The CPU and GPU arithmetic is
  bit-identical; the tests assert exact equality.

## 6. Invariants and where they are enforced

| Invariant | Enforcement | Test |
|---|---|---|
| A bed / doctor / desk is never assigned twice | `ResourcePool::acquire` only hands out free slots and throws if the slot is occupied; `release` verifies the occupant | `resource_allocation_and_release`, `checkInvariants` |
| Exclusive doctor | doctor pool, `checkInvariants` ("holds a doctor outside treatment") | `state_machine_full_run_cpu` |
| No discharge before treatment | `SimulationEngine::discharge` throws if `!treatmentDone` | `state_machine_full_run_cpu` |
| Consistent waiting time | GPU accumulated `waitingTime` == Σ CPU queue timestamps | `waiting_time_consistency` |
| Priority queue policy | `QueueKey` ordering, escalation re-keys but keeps FIFO sequence; work-conserving ER allocation | `queue_ordering_priority_then_fifo`, `state_machine_full_run_cpu` |
| GPU bounds | `if (id < n)` in every kernel; scatter kernel rejects out-of-range ids; compute-sanitizer memcheck | `gpu_zero_patients_and_bounds`, `docs/BENCHMARKS.md` §Sanitizer |

## 7. Rendering pipeline

* Static scene built once per reset (`Renderer::buildScene`): floors, low walls with door
  gaps, beds (ICU beds with monitors), triage desks, lab machines, waiting chairs, ambulance.
  It is one VBO/IBO, drawn with a Blinn-Phong shader (directional light + hemisphere ambient + fog).
* Patients: one `glDrawElementsInstanced` call for *all* patients. The instance attribute
  is `RenderInstance` (12 B). The vertex shader decodes state/priority/flags. Body colour = priority
  (grey = untriaged), floating marker = state (pulsing while waiting), and patients in beds are
  rotated to lie down. Inactive patients are moved outside the clip volume.
* Staff: doctors walk to the bed of the patient they treat, and nurses stand at staffed triage desks.
* Labels and legend: ImGui background draw list, using 3D→2D projection.

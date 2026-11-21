# User Guide

## 1. Starting the application

```
build\bin\hospital_sim.exe
```

The window opens maximised. The console prints the OpenGL renderer. On hybrid laptops
the program asks the NVIDIA driver for the discrete GPU (`NvOptimusEnablement`).

## 2. Screen layout

| Area | Contents |
|---|---|
| **Control Panel** (left) | Start / Pause / Resume / Reset / Step, simulation speed, scenario preset and parameters, hospital resources, engine (backend, block size, interop, dt), camera and view options |
| **3D view** (centre) | hospital, patients, staff, department labels with live occupancy |
| **Hospital Dashboard** (right) | tabs: *Live Metrics*, *Performance*, *Benchmarks*, *Decision Support*, *Patient* |
| **Status bar** (bottom) | running/paused, CPU or GPU backend, simulated time, counts, colour legend |

## 3. Reading the 3D view

| Visual | Meaning |
|---|---|
| Body colour red / orange / yellow / green | P1 Critical / P2 Serious / P3 Moderate / P4 Minor |
| Body light grey | not yet triaged |
| Cube above the head | patient state (colour). **Pulsing** = waiting in a queue |
| Patient lying on a bed | in treatment, boarding, ICU or ward stay |
| White figures (blue band = busy, grey = idle) | doctors; they walk to the bed of the patient they treat |
| Green figures | triage nurses at their desks |
| Rows of patients outside the walls | queue overflow (waiting room / triage queue full) |

Click a patient to select it (the label shows id and state). The *Patient* tab shows its record,
including the GPU-side waiting time, remaining service time and synthetic health.

## 4. Controls

| Input | Action |
|---|---|
| Left-drag | pan |
| Right-drag | rotate (orbit / follow) or look around (free-look) |
| Mouse wheel | zoom |
| W A S D (Q/E) | pan, or move in free-look (Q/E down/up), Shift = faster |
| 1 / 2 / 3 / 4 | top-down / orbit / free-look / follow selected patient |
| Space | start / pause |
| Esc | deselect patient |

## 5. Running a scenario

1. Choose a **Preset** (this resets the simulation): Normal, Patient Surge, Ambulance Surge, Mass-Casualty.
2. Optionally edit patients, arrival rate, priority mix, surge/burst/MCI parameters and resources.
   **Changes apply on Reset.**
3. Choose the **Backend**:
   * CPU single-thread / CPU multi-thread (OpenMP)
   * CUDA naive: AoS, split kernels, full copies every step
   * CUDA optimized: SoA, fused kernel, compacted events, event-driven batching
   * CUDA optimized + Graph
4. Press **Start**. Set **Speed** in simulated minutes per real second. If the backend cannot keep up,
   a red message shows the achieved steps/s. Try 100,000 patients at high speed with the CPU and then
   with the CUDA backend to see the difference live.

## 6. Dashboard tabs

* **Live Metrics:** totals, waiting/in-treatment/discharged, P1–P4 counts, resource occupancy and
  time-averaged utilisation, queue table (length, average/max wait, served), **bottleneck**
  (largest queue, longest average wait), average/max waiting time (GPU reduction, with a CPU
  cross-check), treatment delay, length of stay, throughput, escalations, time-series plots.
* **Performance (live):** current step wall time, kernel time, H2D/D2H, controller time, bytes per step,
  steps/s, patient updates/s, frame time. *Measure live speedup* snapshots the current state and
  times 200 steps on CPU-1T, CPU-MT and CUDA, checking that the outputs are identical.
  These are live numbers for the running simulation, **not** benchmark results.
* **Benchmarks:** runs the quick or full benchmark in a background thread. The simulation pauses while it runs.
  Results are shown per experiment, and CSVs are written to `benchmarks/results_gui/`.
* **Decision Support:** configuration A (Control Panel) vs B (sliders). Both run the *same* seeded scenario
  to completion headless. The table compares waiting times, treatment delay, length of stay, time to clear,
  throughput, per-queue waits and utilisation. These are simulation results, not medical recommendations.
* **Patient:** inspector, *select a random active P1*, *follow with camera*.

## 7. Command-line modes

| Command | Purpose |
|---|---|
| `hospital_sim --info` | CPU / OpenMP / CUDA / GPU information |
| `hospital_sim --benchmark [--quick]` | kernel suite + scenario suite → `benchmarks/results/*.csv` |
| `hospital_sim --kernel-bench` / `--scenario-bench` | one suite only |
| options: `--out DIR --steps N --repeats R --seed S --block B` | benchmark parameters |
| `hospital_sim --run <0-3> [--backend cpu\|cpumt\|naive\|gpu\|graph] [--patients N]` | headless scenario run with metrics |
| `hospital_sim --compare <0-3>` | configuration A vs B (+5 doctors, +10 ICU beds) |
| `hospital_sim --no-event-timing ...` | disable per-step CUDA event timing (overhead measurement) |
| `hospital_sim --screenshot file.bmp --scenario 3 --sim-minutes 150 --camera 1` | render one frame to a BMP and exit |
| `python benchmarks/plot_results.py [dir]` | graphs from the CSVs → `<dir>/graphs/*.png` |
| `build\bin\hospital_tests.exe [filter]` | unit + integration tests |

## 8. Demonstration script (≈ 8 minutes)

1. **(1 min) Context.** Normal preset, orbit camera. Point out departments, colours, staff. Start at 20 sim-min/s:
   patients arrive, are triaged (grey → colour), wait (pulsing), get treated (lie down), leave.
2. **(1 min) Mass-casualty.** Select the preset and start. The triage queue and waiting room spill outside, P1/P2
   escalations appear, and ER occupancy and doctors hit 100%. The Live Metrics tab names the bottleneck.
3. **(1 min) Patient view.** Click a red patient → Patient tab → *Follow with camera* (key 4).
4. **(1.5 min) Decision support.** Set B = +5 doctors, +10 ICU beds and run the comparison. Discuss the lower
   waits, and that the bottleneck moves (e.g. to ICU admission / boarding).
5. **(2 min) GPU.** Set Patients to 100000 and Arrivals/hour to ~3000 (or scale MCI). With the CPU backend at high
   speed the red "cannot keep up" message appears. Switch to CUDA optimized: steps/s jump. On the Performance tab,
   click *Measure live speedup* and point out that the outputs are identical.
6. **(1.5 min) Benchmarks.** Show `benchmarks/results/graphs`. Cover the crossover point, kernel vs end-to-end
   speedup, AoS vs SoA, block size, and why naive CUDA loses to the CPU (transfers).

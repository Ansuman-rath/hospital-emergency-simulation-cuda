@echo off
rem Collect Nsight Compute kernel metrics (occupancy, memory throughput, divergence, ...).
rem GPU performance counters need administrator rights on Windows by default:
rem   right-click -> "Run as administrator"   (or enable counter access for all users in the
rem   NVIDIA Control Panel: Developer -> Manage GPU Performance Counters).
rem Each kernel is profiled in its own pass (one launch after one warm-up launch), because
rem --launch-count is shared by all kernels matching the filter.
setlocal enabledelayedexpansion
cd /d "%~dp0.."
set "NCU="
for /d %%d in ("%ProgramFiles%\NVIDIA Corporation\Nsight Compute*") do set "NCU=%%d\target\windows-desktop-win7-x64\ncu.exe"
if not exist "%NCU%" (
  echo Nsight Compute not found.
  exit /b 1
)
set "OUT=benchmarks\profiling"
set "N=1000000"
for %%k in (k_move_aos k_fused_aos k_move_soa k_compact_soa k_fused_soa k_fused_soa_active) do (
  echo === %%k ===
  "%NCU%" --kernel-name "regex:^%%k$" --launch-skip 1 --launch-count 1 --set full --import-source no ^
    --force-overwrite -o %OUT%\ncu_%%k build\bin\hospital_sim.exe --profile %N% --steps 3 || exit /b 1
)
rem ncu 2026.x writes compressed .ncu-repz; older versions write .ncu-rep.
> %OUT%\kernels_1M_details.csv (
  for %%f in (%OUT%\ncu_*.ncu-rep*) do (
    "%NCU%" --import "%%f" --csv --page details --section SpeedOfLight --section Occupancy ^
      --section LaunchStats --section WarpStateStats --section SourceCounters --section MemoryWorkloadAnalysis
  )
)
echo Reports: %OUT%\ncu_*.ncu-rep(z) (open in Nsight Compute UI)
echo Summary: %OUT%\kernels_1M_details.csv
endlocal

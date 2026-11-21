cmake_minimum_required(VERSION 3.24)
project(HospitalEmergencySim VERSION 1.0 LANGUAGES C CXX)

# ------------------------------------------------------------------ options ----
option(HS_ENABLE_CUDA "Build the CUDA simulation engine" ON)
option(HS_BUILD_GUI "Build the OpenGL / ImGui application" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(MSVC)
  # /fp:precise: no FMA contraction on the host -> bit-identical CPU/GPU arithmetic
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/W3> $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
                      $<$<COMPILE_LANGUAGE:C,CXX>:/fp:precise> $<$<COMPILE_LANGUAGE:C,CXX>:/MP>)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
endif()

# --------------------------------------------------------------------- CUDA ----
set(HS_HAVE_CUDA OFF)
if(HS_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    find_package(CUDAToolkit REQUIRED)
    set(HS_HAVE_CUDA ON)
    message(STATUS "CUDA ${CMAKE_CUDA_COMPILER_VERSION}, architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(WARNING "No CUDA compiler found: building the CPU implementation only. "
                    "GPU features will report 'unavailable' at runtime.")
  endif()
endif()

find_package(OpenMP)

# --------------------------------------------------------------- core library ----
add_library(hs_core STATIC
  src/Hospital.cpp
  src/Scenario.cpp
  src/PatientGenerator.cpp
  src/ResourceManager.cpp
  src/CpuBackend.cpp
  src/SimulationEngine.cpp
  src/Benchmark.cpp)
target_include_directories(hs_core PUBLIC include)
if(OpenMP_CXX_FOUND)
  target_link_libraries(hs_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(HS_HAVE_CUDA)
  add_library(hs_cuda STATIC
    cuda/MemoryManager.cu
    cuda/PatientKernels.cu
    cuda/CudaEngine.cu
    cuda/CudaBenchmark.cu)
  target_include_directories(hs_cuda PUBLIC include cuda)
  # --fmad=false: no fused multiply-add contraction -> bit-identical with the CPU baseline
  # -lineinfo   : source-level correlation in Nsight Compute / compute-sanitizer
  target_compile_options(hs_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -lineinfo --expt-relaxed-constexpr -Xcompiler=/utf-8>)
  target_link_libraries(hs_cuda PUBLIC CUDA::cudart_static)
  target_link_libraries(hs_core PUBLIC hs_cuda)
  target_compile_definitions(hs_core PUBLIC HS_HAVE_CUDA=1)
else()
  target_sources(hs_core PRIVATE src/CudaStub.cpp)
endif()

# -------------------------------------------------------------------- tests ----
enable_testing()
add_executable(hospital_tests tests/test_main.cpp)
target_link_libraries(hospital_tests PRIVATE hs_core)
add_test(NAME hospital_tests COMMAND hospital_tests)

# ---------------------------------------------------------------------- GUI ----
if(HS_BUILD_GUI)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
  add_subdirectory(third_party/glfw EXCLUDE_FROM_ALL)

  add_library(glad STATIC third_party/glad/src/gl.c)
  target_include_directories(glad PUBLIC third_party/glad/include)

  set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui)
  add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp ${IMGUI_DIR}/imgui_draw.cpp ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp)
  target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
  target_link_libraries(imgui PUBLIC glfw)

  add_executable(hospital_sim
    src/main.cpp
    src/App.cpp
    src/Renderer.cpp
    src/Camera.cpp
    src/Dashboard.cpp)
  target_include_directories(hospital_sim PRIVATE include third_party/glm)
  target_link_libraries(hospital_sim PRIVATE hs_core glad glfw imgui)
  if(WIN32)
    target_link_libraries(hospital_sim PRIVATE opengl32)
  endif()
  target_compile_definitions(hospital_sim PRIVATE HS_SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders")
  add_custom_command(TARGET hospital_sim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/shaders $<TARGET_FILE_DIR:hospital_sim>/shaders)
endif()

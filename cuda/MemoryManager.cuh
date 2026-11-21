// MemoryManager.cuh - CUDA error checking and RAII wrappers for device / pinned memory.
#pragma once
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "Patient.h"

namespace hs {
namespace gpu {

[[noreturn]] void throwCuda(cudaError_t e, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr)                                                         \
    do {                                                                         \
        cudaError_t _e = (expr);                                                 \
        if (_e != cudaSuccess) ::hs::gpu::throwCuda(_e, #expr, __FILE__, __LINE__); \
    } while (0)
// Check for launch errors right after a kernel launch.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

// Device memory buffer (global memory).
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { resize(n); }
    ~DeviceBuffer() { release(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void resize(size_t n) {
        if (n <= cap_) {
            size_ = n;
            return;
        }
        release();
        if (n > 0) CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
        cap_ = size_ = n;
    }
    void release() {
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        cap_ = size_ = 0;
    }
    T* get() const { return ptr_; }
    size_t size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }
    void zero() {
        if (ptr_) CUDA_CHECK(cudaMemset(ptr_, 0, bytes()));
    }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0, cap_ = 0;
};

// Page-locked host memory: required for truly asynchronous cudaMemcpyAsync and faster DMA.
template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    ~PinnedBuffer() { release(); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    void resize(size_t n) {
        if (n <= cap_) {
            size_ = n;
            return;
        }
        release();
        if (n > 0) CUDA_CHECK(cudaMallocHost(&ptr_, n * sizeof(T)));
        cap_ = size_ = n;
    }
    void release() {
        if (ptr_) cudaFreeHost(ptr_);
        ptr_ = nullptr;
        cap_ = size_ = 0;
    }
    T* get() const { return ptr_; }
    T& operator[](size_t i) { return ptr_[i]; }
    size_t size() const { return size_; }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0, cap_ = 0;
};

// Structure-of-Arrays patient storage in device global memory.
class DeviceSoA {
public:
    void allocate(int n);
    void release();
    ~DeviceSoA() { release(); }
    const PatientSoA& view() const { return v_; }
    int count() const { return n_; }
    // AoS host data <-> SoA device arrays (staged through pinned memory).
    void upload(const PatientAoS* host, int n, cudaStream_t s);
    void download(PatientAoS* host, int n, cudaStream_t s);  // synchronises the stream
    size_t bytes() const { return (size_t)n_ * sizeof(PatientAoS); }

private:
    PatientSoA v_;
    int n_ = 0;
    std::vector<void*> ptrs_;
    PinnedBuffer<uint8_t> staging_;
};

// Global switch: per-step CUDA event timing can be disabled to measure its own overhead.
inline bool g_eventTimingEnabled = true;

// Pair of CUDA events for timing a region on a stream.
class EventTimer {
public:
    EventTimer() {
        CUDA_CHECK(cudaEventCreate(&a_));
        CUDA_CHECK(cudaEventCreate(&b_));
    }
    ~EventTimer() {
        cudaEventDestroy(a_);
        cudaEventDestroy(b_);
    }
    void start(cudaStream_t s) {
        if (g_eventTimingEnabled) CUDA_CHECK(cudaEventRecord(a_, s));
    }
    void stop(cudaStream_t s) {
        if (g_eventTimingEnabled) CUDA_CHECK(cudaEventRecord(b_, s));
    }
    float ms() const {  // requires the stop event to have completed
        if (!g_eventTimingEnabled) return 0.f;
        float m = 0;
        CUDA_CHECK(cudaEventElapsedTime(&m, a_, b_));
        return m;
    }

private:
    cudaEvent_t a_{}, b_{};
};

}  // namespace gpu
}  // namespace hs

// MemoryManager.cu - device SoA allocation and AoS <-> SoA transfers.
#include <cstring>

#include "MemoryManager.cuh"

namespace hs {
namespace gpu {

void throwCuda(cudaError_t e, const char* expr, const char* file, int line) {
    throw std::runtime_error(std::string("CUDA error '") + cudaGetErrorString(e) + "' (" + cudaGetErrorName(e) +
                             ") in " + expr + " at " + file + ":" + std::to_string(line));
}

namespace {
constexpr int kFields = 13;
}

void DeviceSoA::allocate(int n) {
    release();
    n_ = n;
    if (n <= 0) return;
    // One allocation per field: each array is 256-byte aligned by cudaMalloc, so warp
    // accesses to consecutive patients map to fully coalesced 128-byte transactions.
    void** fields[kFields] = {(void**)&v_.x, (void**)&v_.z, (void**)&v_.tx, (void**)&v_.tz,
                              (void**)&v_.speed, (void**)&v_.waitingTime, (void**)&v_.serviceRemaining,
                              (void**)&v_.health, (void**)&v_.state, (void**)&v_.priority,
                              (void**)&v_.route, (void**)&v_.waypoint, (void**)&v_.flags};
    for (int f = 0; f < kFields; ++f) {
        CUDA_CHECK(cudaMalloc(fields[f], (size_t)n * 4));
        ptrs_.push_back(*fields[f]);
    }
    staging_.resize((size_t)n * 4 * kFields);
}

void DeviceSoA::release() {
    for (void* p : ptrs_) cudaFree(p);
    ptrs_.clear();
    v_ = PatientSoA{};
    n_ = 0;
}

void DeviceSoA::upload(const PatientAoS* h, int n, cudaStream_t s) {
    if (n != n_) throw std::runtime_error("DeviceSoA::upload size mismatch");
    if (n == 0) return;
    // Transpose AoS -> SoA in pinned staging memory, then one DMA per field.
    uint8_t* st = staging_.get();
    float* f[8];
    int32_t* iv[5];
    for (int k = 0; k < 8; ++k) f[k] = (float*)(st + (size_t)k * n * 4);
    for (int k = 0; k < 5; ++k) iv[k] = (int32_t*)(st + (size_t)(8 + k) * n * 4);
    for (int i = 0; i < n; ++i) {
        const PatientAoS& q = h[i];
        f[0][i] = q.x; f[1][i] = q.z; f[2][i] = q.tx; f[3][i] = q.tz; f[4][i] = q.speed;
        f[5][i] = q.waitingTime; f[6][i] = q.serviceRemaining; f[7][i] = q.health;
        iv[0][i] = q.state; iv[1][i] = q.priority; iv[2][i] = q.route; iv[3][i] = q.waypoint; iv[4][i] = q.flags;
    }
    for (int k = 0; k < kFields; ++k)
        CUDA_CHECK(cudaMemcpyAsync(ptrs_[k], st + (size_t)k * n * 4, (size_t)n * 4, cudaMemcpyHostToDevice, s));
}

void DeviceSoA::download(PatientAoS* h, int n, cudaStream_t s) {
    if (n != n_) throw std::runtime_error("DeviceSoA::download size mismatch");
    if (n == 0) return;
    uint8_t* st = staging_.get();
    for (int k = 0; k < kFields; ++k)
        CUDA_CHECK(cudaMemcpyAsync(st + (size_t)k * n * 4, ptrs_[k], (size_t)n * 4, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    const float* f[8];
    const int32_t* iv[5];
    for (int k = 0; k < 8; ++k) f[k] = (const float*)(st + (size_t)k * n * 4);
    for (int k = 0; k < 5; ++k) iv[k] = (const int32_t*)(st + (size_t)(8 + k) * n * 4);
    for (int i = 0; i < n; ++i) {
        PatientAoS& q = h[i];
        q.x = f[0][i]; q.z = f[1][i]; q.tx = f[2][i]; q.tz = f[3][i]; q.speed = f[4][i];
        q.waitingTime = f[5][i]; q.serviceRemaining = f[6][i]; q.health = f[7][i];
        q.state = iv[0][i]; q.priority = iv[1][i]; q.route = iv[2][i]; q.waypoint = iv[3][i]; q.flags = iv[4][i];
    }
}

}  // namespace gpu
}  // namespace hs

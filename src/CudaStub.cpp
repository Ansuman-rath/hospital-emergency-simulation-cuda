// CudaStub.cpp - compiled only when the project is configured WITHOUT CUDA.
// Every GPU entry point reports "unavailable"; nothing is emulated on the CPU.
#include "CudaEngine.h"

namespace hs {

bool cudaBuildEnabled() { return false; }
void setCudaEventTiming(bool) {}

GpuDeviceInfo queryGpu() {
    GpuDeviceInfo i;
    i.error = "built without CUDA support (HS_ENABLE_CUDA=OFF or no CUDA toolkit found)";
    return i;
}

const char* cudaVariantName(CudaVariant v) {
    (void)v;
    return "CUDA (unavailable)";
}

const char* gpuKernelVariantName(GpuKernelVariant) { return "CUDA (unavailable)"; }

std::unique_ptr<PatientBackend> createCudaBackend(CudaVariant, int, std::string* error) {
    if (error) *error = queryGpu().error;
    return nullptr;
}

GpuRunResult runGpuKernelBenchmark(const std::vector<PatientAoS>&, const std::vector<int32_t>&, const UpdateParams&,
                                   const RouteTable&, const GpuRunConfig&) {
    GpuRunResult r;
    r.ok = false;
    r.error = queryGpu().error;
    return r;
}

}  // namespace hs

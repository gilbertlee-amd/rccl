__global__ void PrimitiveReduceKernel(int const N,
                                      float const* __restrict__ src1,
                                      float const* __restrict__ src2,
                                      float* __restrict__ dst1)
{
    // Each threadblock works on its own chunk of the array
    // Keep each chunk a multiple of 128-bytes (32 floats)
    int const chunkSize = (((N + 31) / 32) + gridDim.x - 1) / gridDim.x * 32;
    int const chunkOffset = blockIdx.x * chunkSize;

    // Account for last chunk size
    int const actualSize = min(chunkSize, N - chunkOffset);

#define UNROLL 4
    ReduceOrCopy<UNROLL, FuncSum<float>, float, false, true>(
        threadIdx.x, blockDim.x,
        dst1 + blockIdx.x * chunkSize, nullptr,
        src1 + blockIdx.x * chunkSize,
        src2 + blockIdx.x * chunkSize,
        actualSize);
}

__global__ void PrimitiveReduceCopyKernel(int const N,
                                          float const* __restrict__ src1,
                                          float const* __restrict__ src2,
                                          float* __restrict__ dst1,
                                          float* __restrict__ dst2)
{
    // Each threadblock works on its own chunk of the array
    // Keep each chunk a multiple of 128-bytes (32 floats)
    int const chunkSize = (((N + 31) / 32) + gridDim.x - 1) / gridDim.x * 32;
    int const chunkOffset = blockIdx.x * chunkSize;

    // Account for last chunk size
    int const actualSize = min(chunkSize, N - chunkOffset);

#define UNROLL 4
    ReduceOrCopy<UNROLL, FuncSum<float>, float, true, true>(
        threadIdx.x, blockDim.x,
        dst1 + blockIdx.x * chunkSize,
        dst2 + blockIdx.x * chunkSize,
        src1 + blockIdx.x * chunkSize,
        src2 + blockIdx.x * chunkSize,
        actualSize);
}


void Reduce_Primitive(ConfigParams const& config,
                      GpuMem const& gpuMem,
                      int const numWorkgroups,
                      int const blockSize,
                      int const src1MemType,
                      int const src2MemType,
                      int const dst1MemType,
                      int const dst2MemType,
                      hipStream_t const stream)
{
    hipLaunchKernelGGL(PrimitiveReduceKernel,
                       numWorkgroups,
                       blockSize,
                       0,
                       stream,
                       gpuMem.N,
                       gpuMem.src1[GPU1][src1MemType],
                       gpuMem.src2[GPU1][src2MemType],
                       gpuMem.dst1[GPU2][dst1MemType]);
}

void ReduceCopy_Primitive(ConfigParams const& config,
                          GpuMem const& gpuMem,
                          int const numWorkgroups,
                          int const blockSize,
                          int const src1MemType,
                          int const src2MemType,
                          int const dst1MemType,
                          int const dst2MemType,
                          hipStream_t const stream)
{
    hipLaunchKernelGGL(PrimitiveReduceCopyKernel,
                       numWorkgroups,
                       blockSize,
                       0,
                       stream,
                       gpuMem.N,
                       gpuMem.src1[GPU1][src1MemType],
                       gpuMem.src2[GPU1][src2MemType],
                       gpuMem.dst1[GPU1][dst1MemType],
                       gpuMem.dst2[GPU2][dst2MemType]);
}

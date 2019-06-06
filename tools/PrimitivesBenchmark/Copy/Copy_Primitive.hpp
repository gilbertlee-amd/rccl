__global__ void PrimitiveCopyKernel(int const N,
                                    float const* __restrict__ src,
                                    float* __restrict__ dst)
{
    // Each threadblock works on its own chunk of the array
    // Keep each chunk a multiple of 128-bytes (32 floats)
    int const chunkSize = (((N + 31) / 32) + gridDim.x - 1) / gridDim.x * 32;
    int const chunkOffset = blockIdx.x * chunkSize;

    // Account for last chunk size
    int const actualSize = min(chunkSize, N - chunkOffset);

#define UNROLL 4
    ReduceOrCopy<UNROLL, FuncSum<float>, float, false, false>(
        threadIdx.x, blockDim.x,
        dst + blockIdx.x * chunkSize, nullptr,
        src + blockIdx.x * chunkSize, nullptr,
        actualSize);
}

__global__ void PrimitiveDoubleCopyKernel(int const N,
                                          float const* __restrict__ src,
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
    ReduceOrCopy<UNROLL, FuncSum<float>, float, true, false>(
        threadIdx.x, blockDim.x,
        dst1 + blockIdx.x * chunkSize,
        dst2 + blockIdx.x * chunkSize,
        src + blockIdx.x * chunkSize, nullptr,
        actualSize);
}


void RemoteCopy_Primitive(ConfigParams const& config,
                          GpuMem const& gpuMem,
                          int const numWorkgroups,
                          int const blockSize,
                          int const src1MemType,
                          int const src2MemType,
                          int const dst1MemType,
                          int const dst2MemType,
                          hipStream_t const stream)
{
    hipLaunchKernelGGL(PrimitiveCopyKernel,
                       numWorkgroups,
                       blockSize,
                       0,
                       stream,
                       gpuMem.N,
                       gpuMem.src1[GPU1][src1MemType],
                       gpuMem.dst1[GPU2][dst1MemType]);
}

void LocalCopy_Primitive(ConfigParams const& config,
                         GpuMem const& gpuMem,
                         int const numWorkgroups,
                         int const blockSize,
                         int const src1MemType,
                         int const src2MemType,
                         int const dst1MemType,
                         int const dst2MemType,
                         hipStream_t const stream)
{
    hipLaunchKernelGGL(PrimitiveCopyKernel,
                       numWorkgroups,
                       blockSize,
                       0,
                       stream,
                       gpuMem.N,
                       gpuMem.src1[GPU1][src1MemType],
                       gpuMem.dst1[GPU1][dst1MemType]);
}

void DoubleCopy_Primitive(ConfigParams const& config,
                          GpuMem const& gpuMem,
                          int const numWorkgroups,
                          int const blockSize,
                          int const src1MemType,
                          int const src2MemType,
                          int const dst1MemType,
                          int const dst2MemType,
                          hipStream_t const stream)
{
    hipLaunchKernelGGL(PrimitiveDoubleCopyKernel,
                       numWorkgroups,
                       blockSize,
                       0,
                       stream,
                       gpuMem.N,
                       gpuMem.src1[GPU1][src1MemType],
                       gpuMem.dst1[GPU1][dst1MemType],
                       gpuMem.dst2[GPU2][dst2MemType]);
}

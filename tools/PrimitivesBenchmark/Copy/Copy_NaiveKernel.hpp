__global__ void NaiveCopyKernel(int const N,
                                float const* __restrict__ src,
                                float* __restrict__ dst)
{
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < N;
         idx += blockDim.x * gridDim.x)
    {
        dst[idx] = src[idx];
    }
}

__global__ void NaiveDoubleCopyKernel(int const N,
                                      float const* __restrict__ src,
                                      float* __restrict__ dst1,
                                      float* __restrict__ dst2)
{
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < N;
         idx += blockDim.x * gridDim.x)
    {
        dst1[idx] = src[idx];
        dst2[idx] = src[idx];
    }
}

void RemoteCopy_NaiveKernel(ConfigParams const& config,
                            GpuMem const& gpuMem,
                            int const numWorkgroups,
                            int const blockSize,
                            int const src1MemType,
                            int const src2MemType,
                            int const dst1MemType,
                            int const dst2MemType,
                            hipStream_t const stream)
{
    hipLaunchKernelGGL(NaiveCopyKernel,
                       numWorkgroups,
                       blockSize,
                       0,
                       stream,
                       gpuMem.N,
                       gpuMem.src1[GPU1][src1MemType],
                       gpuMem.dst1[GPU2][dst1MemType]);
}

void LocalCopy_NaiveKernel(ConfigParams const& config,
                           GpuMem const& gpuMem,
                           int const numWorkgroups,
                           int const blockSize,
                           int const src1MemType,
                           int const src2MemType,
                           int const dst1MemType,
                           int const dst2MemType,
                           hipStream_t const stream)
{
    hipLaunchKernelGGL(NaiveCopyKernel,
                       numWorkgroups,
                       blockSize,
                       0,
                       stream,
                       gpuMem.N,
                       gpuMem.src1[GPU1][src1MemType],
                       gpuMem.dst1[GPU1][dst1MemType]);
}

void DoubleCopy_NaiveKernel(ConfigParams const& config,
                            GpuMem const& gpuMem,
                            int const numWorkgroups,
                            int const blockSize,
                            int const src1MemType,
                            int const src2MemType,
                            int const dst1MemType,
                            int const dst2MemType,
                            hipStream_t const stream)
{
    hipLaunchKernelGGL(NaiveDoubleCopyKernel,
                       numWorkgroups,
                       blockSize,
                       0,
                       stream,
                       gpuMem.N,
                       gpuMem.src1[GPU1][src1MemType],
                       gpuMem.dst1[GPU1][dst1MemType],
                       gpuMem.dst2[GPU2][dst2MemType]);
}

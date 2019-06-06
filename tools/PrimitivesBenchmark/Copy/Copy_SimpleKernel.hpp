__global__ void SimpleCopyKernel(int const N,
                                float const* __restrict__ src,
                                float* __restrict__ dst)
{
    float4* const __restrict__ src4 = (float4 *)src;
    float4*       __restrict__ dst4 = (float4 *)dst;

    int const offset = blockDim.x * gridDim.x;
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < N / 4;
         idx += offset)
    {
        dst4[idx] = src4[idx];
    }
}

__global__ void SimpleDoubleCopyKernel(int const N,
                                       float const* __restrict__ src,
                                       float* __restrict__ dst1,
                                       float* __restrict__ dst2)
{
    float4* const __restrict__ src4  = (float4 *)src;
    float4*       __restrict__ dst14 = (float4 *)dst1;
    float4*       __restrict__ dst24 = (float4 *)dst2;

    int const offset = blockDim.x * gridDim.x;
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < N / 4;
         idx += offset)
    {
        dst14[idx] = src4[idx];
        dst24[idx] = src4[idx];
    }
}

void RemoteCopy_SimpleKernel(ConfigParams const& config,
                             GpuMem const& gpuMem,
                             int const numWorkgroups,
                             int const blockSize,
                             int const src1MemType,
                             int const src2MemType,
                             int const dst1MemType,
                             int const dst2MemType,
                             hipStream_t const stream)
{
    hipLaunchKernelGGL(SimpleCopyKernel,
                       numWorkgroups,
                       blockSize,
                       0,
                       stream,
                       gpuMem.N,
                       gpuMem.src1[GPU1][src1MemType],
                       gpuMem.dst1[GPU2][dst1MemType]);
}

void LocalCopy_SimpleKernel(ConfigParams const& config,
                            GpuMem const& gpuMem,
                            int const numWorkgroups,
                            int const blockSize,
                            int const src1MemType,
                            int const src2MemType,
                            int const dst1MemType,
                            int const dst2MemType,
                            hipStream_t const stream)
{
    hipLaunchKernelGGL(SimpleCopyKernel,
                       numWorkgroups,
                       blockSize,
                       0,
                       stream,
                       gpuMem.N,
                       gpuMem.src1[GPU1][src1MemType],
                       gpuMem.dst1[GPU1][dst1MemType]);
}

void DoubleCopy_SimpleKernel(ConfigParams const& config,
                             GpuMem const& gpuMem,
                             int const numWorkgroups,
                             int const blockSize,
                             int const src1MemType,
                             int const src2MemType,
                             int const dst1MemType,
                             int const dst2MemType,
                             hipStream_t const stream)
{
    hipLaunchKernelGGL(SimpleDoubleCopyKernel,
                       numWorkgroups,
                       blockSize,
                       0,
                       stream,
                       gpuMem.N,
                       gpuMem.src1[GPU1][src1MemType],
                       gpuMem.dst1[GPU1][dst1MemType],
                       gpuMem.dst2[GPU2][dst2MemType]);
}

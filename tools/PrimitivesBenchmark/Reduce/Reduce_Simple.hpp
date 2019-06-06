__global__ void SimpleReduceKernel(int const N,
                                   float const* __restrict__ src1,
                                   float const* __restrict__ src2,
                                   float* __restrict__ dst1)
{
    float4* const __restrict__ input1  = (float4 *)src1;
    float4* const __restrict__ input2  = (float4 *)src2;
    float4*       __restrict__ output1 = (float4 *)dst1;

    int const offset = blockDim.x * gridDim.x;
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < N / 4;
         idx += offset)
    {
        output1[idx] = input1[idx] + input2[idx];
    }
}

__global__ void SimpleReduceCopyKernel(int const N,
                                       float const* __restrict__ src1,
                                       float const* __restrict__ src2,
                                       float* __restrict__ dst1,
                                       float* __restrict__ dst2)
{
    float4* const __restrict__ input1  = (float4 *)src1;
    float4* const __restrict__ input2  = (float4 *)src2;
    float4*       __restrict__ output1 = (float4 *)dst1;
    float4*       __restrict__ output2 = (float4 *)dst2;

    int const offset = blockDim.x * gridDim.x;
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < N / 4;
         idx += offset)
    {
        float4 sum = input1[idx] + input2[idx];
        output1[idx] = sum;
        output2[idx] = sum;
    }
}


void Reduce_Simple(ConfigParams const& config,
                   GpuMem const& gpuMem,
                   int const numWorkgroups,
                   int const blockSize,
                   int const src1MemType,
                   int const src2MemType,
                   int const dst1MemType,
                   int const dst2MemType,
                   hipStream_t const stream)
{
    hipLaunchKernelGGL(SimpleReduceKernel,
                       numWorkgroups,
                       blockSize,
                       0,
                       stream,
                       gpuMem.N,
                       gpuMem.src1[GPU1][src1MemType],
                       gpuMem.src2[GPU1][src2MemType],
                       gpuMem.dst1[GPU2][dst1MemType]);
}

void ReduceCopy_Simple(ConfigParams const& config,
                       GpuMem const& gpuMem,
                       int const numWorkgroups,
                       int const blockSize,
                       int const src1MemType,
                       int const src2MemType,
                       int const dst1MemType,
                       int const dst2MemType,
                       hipStream_t const stream)
{
    hipLaunchKernelGGL(SimpleReduceCopyKernel,
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

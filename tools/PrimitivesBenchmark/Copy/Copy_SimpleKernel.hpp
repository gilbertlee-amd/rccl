#ifndef COPYSIMPLEKERNEL_HPP
#define COPYSIMPLEKERNEL_HPP

__global__ void SimpleCopyKernel(int const N,
                                float const* __restrict__ src,
                                float* __restrict__ dst)
{
    int const offset = blockDim.x * gridDim.x;
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < N / 4;
         idx += offset)
    {
        float4* const __restrict__ src4 = (float4 *)src;
        float4*       __restrict__ dst4 = (float4 *)dst;

        dst4[idx] = src4[idx];
    }
}

void Copy_SimpleKernel(ConfigParams const& config,
                       GpuMem const& gpuMem,
                       int const numWorkgroups,
                       int const blockSize,
                       int const srcMemType,
                       int const dstMemType,
                       hipStream_t const stream)
{
    hipLaunchKernelGGL(SimpleCopyKernel,
                       numWorkgroups,
                       blockSize,
                       0,
                       stream,
                       gpuMem.N,
                       gpuMem.src[0][srcMemType],
                       gpuMem.dst[1][dstMemType]);
}

#endif

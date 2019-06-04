#ifndef COPYNAIVEKERNEL_HPP
#define COPYNAIVEKERNEL_HPP

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

void Copy_NaiveKernel(ConfigParams const& config,
                      GpuMem const& gpuMem,
                      int const numWorkgroups,
                      int const blockSize,
                      int const srcMemType,
                      int const dstMemType,
                      hipStream_t const stream)
{
    hipLaunchKernelGGL(NaiveCopyKernel,
                       numWorkgroups,
                       blockSize,
                       0,
                       stream,
                       gpuMem.N,
                       gpuMem.src[0][srcMemType],
                       gpuMem.dst[1][dstMemType]);
}

#endif

#ifndef PRIMITIVECOPYKERNEL_HPP
#define PRIMITIVECOPYKERNEL_HPP

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

void Copy_Primitive(ConfigParams const& config,
                    GpuMem const& gpuMem,
                    int const numWorkgroups,
                    int const blockSize,
                    int const srcMemType,
                    int const dstMemType,
                    hipStream_t const stream)
{
    hipLaunchKernelGGL(PrimitiveCopyKernel,
                       numWorkgroups,
                       blockSize,
                       0,
                       stream,
                       gpuMem.N,
                       gpuMem.src[0][srcMemType],
                       gpuMem.dst[1][dstMemType]);
}

#endif

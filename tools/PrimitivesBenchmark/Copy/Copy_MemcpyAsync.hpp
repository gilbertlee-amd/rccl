#ifndef COPYMEMCPYASYNC_HPP
#define COPYMEMCPYASYNC_HPP

void Copy_MemcpyAsync(ConfigParams const& config,
                      GpuMem const& gpuMem,
                      int const numWorkgroups,
                      int const blockSize,
                      int const srcMemType,
                      int const dstMemType,
                      hipStream_t const stream)
{
    HIP_CALL(hipMemcpyAsync(gpuMem.dst[1][dstMemType],
                            gpuMem.src[0][srcMemType],
                            gpuMem.numBytes,
                            hipMemcpyDeviceToDevice,
                            stream));
}

#endif

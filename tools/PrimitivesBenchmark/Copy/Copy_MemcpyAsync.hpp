void RemoteCopy_MemcpyAsync(ConfigParams const& config,
                            GpuMem const& gpuMem,
                            int const numWorkgroups,
                            int const blockSize,
                            int const src1MemType,
                            int const src2MemType,
                            int const dst1MemType,
                            int const dst2MemType,
                            hipStream_t const stream)
{
    HIP_CALL(hipMemcpyAsync(gpuMem.dst1[GPU2][dst1MemType],
                            gpuMem.src1[GPU1][src1MemType],
                            gpuMem.numBytes,
                            hipMemcpyDeviceToDevice,
                            stream));
}

void LocalCopy_MemcpyAsync(ConfigParams const& config,
                           GpuMem const& gpuMem,
                           int const numWorkgroups,
                           int const blockSize,
                           int const src1MemType,
                           int const src2MemType,
                           int const dst1MemType,
                           int const dst2MemType,
                           hipStream_t const stream)
{
    HIP_CALL(hipMemcpyAsync(gpuMem.dst1[GPU1][dst1MemType],
                            gpuMem.src1[GPU1][src1MemType],
                            gpuMem.numBytes,
                            hipMemcpyDeviceToDevice,
                            stream));
}

void DoubleCopy_MemcpyAsync(ConfigParams const& config,
                            GpuMem const& gpuMem,
                            int const numWorkgroups,
                            int const blockSize,
                            int const src1MemType,
                            int const src2MemType,
                            int const dst1MemType,
                            int const dst2MemType,
                            hipStream_t const stream)
{
    HIP_CALL(hipMemcpyAsync(gpuMem.dst1[GPU1][dst1MemType],
                            gpuMem.src1[GPU1][src1MemType],
                            gpuMem.numBytes,
                            hipMemcpyDeviceToDevice,
                            stream));
    HIP_CALL(hipMemcpyAsync(gpuMem.dst2[GPU2][dst2MemType],
                            gpuMem.src1[GPU1][src1MemType],
                            gpuMem.numBytes,
                            hipMemcpyDeviceToDevice,
                            stream));
}

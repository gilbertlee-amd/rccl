void Copy_Validate(GpuMem const& gpuMem,
                   float const* src,
                   float const* dst)
{
    float* src_Host = (float *)malloc(gpuMem.numBytes);
    float* dst_Host = (float *)malloc(gpuMem.numBytes);
    if (!src_Host || !dst_Host)
    {
        fprintf(stderr, "[ERROR] Unable to allocate temporary host buffers\n");
        exit(-1);
    }

    HIP_CALL(hipMemcpy(src_Host, src, gpuMem.numBytes, hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(dst_Host, dst, gpuMem.numBytes, hipMemcpyDeviceToHost));

    for (int i = 0; i < gpuMem.N; i++)
    {
        if (src_Host[i] != dst_Host[i])
        {
            fprintf(stderr, "[ERROR] Error in copy.  Element %d dst [%f] does not match src [%f]\n",
                    i, dst_Host[i], src_Host[i]);
            exit(-1);
        }
    }

    free(src_Host);
    free(dst_Host);
}

void RemoteCopy_Validate(GpuMem const& gpuMem,
                         int const src1MemType,
                         int const src2MemType,
                         int const dst1MemType,
                         int const dst2MemType)
{
    // RemoteCopy expects GPU2 dst1 to match GPU1 src1
    Copy_Validate(gpuMem, gpuMem.src1[GPU1][src1MemType], gpuMem.dst1[GPU2][dst1MemType]);
}

void LocalCopy_Validate(GpuMem const& gpuMem,
                        int const src1MemType,
                        int const src2MemType,
                        int const dst1MemType,
                        int const dst2MemType)
{
    // LocalCopy expects GPU1 dst1 to match GPU1 src1
    Copy_Validate(gpuMem, gpuMem.src1[GPU1][src1MemType], gpuMem.dst1[GPU1][dst1MemType]);
}

void DoubleCopy_Validate(GpuMem const& gpuMem,
                         int const src1MemType,
                         int const src2MemType,
                         int const dst1MemType,
                         int const dst2MemType)
{
    // DoubleCopy expects GPU src1 to be copied to [GPU1-dst1 and GPU2-dst2]
    Copy_Validate(gpuMem, gpuMem.src1[GPU1][src1MemType], gpuMem.dst1[GPU1][dst1MemType]);
    Copy_Validate(gpuMem, gpuMem.src1[GPU1][src1MemType], gpuMem.dst2[GPU2][dst2MemType]);
}

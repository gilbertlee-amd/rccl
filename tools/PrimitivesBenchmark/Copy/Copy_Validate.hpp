void Copy_Validate(GpuMem const& gpuMem,
                   int const srcMemType,
                   int const dstMemType)
{
    // Copy expects GPU2 dst to match GPU1 src
    float* src_Host = (float *)malloc(gpuMem.numBytes);
    float* dst_Host = (float *)malloc(gpuMem.numBytes);
    if (!src_Host || !dst_Host)
    {
        fprintf(stderr, "[ERROR] Unable to allocate temporary host buffers\n");
        exit(-1);
    }

    HIP_CALL(hipMemcpy(src_Host, gpuMem.src[0][srcMemType], gpuMem.numBytes, hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(dst_Host, gpuMem.dst[1][dstMemType], gpuMem.numBytes, hipMemcpyDeviceToHost));

    for (int i = 0; i < gpuMem.N; i++)
    {
        if (src_Host[i] != dst_Host[i])
        {
            fprintf(stderr, "[ERROR] Error in copy.  Element %d dst [%f] does not match src [%f]\n",
                    i, dst_Host[i], src_Host[i]);
            exit(-1);
        }
    }
}

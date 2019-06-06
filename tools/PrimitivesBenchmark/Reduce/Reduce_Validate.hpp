void Reduce_Validate(GpuMem const& gpuMem,
                     int const src1MemType,
                     int const src2MemType,
                     int const dst1MemType,
                     int const dst2MemType)
{
    float* src1_Host = (float *)malloc(gpuMem.numBytes);
    float* src2_Host = (float *)malloc(gpuMem.numBytes);
    float* dst1_Host = (float *)malloc(gpuMem.numBytes);
    if (!src1_Host || !src2_Host || !dst1_Host)
    {
        fprintf(stderr, "[ERROR] Unable to allocate temporary host buffers\n");
        exit(-1);
    }

    HIP_CALL(hipMemcpy(src1_Host, gpuMem.src1[GPU1][src1MemType], gpuMem.numBytes, hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(src2_Host, gpuMem.src2[GPU1][src2MemType], gpuMem.numBytes, hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(dst1_Host, gpuMem.dst1[GPU2][dst1MemType], gpuMem.numBytes, hipMemcpyDeviceToHost));

    for (int i = 0; i < gpuMem.N; i++)
    {
        float sum = src1_Host[i] + src2_Host[i];
        if (sum != dst1_Host[i])
        {
            fprintf(stderr, "[ERROR] Error in reduce.  Element %d dst [%f] does not match src sum [%f] %f + %f\n",
                    i, dst1_Host[i], sum, src1_Host[i], src2_Host[i]);
            exit(-1);
        }
    }

    free(src1_Host);
    free(src2_Host);
    free(dst1_Host);
}

void ReduceCopy_Validate(GpuMem const& gpuMem,
                         int const src1MemType,
                         int const src2MemType,
                         int const dst1MemType,
                         int const dst2MemType)
{
    float* src1_Host = (float *)malloc(gpuMem.numBytes);
    float* src2_Host = (float *)malloc(gpuMem.numBytes);
    float* dst1_Host = (float *)malloc(gpuMem.numBytes);
    float* dst2_Host = (float *)malloc(gpuMem.numBytes);
    if (!src1_Host || !src2_Host || !dst1_Host || !dst2_Host)
    {
        fprintf(stderr, "[ERROR] Unable to allocate temporary host buffers\n");
        exit(-1);
    }

    HIP_CALL(hipMemcpy(src1_Host, gpuMem.src1[GPU1][src1MemType], gpuMem.numBytes, hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(src2_Host, gpuMem.src2[GPU1][src2MemType], gpuMem.numBytes, hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(dst1_Host, gpuMem.dst1[GPU1][dst1MemType], gpuMem.numBytes, hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(dst2_Host, gpuMem.dst2[GPU2][dst2MemType], gpuMem.numBytes, hipMemcpyDeviceToHost));

    for (int i = 0; i < gpuMem.N; i++)
    {
        float sum = src1_Host[i] + src2_Host[i];
        if (sum != dst1_Host[i])
        {
            fprintf(stderr, "[ERROR] Error in reducecopy.  Element %d dst [%f] does not match src sum [%f] %f + %f\n",
                    i, dst1_Host[i], sum, src1_Host[i], src2_Host[i]);
            exit(-1);
        }
        if (sum != dst2_Host[i])
        {
            fprintf(stderr, "[ERROR] Error in reducecopy.  Element %d dst [%f] does not match src sum [%f] %f + %f\n",
                    i, dst2_Host[i], sum, src1_Host[i], src2_Host[i]);
            exit(-1);
        }
    }

    free(src1_Host);
    free(src2_Host);
    free(dst1_Host);
    free(dst2_Host);
}

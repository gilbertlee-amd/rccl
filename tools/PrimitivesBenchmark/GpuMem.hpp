#ifndef GPUMEM_HPP
#define GPUMEM_HPP

typedef enum
{
    MEM_COARSE     = 0, // Standard coarse-grained device memory
    MEM_FINE       = 1, // Fine-grained device memory (bypasses caching)
    NUM_MEM_TYPES  = 2
} MemTypeEnum;

char const memTypeNames[2][3] = {"CG", "FG"};

typedef struct
{
    int numBytes;                 // # of bytes per array
    int N;                        // Number of floats per array
    float* src[2][NUM_MEM_TYPES]; // Source buffers
    float* dst[2][NUM_MEM_TYPES]; // Destination buffers
} GpuMem;

// Fills or validates the contents of an array
void FillArrayWithPattern(int const N, float* array)
{
    float* hostArray = (float *) malloc(N * sizeof(float));
    if (!hostArray)
    {
        fprintf(stderr, "[ERROR] Unable to allocate host memory\n");
        exit(-1);
    }

    // Fill with pseudo-random pattern
    for (int i = 0; i < N; i++)
        hostArray[i] = static_cast<float>((i * 43) % 57);

    HIP_CALL(hipMemcpy(array, hostArray, N * sizeof(float), hipMemcpyHostToDevice));
    free(hostArray);
}

// Clears all destination buffers to 0
void ClearOutputs(ConfigParams const& config, GpuMem const& gpuMem)
{
    for (int i = 0; i < 2; i++)
    {
        HIP_CALL(hipSetDevice(config.deviceId[i]));
        HIP_CALL(hipMemset(gpuMem.dst[i][MEM_COARSE], 0, gpuMem.numBytes));
        HIP_CALL(hipMemset(gpuMem.dst[i][MEM_FINE], 0, gpuMem.numBytes));
    }
}

// Sets up peer-to-peer transfers, allocates memory, fills source data with values
void PrepareGpuMem(ConfigParams const& config, GpuMem& gpuMem, int const numBytes)
{
    gpuMem.numBytes = numBytes;
    gpuMem.N = numBytes / sizeof(float);

    for (int i = 0; i < 2; i++)
    {
        // Enable peer access between the two GPUs
        int canAccessPeer;
        HIP_CALL(hipDeviceCanAccessPeer(&canAccessPeer, config.deviceId[i], config.deviceId[1-i]));
        if (!canAccessPeer)
        {
            fprintf(stderr, "[ERROR] GPU %d cannot access GPU %d\n", config.deviceId[i], config.deviceId[1-i]);
            exit(-1);
        }
        HIP_CALL(hipSetDevice(config.deviceId[i]));
        hipError_t result = hipDeviceEnablePeerAccess(config.deviceId[1-i], 0);
        if (result != hipSuccess && result != hipErrorPeerAccessAlreadyEnabled)
        {
            fprintf(stderr, "[ERROR] Unable to enable peer access from GPU %d to %d\n",
                    config.deviceId[i], config.deviceId[1-i]);
            exit(-1);
        }

        // Allocate GPU resources (streams + memory buffers)
        HIP_CALL(hipMalloc((void**)&gpuMem.src[i][MEM_COARSE], numBytes));
        HIP_CALL(hipMalloc((void**)&gpuMem.dst[i][MEM_COARSE], numBytes));
        HIP_CALL(hipExtMallocWithFlags((void**)&gpuMem.src[i][MEM_FINE], numBytes,
                                       hipDeviceMallocFinegrained));
        HIP_CALL(hipExtMallocWithFlags((void**)&gpuMem.dst[i][MEM_FINE], numBytes,
                                       hipDeviceMallocFinegrained));

        // Fill the source data array with a pattern
        FillArrayWithPattern(gpuMem.N, gpuMem.src[i][MEM_COARSE]);
        FillArrayWithPattern(gpuMem.N, gpuMem.src[i][MEM_FINE]);
    }
}

// Free allocated device memory
void ReleaseGpuMem(GpuMem &gpuMem)
{
    for (int i = 0; i < 2; i++)
    {
        HIP_CALL(hipFree(gpuMem.src[i][MEM_COARSE]));
        HIP_CALL(hipFree(gpuMem.src[i][MEM_FINE]));
        HIP_CALL(hipFree(gpuMem.dst[i][MEM_COARSE]));
        HIP_CALL(hipFree(gpuMem.dst[i][MEM_FINE]));
    }
}

#endif

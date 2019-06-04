#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "../../src/collectives/device/primitives.h"

#include "PrimitivesBenchmark.hpp"
#include "LaunchCase.hpp"
#include "Config.hpp"
#include "GpuMem.hpp"

// Copy methods
#include "Copy/Copy_Validate.hpp"
#include "Copy/Copy_MemcpyAsync.hpp"
#include "Copy/Copy_NaiveKernel.hpp"
#include "Copy/Copy_SimpleKernel.hpp"
#include "Copy/Copy_Primitive.hpp"

// Double copy methods


int main(int argc, char** argv)
{
    // Currently an environment variable is required in order to enable fine-grained VRAM allocations
    if (!getenv("HSA_FORCE_FINE_GRAIN_PCIE"))
    {
        printf("[ERROR] Currently you must set HSA_FORCE_FINE_GRAIN_PCIE=1 prior to execution\n");
        exit(1);
    }

     // Parse command-line parameters
    ConfigParams config;
    ParseConfig(argc, argv, config);

    // Create new 'results.csv' file if requested
    if (config.outputToCsv) WriteCsvHeader();

    // Loop over bytes / # workgroups / blocksize
    for (auto numBytes : config.numBytes)
    {
        GpuMem gpuMem;
        PrepareGpuMem(config, gpuMem, numBytes);

        for (auto numWorkgroups : config.numWorkgroups)
        for (auto blockSize : config.blockSizes)
        {
            // Launch Copy methods
            LaunchCase("MemcpyAsync"  , config, gpuMem, Copy_MemcpyAsync , Copy_Validate, numWorkgroups, blockSize);
            LaunchCase("NaiveKernel"  , config, gpuMem, Copy_NaiveKernel , Copy_Validate, numWorkgroups, blockSize);
            LaunchCase("SimpleKernel" , config, gpuMem, Copy_SimpleKernel, Copy_Validate, numWorkgroups, blockSize);
            LaunchCase("CopyPrimitive", config, gpuMem, Copy_Primitive   , Copy_Validate, numWorkgroups, blockSize);
        }

        ReleaseGpuMem(gpuMem);
    }

    return 0;
}

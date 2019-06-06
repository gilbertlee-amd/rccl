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

// Copy methods (RemoteCopy / LocalCopy / DoubleCopy)
#include "Copy/Copy_Validate.hpp"
#include "Copy/Copy_MemcpyAsync.hpp"
#include "Copy/Copy_NaiveKernel.hpp"
#include "Copy/Copy_SimpleKernel.hpp"
#include "Copy/Copy_Primitive.hpp"

// Reduce methods (Reduce / ReduceCopy)
#include "Reduce/Reduce_Validate.hpp"
#include "Reduce/Reduce_Simple.hpp"
#include "Reduce/Reduce_Primitive.hpp"

int main(int argc, char** argv)
{
    // Currently an environment variable is required in order to enable fine-grained VRAM allocations
    if (!getenv("HSA_FORCE_FINE_GRAIN_PCIE"))
    {
        printf("[ERROR] Currently you must set HSA_FORCE_FINE_GRAIN_PCIE=1 prior to execution\n");
        exit(1);
    }

    // For more accurate comparisions, disable dedicated DMA copy engines and use shader blit kernels
    if (!getenv("HSA_ENABLE_SDMA"))
    {
        printf("[WARN] HSA_ENABLE_SDMA should be set to 0 to compare copy kernels vs DMA copy engines\n");
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
            // Launch RemoteCopy methods - Copies src1 array from GPU1 to dst1 array on GPU2
            for (int src1MemType = 0; src1MemType < NUM_MEM_TYPES; src1MemType++)
            for (int dst1MemType = 0; dst1MemType < NUM_MEM_TYPES; dst1MemType++)
            {
                LaunchCase("RemoteMemcpyAsync", config, gpuMem,
                           RemoteCopy_MemcpyAsync, RemoteCopy_Validate,
                           numWorkgroups, blockSize,
                           src1MemType, MEM_UNUSED,
                           dst1MemType, MEM_UNUSED);

                LaunchCase("RemoteSimpleKernel", config, gpuMem,
                           RemoteCopy_SimpleKernel, RemoteCopy_Validate,
                           numWorkgroups, blockSize,
                           src1MemType, MEM_UNUSED,
                           dst1MemType, MEM_UNUSED);

                LaunchCase("RemoteCopyPrimitive", config, gpuMem,
                           RemoteCopy_Primitive, RemoteCopy_Validate,
                           numWorkgroups, blockSize,
                           src1MemType, MEM_UNUSED,
                           dst1MemType, MEM_UNUSED);
            }

            // Launch Local Copy methods - Copies src1 array from GPU1 to dst1 array on GPU1
            for (int src1MemType = 0; src1MemType < NUM_MEM_TYPES; src1MemType++)
            for (int dst1MemType = 0; dst1MemType < NUM_MEM_TYPES; dst1MemType++)
            {
                LaunchCase("LocalMemcpyAsync", config, gpuMem,
                           LocalCopy_MemcpyAsync, LocalCopy_Validate,
                           numWorkgroups, blockSize,
                           src1MemType, MEM_UNUSED,
                           dst1MemType, MEM_UNUSED);

                LaunchCase("LocalSimpleKernel", config, gpuMem,
                           LocalCopy_SimpleKernel, LocalCopy_Validate,
                           numWorkgroups, blockSize,
                           src1MemType, MEM_UNUSED,
                           dst1MemType, MEM_UNUSED);

                LaunchCase("LocalCopyPrimitive", config, gpuMem,
                           LocalCopy_Primitive, LocalCopy_Validate,
                           numWorkgroups, blockSize,
                           src1MemType, MEM_UNUSED,
                           dst1MemType, MEM_UNUSED);
            }

            // Launch DoubleCopy methods - Copy src1 array from GPU1 to dst1 array on GPU1,
            //                             and fine-grained dst2 array on GPU2
            for (int src1MemType = 0; src1MemType < NUM_MEM_TYPES; src1MemType++)
            for (int dst1MemType = 0; dst1MemType < NUM_MEM_TYPES; dst1MemType++)
            {
                LaunchCase("DoubleMemcpyAsync", config, gpuMem,
                           DoubleCopy_MemcpyAsync, DoubleCopy_Validate,
                           numWorkgroups, blockSize,
                           src1MemType, MEM_UNUSED,
                           dst1MemType, MEM_FINE);

                LaunchCase("DoubleSimpleKernel", config, gpuMem,
                           DoubleCopy_SimpleKernel, DoubleCopy_Validate,
                           numWorkgroups, blockSize,
                           src1MemType, MEM_UNUSED,
                           dst1MemType, MEM_FINE);

                LaunchCase("DoubleCopyPrimitive", config, gpuMem,
                           DoubleCopy_Primitive, DoubleCopy_Validate,
                           numWorkgroups, blockSize,
                           src1MemType, MEM_UNUSED,
                           dst1MemType, MEM_FINE);
            }


            // Launch Reduce methods - GPU1 src1 + GPU1 src2 -> GPU2 dst1
            for (int src1MemType = 0; src1MemType < NUM_MEM_TYPES; src1MemType++)
            {
                LaunchCase("ReduceSimple", config, gpuMem,
                           Reduce_Simple, Reduce_Validate,
                           numWorkgroups, blockSize,
                           src1MemType, MEM_COARSE,
                           MEM_FINE, MEM_UNUSED);

                LaunchCase("ReducePrimitive", config, gpuMem,
                           Reduce_Primitive, Reduce_Validate,
                           numWorkgroups, blockSize,
                           src1MemType, MEM_COARSE,
                           MEM_FINE, MEM_UNUSED);
            }

            // Launch ReduceCopy methods - GPU1 src1 + GPU1 src2 -> GPU1 dst1 GPU2 dst2
            for (int src1MemType = 0; src1MemType < NUM_MEM_TYPES; src1MemType++)
            for (int dst1MemType = 0; dst1MemType < NUM_MEM_TYPES; dst1MemType++)
            {
                LaunchCase("ReduceCopySimple", config, gpuMem,
                           ReduceCopy_Simple, ReduceCopy_Validate,
                           numWorkgroups, blockSize,
                           src1MemType, MEM_COARSE,
                           dst1MemType, MEM_FINE);

                LaunchCase("ReduceCopyPrimitive", config, gpuMem,
                           ReduceCopy_Primitive, ReduceCopy_Validate,
                           numWorkgroups, blockSize,
                           src1MemType, MEM_COARSE,
                           dst1MemType, MEM_FINE);
            }
        }
        ReleaseGpuMem(gpuMem);
    }

    return 0;
}

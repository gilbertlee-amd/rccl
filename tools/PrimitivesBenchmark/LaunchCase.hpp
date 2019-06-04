#ifndef LAUNCHCASE_HPP
#define LAUNCHCASE_HPP

#include "Config.hpp"
#include "GpuMem.hpp"

// Test function interface
typedef void (TestFunc(ConfigParams const& config,
                       GpuMem const& gpuMem,
                       int const numWorkgroups,
                       int const blockSize,
                       int const srcMemType,
                       int const dstMemType,
                       hipStream_t const stream));

// Validation function interface
typedef void (ValidateFunc(GpuMem const& gpuMem,
                           int const srcMemType,
                           int const dstMemType));

void WriteCsvHeader()
{
    FILE* fp = fopen("results.csv", "w");
    if (!fp)
    {
        fprintf(stderr, "[ERROR] Unable to create results.csv file\n");
        exit(-1);

    }
    fprintf(fp, "MethodName,");
    fprintf(fp, "SrcMemType,");
    fprintf(fp, "DstMemType,");
    fprintf(fp, "TotalBytes,");
    fprintf(fp, "AvgTime,");
    fprintf(fp, "GB/sec,");
    fprintf(fp, "Workgroups,");
    fprintf(fp, "Blocksize,");
    fprintf(fp, "\n");
    fclose(fp);
}

// Test case launcher
void LaunchCase(std::string const& name,
                ConfigParams const& config,
                GpuMem const& gpuMem,
                TestFunc testFunc,
                ValidateFunc validateFunc,
                int const numWorkgroups,
                int const blockSize)
{
    FILE *fp;
    if (config.outputToCsv)
    {
        fp = fopen("results.csv", "a");
        if (!fp)
        {
            fprintf(stderr, "[ERROR] Unable to append to results.csv file\n");
            exit(-1);
        }
    }

    // Conversion ratios
    float const BYTES_TO_GBYTES = (1e-9);
    float const MSEC_TO_SEC     = (1e-3);

    float minTime, avgTime, maxTime, elapsedTimeMs;

    // Create HIP events / streams
    hipEvent_t eventStart, eventStop;
    HIP_CALL(hipEventCreateWithFlags(&eventStart, hipEventBlockingSync));
    HIP_CALL(hipEventCreateWithFlags(&eventStop, hipEventBlockingSync));
    hipStream_t stream;
    HIP_CALL(hipStreamCreate(&stream));

    // Loop over all possible memory type combinations
    for (int srcMemType = 0; srcMemType < NUM_MEM_TYPES; srcMemType++)
    for (int dstMemType = 0; dstMemType < NUM_MEM_TYPES; dstMemType++)
    {
        for (int iter = -config.numWarmup; iter < config.numIterations; iter++)
        {
            ClearOutputs(config, gpuMem);

            HIP_CALL(hipSetDevice(config.deviceId[0]));
            HIP_CALL(hipEventRecord(eventStart, stream));

            // Call provided test function
            testFunc(config, gpuMem, numWorkgroups, blockSize,
                     srcMemType, dstMemType, stream);

            HIP_CALL(hipEventRecord(eventStop, stream));
            HIP_CALL(hipEventSynchronize(eventStop));

            if (iter >= 0)
            {
                HIP_CALL(hipEventElapsedTime(&elapsedTimeMs, eventStart, eventStop));

                if (iter == 0)
                {
                    minTime = maxTime = avgTime = elapsedTimeMs;
                    if (config.verbose)
                    {
                        printf("Testing %s [%s -> %s]\n", name.c_str(),
                               memTypeNames[srcMemType], memTypeNames[dstMemType]);
                    }
                }
                else
                {
                    minTime = std::min(minTime, elapsedTimeMs);
                    maxTime = std::max(maxTime, elapsedTimeMs);
                    avgTime += elapsedTimeMs;
                }

                if (config.verbose)
                {
                    printf("  Iteration %d: Elapsed Time: %.3f ms (%.3f GB/s)\n", iter, elapsedTimeMs,
                           gpuMem.numBytes * BYTES_TO_GBYTES / (elapsedTimeMs * MSEC_TO_SEC));
                }
            }
        }
        avgTime /= config.numIterations;
        float const avgBandwidth = (gpuMem.numBytes * BYTES_TO_GBYTES / (avgTime * MSEC_TO_SEC));

        // Validate results (only validate last iteration for performance)
        validateFunc(gpuMem, srcMemType, dstMemType);

        printf("[%s->%s]: %-15s Min: %7.3f ms Avg: %7.3f ms Max: %7.3f ms (avg %7.3f GB/s)\n",
               memTypeNames[srcMemType], memTypeNames[dstMemType],
               name.c_str(), minTime, avgTime, maxTime, avgBandwidth);

        if (config.outputToCsv)
        {
            fprintf(fp, "%s,", name.c_str());
            fprintf(fp, "%s,", memTypeNames[srcMemType]);
            fprintf(fp, "%s,", memTypeNames[dstMemType]);
            fprintf(fp, "%d,", gpuMem.numBytes);
            fprintf(fp, "%.3f,", avgTime);
            fprintf(fp, "%.3f,", avgBandwidth);
            fprintf(fp, "%d,", numWorkgroups);
            fprintf(fp, "%d,", blockSize);
            fprintf(fp, "\n");
        }

    }

    if (config.outputToCsv) fclose(fp);
}

#endif

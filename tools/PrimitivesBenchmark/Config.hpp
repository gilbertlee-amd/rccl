#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <sstream>

// Configuration parameters
typedef struct
{
    int  deviceId[2];                // GPUs to target
    int  numWarmup;                  // # of (untimed) warmup iterations
    int  numIterations;              // # of timed iterations
    bool verbose;                    // Display extra timing information
    bool outputToCsv;                // Outputs results to CSV file
    std::vector<int> numBytes;       // # of bytes to operate on
    std::vector<int> numWorkgroups;  // # of workgroups to use
    std::vector<int> blockSizes;     // # of threads to use per block
} ConfigParams;

// Returns the next token following 'option'
char* GetCmdOption(char** begin, char** end, std::string const& option)
{
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) return *itr;
    return 0;
}

// Returns true if 'option' exists
bool CmdOptionExists(char** begin, char** end, std::string const& option)
{
    return std::find(begin, end, option) != end;
}

// Converts a string of comma-separated ints to a vector of ints
void ParseVectorInt(std::string str, std::vector<int>& array)
{
    std::stringstream ss(str);
    array.clear();
    int val;
    while (ss >> val)
    {
        array.push_back(val);
        if (ss.peek() == ',') ss.ignore();
    }
}

// Function to parse command line arguments
void ParseConfig(int const argc, char **argv, ConfigParams& config)
{
    // Ensure that at least 2 GPUs are found
    int numDevices;
    HIP_CALL(hipGetDeviceCount(&numDevices));
    if (numDevices < 2)
    {
        fprintf(stderr, "[ERROR] %s requires at least 2 GPUs\n", argv[0]);
        exit(-1);
    }
    printf("Detected %d GPUs\n", numDevices);

    // Display help
    if (CmdOptionExists(argv, argv + argc, "-h")) {
        printf("%s <flag1> <value1> flagN> <valueN>\n", argv[0]);
        printf("===============================================================\n");
        printf("Available flags:                                [Default Value]\n");
        printf(" -gpu1 <value>     : Set 1st GPU device to use to <value>     0\n");
        printf(" -gpu2 <value>     : Set 2nd GPU device to use to <value>     1\n");
        printf(" -h                : Display this help\n");
        printf(" -n    <v1,v2,...> : # of bytes to operate on              2^24\n");
        printf(" -wg   <v1,v2,...> : # of workgroups to execute with n)      32\n");
        printf(" -bs   <v1,v2,...> : Blocksizes                             256\n");
        printf(" -nw   <value>     : # of warmup iterations                   1\n");
        printf(" -ni   <value>     : # of iterations                          5\n");
        printf(" -v                : Print each iteration timing               \n");
        printf(" -csv              : Outputs results to CSV file 'results.csv' \n");
        exit(0);
    }

    // Set defaults
    config.deviceId[0] = 0;
    config.deviceId[1] = 1;
    config.numBytes.clear();
    config.numBytes.push_back(1<<24);
    config.numWorkgroups.clear();
    config.numWorkgroups.push_back(32);
    config.blockSizes.clear();
    config.blockSizes.push_back(256);
    config.numWarmup = 1;
    config.numIterations = 5;
    config.verbose = false;
    config.outputToCsv = false;

    // Parse target GPUs
    char* arg;
    arg = GetCmdOption(argv, argv + argc, "-gpu1");
    if (arg) config.deviceId[0] = atoi(arg);
    arg = GetCmdOption(argv, argv + argc, "-gpu2");
    if (arg) config.deviceId[1] = atoi(arg);

    if (config.deviceId[0] == config.deviceId[1])
    {
        fprintf(stderr, "[ERROR] Must use two different GPUs\n");
        exit(-1);
    }

    arg = GetCmdOption(argv, argv + argc, "-n");
    if (arg) ParseVectorInt(arg, config.numBytes);
    for (int bytes : config.numBytes)
    {
        if (bytes % (4 * sizeof(float)))
        {
            fprintf(stderr, "[ERROR] The number of bytes must be a multiple of %lu\n", 4 * sizeof(float));
            exit(-1);
        }
    }

    arg = GetCmdOption(argv, argv + argc, "-wg");
    if (arg) ParseVectorInt(arg, config.numWorkgroups);

    arg = GetCmdOption(argv, argv + argc ,"-bs");
    if (arg) ParseVectorInt(arg, config.blockSizes);

    arg = GetCmdOption(argv, argv + argc ,"-nw");
    if (arg) config.numWarmup = atoi(arg);

    arg = GetCmdOption(argv, argv + argc ,"-ni");
    if (arg) config.numIterations = atoi(arg);

    config.verbose = CmdOptionExists(argv, argv + argc, "-v");
    config.outputToCsv = CmdOptionExists(argv, argv + argc, "-csv");

    // Print Config:
    printf("Testing GPU %d / GPU %d\n", config.deviceId[0], config.deviceId[1]);
    printf("Testing bytes:");
    for (auto bytes: config.numBytes) printf(" %d", bytes);  printf("\n");
    printf("Testing workgroups:");
    for (auto wg: config.numWorkgroups) printf(" %d", wg); printf("\n");
    printf("Testing blocksizes:");
    for (auto bs: config.blockSizes) printf(" %d", bs); printf("\n");
    printf("# Warmup iterations %d   # iterations: %d\n", config.numWarmup, config.numIterations);
    printf("Verbose output: %s\n", config.verbose ? "YES" : "NO");
    printf("Output to CSV: %s\n", config.outputToCsv ? "YES" : "NO");
}

#endif

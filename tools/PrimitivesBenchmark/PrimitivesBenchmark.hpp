#ifndef PRIMITIVESBENCHMARK_HPP
#define PRIMITIVESBENCHMARK_HPP

#define HIP_CALL(cmd)                                               \
    do {                                                            \
        hipError_t error = (cmd);                                   \
        if (error != hipSuccess)                                    \
        {                                                           \
            fprintf(stderr, "[ERROR] %s at line %d in file %s\n",   \
                    hipGetErrorString(error), __LINE__, __FILE__);  \
            exit(-1);                                               \
        }                                                           \
    } while (0)


#endif

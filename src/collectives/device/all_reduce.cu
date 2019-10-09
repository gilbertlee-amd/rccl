/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "all_reduce.h"
#include "collectives.h"

IMPL_COLL2(ncclAllReduce, prod, FuncProd, ncclCollAllReduce, ncclProd);
IMPL_COLL2(ncclAllReduce, min,  FuncMin,  ncclCollAllReduce, ncclMin);
IMPL_COLL2(ncclAllReduce, max,  FuncMax,  ncclCollAllReduce, ncclMax);

//IMPL_COLL2(ncclAllReduce, sum,  FuncSum,  ncclCollAllReduce, ncclSum);
IMPL_COLL3(ncclAllReduce, sum, FuncSum, i8,  int8_t,   ncclCollAllReduce, ncclSum, ncclInt8)
IMPL_COLL3(ncclAllReduce, sum, FuncSum, u8,  uint8_t,  ncclCollAllReduce, ncclSum, ncclUint8)
IMPL_COLL3(ncclAllReduce, sum, FuncSum, i32, int32_t,  ncclCollAllReduce, ncclSum, ncclInt32)
IMPL_COLL3(ncclAllReduce, sum, FuncSum, u32, uint32_t, ncclCollAllReduce, ncclSum, ncclUint32)
IMPL_COLL3(ncclAllReduce, sum, FuncSum, i64, int64_t,  ncclCollAllReduce, ncclSum, ncclInt64)
IMPL_COLL3(ncclAllReduce, sum, FuncSum, u64, uint64_t, ncclCollAllReduce, ncclSum, ncclUint64)
IMPL_COLL3(ncclAllReduce, sum, FuncSum, f16, half,     ncclCollAllReduce, ncclSum, ncclFloat16)
IMPL_COLL3(ncclAllReduce, sum, FuncSum, f64, double,   ncclCollAllReduce, ncclSum, ncclFloat64)

//IMPL_COLL3(ncclAllReduce, sum, FuncSum, f32, float,    ncclCollAllReduce, ncclSum, ncclFloat32)
IMPL_COLL_FUNC(ncclAllReduceRing, sum, FuncSum, f32, float);
IMPL_COLL_FUNC(ncclAllReduceRingLL, sum, FuncSum, f32, float);

// Focusing on ncclAllReduceRingLLKernel_sum_f32
#ifdef USE_ORIG
IMPL_COLL_KERN_sum(ncclAllReduceRingLL, sum, FuncSum, f32, float, \
                   FUNC_INDEX(ncclCollAllReduce, ncclSum, ncclFloat32, 1, 0))
#else

#include "ncclAllReduceRingLLKernel_sum_f32.h"

#endif

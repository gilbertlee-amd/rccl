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

IMPL_COLL_KERN_sum(ncclAllReduceRingLL, sum, FuncSum, f32, float,	\
		   FUNC_INDEX(ncclCollAllReduce, ncclSum, ncclFloat32, 1, 0))
#else

// Mod 01: We never launch with extra thread, so launch bound can be restricted to MAXTHREADS
//__launch_bounds__(MAXTHREADS+WARP_SIZE, 1)
__launch_bounds__(MAXTHREADS, 1)				
__global__ void ncclAllReduceRingLLKernel_sum_f32(struct ncclColl firstColl) {
  int tid = threadIdx.x; 
  int bid = blockIdx.x;

  __shared__ struct ncclColl localColl; 
  __shared__ uint32_t abortCount; 
  if (tid == 0) abortCount = 0;
  __syncthreads(); 

  // Each kernel launches with one threadblock per 'channel'
  // Each channel/threadblock is filled with a queue of collectives to complete
  // The first collective argument is passed in 
  struct ncclDevComm* comm = firstColl.args.comm; 
  struct ncclChannel* channel = comm->channels+bid; 
  struct ncclColl* c; 
  channel->abortCount = &abortCount; 

  if (bid == 0) { 
    /* To optimize for latency, (only) the first operation is passed as argument.*/ 
    c = &firstColl; 
  } else { 
    c = &localColl;

    struct ncclColl* hostColl = channel->devCollectives[channel->collFifoHead];
    int* d = (int*)c;
    int* s = (int*)hostColl;
    exitIfAbortBarrier(0, &abortCount);
    //for (int o = tid; o < (sizeof(struct ncclColl)/sizeof(int)); o += blockDim.x) d[o] = s[o];
    if (tid < sizeof(struct ncclColl)/sizeof(int)) d[tid] = s[tid];
    __syncthreads();
 
    if (tid == 0) hostColl->active = 0;
  }
  
  while (1) { 
    if (tid < c->args.nThreads) { 
      if (c->funcIndex == FUNC_INDEX(ncclCollAllReduce, ncclSum, ncclFloat32, 1, 0)) { 
        ncclAllReduceRingLLKernel<COLL_UNROLL, FuncSum<float>, float>(&c->args);
      } else if (c->funcIndex == FUNC_INDEX(ncclCollAllReduce, ncclSum, ncclFloat32, 0, 0)) {
	//	ncclAllReduceRingKernel<COLL_UNROLL, FuncSum<float>, float>(&c->args);
      } else {
	//NCCL_CALL_FUNCTIONS(c); 
      } 
      
    } 
    int nextIndex = c->nextIndex; 
    if (tid == 0) channel->collFifoHead = nextIndex; 
 
    if (c->active == 2) { 
      return; 
    } 
 
    /* Load next collective operation*/ 
    c = &localColl; /* for bid 0 */ 
    load_coll(c, channel->devCollectives+nextIndex, tid, &abortCount); 
  }
}
#endif








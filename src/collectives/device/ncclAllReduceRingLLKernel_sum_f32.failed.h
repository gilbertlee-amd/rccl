#define RECVOFFSET(i) ((recvStep[i]%NCCL_STEPS)*stepSize)
#define SENDOFFSET(i) ((sendStep[i]%NCCL_STEPS)*stepSize)
#define RECVPTR(i)    (((const T*)recvBuff[i])+RECVOFFSET(i))
#define SENDPTR(i)    (((T*)sendBuff[i])+SENDOFFSET(i))

#define DIRECTRECVPTR(DIRECTRECV, i, directOffset)                      \
  (DIRECTRECV && recvDirectBuff[i] ? recvDirectBuff[i]+directOffset : RECVPTR(i))

#define DIRECTSENDPTR(DIRECTSEND, i, directOffset) \
  (DIRECTSEND && sendDirectBuff[i] ? sendDirectBuff[i]+directOffset : SENDPTR(i))

#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
#define BARRIER() __syncthreads()
#else
#define BARRIER() asm volatile ("bar.sync 1, %0;" :: "r"(nthreads))
#endif

#define WAITSEND(i)                                                     \
  {                                                                     \
    spins = 0;                                                          \
    mismatch = 0;                                                       \
    sendStep[i] += SLICESTEPS;                                          \
    if (tid == WARP_SIZE+i) {                                           \
      while (sendConnHead[i] + NCCL_STEPS < sendStep[i]) {              \
        sendConnHead[i] = LOAD(waitPtr);                                \
        /*if (checkAbort(sendConn[i]->opCountRem)) break;*/             \
        volatile uint64_t* remoteOpCount = sendConn[i]->opCountRem;     \
        spins++;                                                        \
        if (spins == SPINS_BEFORE_CHECK_ABORT) {                        \
          abort = LOAD(comm->abortFlag);                                \
          /*checkMismatch(remoteOpCount);*/                             \
          if (mismatch) {                                               \
            /* In non-LL, we use _threadfence_system before incrementing opCount, yet we are still waiting for credits here, so there must be a size mismatch*/ \
            STORE(comm->fatalDevError, ncclDevAssertedMismatch);        \
          } else if (remoteOpCount && LOAD(remoteOpCount) > opCount) {  \
            mismatch += 1;                                              \
          }                                                             \
          spins = 0;                                                    \
        }                                                               \
        if (abort) break;                                               \
      }                                                                 \
    }                                                                   \
  }

#define WAITRECV(i)                                                     \
  {                                                                     \
    spins = 0;                                                          \
    mismatch = 0;                                                       \
    recvStep[i] += SLICESTEPS;                                          \
    if (tid == i) {                                                     \
      while (LOAD(waitPtr) < recvStep[i]) {                             \
        /*if (checkAbort(recvConn[i]->opCountRem)) break;*/             \
        volatile uint64_t* remoteOpCount = recvConn[i]->opCountRem;     \
        spins++;                                                        \
        if (spins == SPINS_BEFORE_CHECK_ABORT) {                        \
          abort = LOAD(comm->abortFlag);                                \
          /*checkMismatch(remoteOpCount);*/                             \
          if (mismatch) {                                               \
            /* In non-LL, we use _threadfence_system before incrementing opCount, yet we are still waiting for credits here, so there must be a size mismatch*/ \
            STORE(comm->fatalDevError, ncclDevAssertedMismatch);        \
          } else if (remoteOpCount && LOAD(remoteOpCount) > opCount) {  \
            mismatch += 1;                                              \
          }                                                             \
          spins = 0;                                                    \
        }                                                               \
        if (abort) break;                                               \
      }                                                                 \
    }                                                                   \
  }

#define POSTSENDSIZE(i, size) \
  if (sendConn[i]->fifo) STORE(sendConn[i]->fifo+((sendStep[i]-SLICESTEPS)%NCCL_STEPS), size)

#define POSTSEND(i)                                                     \
  {                                                                     \
    if (sendConn[i]->next_hdp_reg) STORE(sendConn[i]->next_hdp_reg, 0x1); \
    STORE(sendConn[i]->tail, sendStep[i]);                              \
  }

#define POSTRECV(i)                             \
  {                                             \
    STORE(recvConn[i]->head, recvStep[i]);      \
  }

#define GENERICOP(DIRECTRECV,                                           \
                  DIRECTSEND,                                           \
                  RECV,                                                 \
                  SEND,                                                 \
                  SRC,                                                  \
                  DST,                                                  \
                  srcPtr,                                               \
                  dstPtr,                                               \
                  nelem,                                                \
                  directOffset)                                         \
  {                                                                     \
    int offset = 0;                                                     \
    int sliceSize = stepSize * SLICESTEPS;                              \
                                                                        \
    const T* srcs[RECV*NRECV+SRC];                                      \
    /*srcs[0] = SRC ? srcPtr : directRecvPtr<DIRECTRECV>(0, directOffset);*/ \
    srcs[0] = SRC ? srcPtr : DIRECTRECVPTR(DIRECTRECV, 0, directOffset); \
    if (RECV) {                                                         \
      /*if (SRC) srcs[1] = recvPtr(0);*/                                \
      if (SRC) srcs[1] = RECVPTR(0);                                    \
      /*for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = recvPtr(i);*/ \
      for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = RECVPTR(i);  \
    }                                                                   \
                                                                        \
    T* dsts[SEND*NSEND+DST];                                            \
    /*dsts[0] = DST ? dstPtr : directSendPtr<DIRECTSEND>(0, directOffset);*/ \
    dsts[0] = DST ? dstPtr : DIRECTSENDPTR(DIRECTSEND, 0, directOffset); \
    if (SEND) {                                                         \
      /*if (DST) dsts[1] = directSendPtr<DIRECTSEND>(0, directOffset);*/ \
      if (DST) dsts[1] = DIRECTSENDPTR(DIRECTSEND, 0, directOffset);    \
      /*for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = directSendPtr<DIRECTSEND>(i, directOffset);*/ \
      for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = DIRECTSENDPTR(DIRECTSEND, i, directOffset); \
    }                                                                   \
                                                                        \
    _Pragma("unroll 1")                                                 \
      for (int slice=0; slice<SLICESPERCHUNK; ++slice) {                \
        int realSize = max(0, min(sliceSize, nelem-offset));            \
        /*FOR_SEND(waitSend);*/                                         \
        do {                                                            \
          if (SEND) {                                                   \
            /* Send to far first, then close */                         \
            /* for (int i=1; i<NSEND && i<nsend; i++) func(i, ##__VA_ARGS__);*/ \
            for (int i=1; i<NSEND && i<nsend; i++) WAITSEND(i);         \
            /* func(0, ##__VA_ARGS__); */                               \
            WAITSEND(0);                                                \
          }                                                             \
        } while (0);                                                    \
        /*FOR_RECV(waitRecv);*/                                         \
        do {                                                            \
          if (RECV) {                                                   \
            /* Recv from close first, then far */                       \
            /*func(0, ##__VA_ARGS__);*/                                 \
            WAITRECV(0);                                                \
            /*for (int i=1; i<NRECV && i<nrecv; i++) func(i, ##__VA_ARGS__);*/ \
            for (int i=1; i<NRECV && i<nrecv; i++) WAITRECV(i);         \
          }                                                             \
        } while (0);                                                    \
        if (realSize > 0) {                                             \
          /*barrier();*/                                                \
          BARRIER();                                                    \
          if (DIRECTRECV && recvDirectBuff[0]) {                        \
            /* We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy */ \
            if (SEND) {                                                 \
              ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, NSEND>(tid, nthreads, 1, srcs, nsend, dsts+1, realSize); \
            }                                                           \
          } else {                                                      \
            ReduceOrCopyMulti<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST>(tid, nthreads, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize); \
          }                                                             \
        }                                                               \
        exitIfAbortBarrier(abort, abortCount);                          \
        if (tid == 0) /*FOR_SEND(postSendSize, realSize*sizeof(T));*/   \
        {                                                               \
          do {                                                          \
            if (SEND) {                                                 \
              /* Send to far first, then close */                       \
              /*for (int i=1; i<NSEND && i<nsend; i++) func(i, ##__VA_ARGS__);*/ \
              for (int i=1; i<NSEND && i<nsend; i++) POSTSENDSIZE(i, realSize*sizeof(T)); \
              /*func(0, ##__VA_ARGS__);*/                               \
              POSTSENDSIZE(0, realSize*sizeof(T));                      \
            }                                                           \
          } while (0);                                                  \
        }                                                               \
        if (SEND) __threadfence_system();                               \
        if (tid == 0)/* FOR_SEND(postSend);*/                           \
        {                                                               \
          do {                                                          \
            if (SEND) {                                                 \
              /* Send to far first, then close */                       \
              /*for (int i=1; i<NSEND && i<nsend; i++) func(i, ##__VA_ARGS__);*/ \
              for (int i=1; i<NSEND && i<nsend; i++) POSTSEND(i);       \
              /*func(0, ##__VA_ARGS__);*/                               \
              POSTSEND(0);                                              \
            }                                                           \
          } while (0);                                                  \
        }                                                               \
        if (tid == 0)/*FOR_RECV(postRecv);*/                            \
        {                                                               \
          do {                                                          \
            if (RECV) {                                                 \
              /* Recv from close first, then far */                     \
              /*func(0, ##__VA_ARGS__);*/                               \
              POSTRECV(0);                                              \
              /*for (int i=1; i<NRECV && i<nrecv; i++) func(i, ##__VA_ARGS__);*/ \
              for (int i=1; i<NRECV && i<nrecv; i++) POSTRECV(i);       \
            }                                                           \
          } while (0);                                                  \
        }                                                               \
      }                                                                 \
    for (int i=0; i<RECV*NRECV+SRC; i++) srcs[i] += sliceSize;          \
    for (int i=0; i<SEND*NSEND+DST; i++) dsts[i] += sliceSize;          \
    offset += sliceSize;                                                \
  }

template<int UNROLL, class FUNC, typename T>
__attribute__((noinline))
__device__ void ncclAllReduceRingKernel2(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const ssize_t size = args->N;
  const int nranks = comm->nRanks;
  const int stepSize = channel->buffSize / (sizeof(T)*NCCL_STEPS);
  const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const ssize_t loopSize = args->nChannels*(ssize_t)chunkSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

/*
  ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, FUNC>
    prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm, args->opCount);
  */

  // ncclPrimitive defines to replace template arguments
  #define SLICESPERCHUNK   (ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS)
  #define SLICESTEPS       ALLREDUCE_SLICESTEPS
  #define NRECV            1
  #define NSEND            1

  // ncclPrimitives private variables
  //const int tid;        => tid
  //const int nthreads;   => nthreads
  int nrecv = 0;
  int nsend = 0;
  //const int stepSize;
  struct ncclConnInfo* recvConn[NRECV];
  struct ncclConnInfo* sendConn[NSEND];
  volatile uint64_t* waitPtr;
  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  uint64_t sendConnHead[NSEND];
  const T* recvDirectBuff[NRECV];
  T* sendDirectBuff[NSEND];
  const T* recvBuff[NRECV];
  T* sendBuff[NSEND];
  //struct ncclDevComm* comm;
  uint32_t* abortCount;
  uint32_t mismatch = 0;
  //const uint64_t opCount; => args->opCount
  #define opCount args->opCount

  uint32_t spins = 0;
  uint32_t abort = 0;

  {
    // Arguments to ncclPrimitives constructor
    int* recvPeers = &ring->prev;
    int* sendPeers = &ring->next;

    // Make sure step is updated before we read it
    abortCount = channel->abortCount;
    __syncthreads();

    // disable directBuff
    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++)
    {
      //loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i, 0);
      {
        // loadRecvConn arguments
        struct ncclConnInfo* conn = &channel->devPeers[recvPeers[i]].recv.conn;
        T* directBuff = 0;

        recvConn[i] = conn;
        recvBuff[i] = (const T*)LOAD(&recvConn[i]->buff);
        recvStep[i] = LOAD(&recvConn[i]->step);
        recvStep[i] = ROUNDUP(recvStep[i], SLICESPERCHUNK*SLICESTEPS);
        // Return credits in case we rounded up.
        if (tid == 0) STORE(recvConn[i]->head, recvStep[i]);
        if (tid == i) {
          waitPtr = LOAD(&recvConn[i]->tail);
          STORE(recvConn[i]->opCountLoc, opCount);
        }
        recvDirectBuff[i] = NULL;
        if (directBuff && recvConn[i]->direct) {
          recvDirectBuff[i] = directBuff;
          if (tid == 0) STORE(recvConn[i]->ptrExchange, directBuff);
        }
        nrecv++;
      }
    }

    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++)
    {
      //loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i, 0);
      {
        struct ncclConnInfo* conn = &channel->devPeers[sendPeers[i]].send.conn;
        T* directBuff = 0;

        sendConn[i] = conn;
        sendBuff[i] = (T*)LOAD(&sendConn[i]->buff);
        sendStep[i] = LOAD(&sendConn[i]->step);
        sendStep[i] = ROUNDUP(sendStep[i], SLICESPERCHUNK*SLICESTEPS);
        if (tid == WARP_SIZE+i) {
          waitPtr = LOAD(&sendConn[i]->head);
          sendConnHead[i] = LOAD(waitPtr);
          STORE(sendConn[i]->opCountLoc, opCount);
        }
        sendDirectBuff[i] = NULL;
        if (directBuff && sendConn[i]->direct) {
          void* volatile* ptr = sendConn[i]->ptrExchange;
          while ((sendDirectBuff[i] = (T*)(LOAD(ptr))) == NULL);
          __syncthreads();
          if (tid == 0) STORE(ptr, NULL);
        }
        nsend++;
      }
    }
  }

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
    int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*args->nChannels));
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int slice;

    // step 0: push data to next GPU
    slice = ring->devUserRanks[nranks-1];
    offset = chunkOffset + slice * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    INIT_COUNTER;
    //prims.send(thisInput+offset, nelem);
    {
      const T* src = thisInput+offset;
      //GenericOp<0, 0, 0, 1, 1, 0>(src, NULL, nelem, 0);
      GENERICOP(0, 0, 0, 1, 1, 0, src, NULL, nelem, 0);
    }
    ACCUMULATE_COUNTER(send);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      INIT_COUNTER;
      //prims.recvReduceSend(thisInput+offset, nelem);
      {
        const T* src = thisInput+offset;
        //GenericOp<0, 0, 1, 1, 1, 0>(src, NULL, nelem, 0);
        GENERICOP(0, 0, 1, 1, 1, 0, src, NULL, nelem, 0);
      }
      ACCUMULATE_COUNTER(recvReduceSend);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    slice = ring->devUserRanks[0];
    offset = chunkOffset + slice * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    INIT_COUNTER;
    //prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);
    {
      const T* src = thisInput+offset;
      T* dst = thisOutput+offset;
      int directOffset = offset;

      //GenericOp<0, 1, 1, 1, 1, 1>(src, dst, nelem, directOffset);
      GENERICOP(0, 1, 1, 1, 1, 1, src, dst, nelem, directOffset);
    }
    ACCUMULATE_COUNTER(directRecvReduceCopySend);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      INIT_COUNTER;
      //prims.directRecvCopySend(thisOutput+offset, offset, nelem);
      {
        T* dst = thisOutput+offset;
        int directOffset = offset;
        //GenericOp<1, 1, 1, 1, 0, 1>(NULL, dst, nelem, directOffset);
        GENERICOP(1, 1, 1, 1, 0, 1, NULL, dst, nelem, directOffset);
      }
      ACCUMULATE_COUNTER(directRecvCopySend);
    }

    // Make final copy from buffer to dest.
    slice = ring->devUserRanks[1];
    offset = chunkOffset + slice * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    // Final wait/copy.
    INIT_COUNTER;
    //prims.directRecv(thisOutput+offset, offset, nelem);
    {
      T* dst = thisOutput+offset;
      int directOffset = offset;

      //GenericOp<1, 0, 1, 0, 0, 1>(NULL, dst, nelem, directOffset);
      GENERICOP(1, 0, 1, 0, 0, 1, NULL, dst, nelem, directOffset);
    }
    ACCUMULATE_COUNTER(directRecv);
  }

  //__device__ ~ncclPrimitives()
  {
    for (int i=0; i<NRECV && i<nrecv; i++)
    {
      //saveRecvConn(i);
      {
        if (tid == i) {
          STORE(&recvConn[i]->step, recvStep[i]);
          __threadfence_system();
          __atomic_fetch_add(recvConn[i]->opCountLoc, 1, __ATOMIC_SEQ_CST);
        }
      }
    }
    for (int i=0; i<NSEND && i<nsend; i++)
    {
      //saveSendConn(i);
      {
        if (tid == WARP_SIZE+i) {
          STORE(&sendConn[i]->step, sendStep[i]);
          __threadfence_system();
          __atomic_fetch_add(sendConn[i]->opCountLoc, 1, __ATOMIC_SEQ_CST);
        }
      }
    }
  }
}

// Mod 01: We never launch with extra thread, so launch bound can be restricted to MAXTHREADS
// 02: Remove unnecessary initial exitIfAbortBarrierCheck from first ncclColl load
// 03: Delay the __syncthreads after initializing abortCount to after ncclColl is loaded in parallel
//     to avoid redundant __syncthread for all blocks aside from the first one
// 04: Explictly instantiate the first ncclColl load, and remove the for loop loading because
//     the size of ncclColl is known.  Add static assert to check that sizeof(ncclColl)  is 64 bytes

__launch_bounds__(MAXTHREADS, 1)
__global__ void ncclAllReduceRingLLKernel_sum_f32(struct ncclColl firstColl) {

  int tid = threadIdx.x;
  int bid = blockIdx.x;

  __shared__ struct ncclColl localColl;
  __shared__ uint32_t abortCount;

  if (tid == 0) abortCount = 0;

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

    {
      struct ncclColl* hostColl = channel->devCollectives + channel->collFifoHead;
      int* d = (int*)c;
      int* s = (int*)hostColl;
      if (tid < sizeof(struct ncclColl)/sizeof(int)) d[tid] = s[tid];
      __syncthreads();

      if (tid == 0) hostColl->active = 0;
    }
  }
  __syncthreads();

  while (1) {
    if (tid < c->args.nThreads) {
      ncclAllReduceRingKernel2<COLL_UNROLL, FuncSum<float>, float>(&c->args);

      /*
      if (c->funcIndex == FUNC_INDEX(ncclCollAllReduce, ncclSum, ncclFloat32, 1, 0)) {
        ncclAllReduceRingLLKernel<COLL_UNROLL, FuncSum<float>, float>(&c->args);
      } else if (c->funcIndex == FUNC_INDEX(ncclCollAllReduce, ncclSum, ncclFloat32, 0, 0)) {
      ncclAllReduceRingKernel<COLL_UNROLL, FuncSum<float>, float>(&c->args);
      } else {
        NCCL_CALL_FUNCTIONS(c);
      }
      */
    }
    uint16_t nextIndex = c->nextIndex;
    if (tid == 0) channel->collFifoHead = nextIndex;

    if (c->active == 2) {
      return;
    }

    /* Load next collective operation*/
    c = &localColl; /* for bid 0 */
    load_coll(c, channel->devCollectives+nextIndex, tid, &abortCount);
    {
      struct ncclColl* hostColl = channel->devCollectives + nextIndex;
      int* d = (int*)c;
      int* s = (int*)hostColl;
      if (tid < sizeof(struct ncclColl)/sizeof(int)) d[tid] = s[tid];
      __syncthreads();

      if (tid == 0) hostColl->active = 0;
    }
  }
}

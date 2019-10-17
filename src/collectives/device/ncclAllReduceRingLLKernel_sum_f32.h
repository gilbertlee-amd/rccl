#ifdef ENABLE_PROFILING
#define INIT_COUNTER \
  if (tid==0) { t0 = clock64(); ws = LOAD(&(devProf->wait_send_cycle[blockIdx.x])); \
                wr = LOAD(&(devProf->wait_recv_cycle[blockIdx.x])); }

#define ACCUMULATE_COUNTER(prim) \
  if (tid==0) { __atomic_fetch_add(&(devProf->prim##_cycle), clock64() - t0 \
                                   + ws - LOAD(&(devProf->wait_send_cycle[blockIdx.x])) \
                                   + wr - LOAD(&(devProf->wait_recv_cycle[blockIdx.x])), \
                                   __ATOMIC_SEQ_CST);                   \
    __atomic_fetch_add(&(devProf->prim##_byte), nelem * sizeof(T), __ATOMIC_SEQ_CST); }

#else
#define INIT_COUNTER
#define ACCUMULATE_COUNTER(prim)
#endif


#define LOADRECVCONN(conn, i, directBuff)                           \
  {                                                                 \
    recvConn[i] = conn;                                             \
    recvBuff[i] = (const T*)LOAD(&recvConn[i]->buff);               \
    recvStep[i] = LOAD(&recvConn[i]->step);                         \
    recvStep[i] = ROUNDUP(recvStep[i], SLICESPERCHUNK*SLICESTEPS);  \
    /* Return credits in case we rounded up. */                     \
    if (tid == 0) STORE(recvConn[i]->head, recvStep[i]);            \
    if (tid == i) {                                                 \
      waitPtr = LOAD(&recvConn[i]->tail);                           \
      STORE(recvConn[i]->opCountLoc, opCount);                      \
    }                                                               \
    recvDirectBuff[i] = NULL;                                       \
    if (directBuff && recvConn[i]->direct) {                        \
      recvDirectBuff[i] = directBuff;                               \
      if (tid == 0) STORE(recvConn[i]->ptrExchange, directBuff);    \
    }                                                               \
    nrecv++;                                                        \
  }

#define LOADSENDCONN(conn, i, directBuff)                           \
  {                                                                 \
    sendConn[i] = conn;                                             \
    sendBuff[i] = (T*)LOAD(&sendConn[i]->buff);                     \
    sendStep[i] = LOAD(&sendConn[i]->step);                         \
    sendStep[i] = ROUNDUP(sendStep[i], SLICESPERCHUNK*SLICESTEPS);  \
    if (tid == WARP_SIZE+i) {                                       \
      waitPtr = LOAD(&sendConn[i]->head);                           \
      sendConnHead[i] = LOAD(waitPtr);                              \
      STORE(sendConn[i]->opCountLoc, opCount);                      \
    }                                                               \
    sendDirectBuff[i] = NULL;                                       \
    if (directBuff && sendConn[i]->direct) {                        \
      void* volatile* ptr = sendConn[i]->ptrExchange;               \
      while ((sendDirectBuff[i] = (T*)(LOAD(ptr))) == NULL);        \
      __syncthreads();                                              \
      if (tid == 0) STORE(ptr, NULL);                               \
    }                                                               \
    nsend++;                                                        \
  }

#define SAVERECVCONN(i)                                                 \
  {                                                                     \
    if (tid == i) {                                                     \
      STORE(&recvConn[i]->step, recvStep[i]);                           \
      __threadfence_system();                                           \
      __atomic_fetch_add(recvConn[i]->opCountLoc, 1, __ATOMIC_SEQ_CST); \
    }                                                                   \
  }                                                                     \

#define SAVESENDCONN(i)                                                 \
  {                                                                     \
    if (tid == WARP_SIZE+i) {                                           \
      STORE(&sendConn[i]->step, sendStep[i]);                           \
      __threadfence_system();                                           \
      __atomic_fetch_add(sendConn[i]->opCountLoc, 1, __ATOMIC_SEQ_CST); \
    }                                                                   \
  }

#define RECVOFFSET(i) ((recvStep[i]%NCCL_STEPS)*stepSize)
#define SENDOFFSET(i) ((sendStep[i]%NCCL_STEPS)*stepSize)
#define RECVPTR(i)    (((const T*)recvBuff[i])+RECVOFFSET(i))
#define SENDPTR(i)    (((      T*)sendBuff[i])+SENDOFFSET(i))

#define DIRECTRECVPTR(DIRECTRECV, i, directOffset)                      \
  (DIRECTRECV && recvDirectBuff[i] ? recvDirectBuff[i]+directOffset : RECVPTR(i))

#define DIRECTSENDPTR(DIRECTSEND, i, directOffset) \
  (DIRECTSEND && sendDirectBuff[i] ? sendDirectBuff[i]+directOffset : SENDPTR(i))


#define CHECKMISMATCH(var)                                              \
  {                                                                     \
    volatile uint64_t* remoteOpCount = var;                             \
    if (mismatch) {                                                     \
      /* In non-LL, we use _threadfence_system before incrementing opCount, yet we are still waiting for credits here, so there must be a size mismatch */ \
      STORE(comm->fatalDevError, ncclDevAssertedMismatch);              \
    } else if (remoteOpCount && LOAD(remoteOpCount) > opCount) {        \
      mismatch += 1;                                                    \
    }                                                                   \
  }

#define CHECKABORT(remoteOpCount)               \
  {                                             \
    spins++;                                    \
    if (spins == SPINS_BEFORE_CHECK_ABORT) {    \
      abort = LOAD(comm->abortFlag);            \
      CHECKMISMATCH(remoteOpCount);             \
      spins = 0;                                \
    }                                           \
  }

#ifdef ENABLE_PROFILING

#define WAITSEND(i)                                                     \
  {                                                                     \
    spins = 0;                                                          \
    mismatch = 0;                                                       \
    sendStep[i] += SLICESTEPS;                                          \
    if (tid == WARP_SIZE+i) {                                           \
      auto devProf = comm->devProf;                                     \
      uint64_t t0 = clock64();                                          \
      while (sendConnHead[i] + NCCL_STEPS < sendStep[i]) {              \
        sendConnHead[i] = LOAD(waitPtr);                                \
        CHECKABORT(sendConn[i]->opCountRem);                            \
        if (abort) break;                                               \
      }                                                                 \
      __atomic_fetch_add(&devProf->wait_send_cycle[blockIdx.x], clock64() - t0, __ATOMIC_SEQ_CST); \
    }                                                                   \
  }

#define WAITRECV(i)                                                     \
  {                                                                     \
    spins = 0;                                                          \
    mismatch = 0;                                                       \
    recvStep[i] += SLICESTEPS;                                          \
    if (tid == i) {                                                     \
      auto devProf = comm->devProf;                                     \
      uint64_t t0 = clock64();                                          \
      while (LOAD(waitPtr) < recvStep[i]) {                             \
        CHECKABORT(recvConn[i]->opCountRem);                            \
        if (abort) break;                                               \
      }                                                                 \
      __atomic_fetch_add(&devProf->wait_recv_cycle[blockIdx.x], clock64() - t0, __ATOMIC_SEQ_CST); \
    }                                                                   \
  }


#else

#define WAITSEND(i)                                                     \
  {                                                                     \
    spins = 0;                                                          \
    mismatch = 0;                                                       \
    sendStep[i] += SLICESTEPS;                                          \
    if (tid == WARP_SIZE+i) {                                           \
      while (sendConnHead[i] + NCCL_STEPS < sendStep[i]) {              \
        sendConnHead[i] = LOAD(waitPtr);                                \
        CHECKABORT(sendConn[i]->opCountRem);                            \
        if (abort) break;                                               \
      }                                                                 \
    }                                                                   \
  }

#define WAITRECV(i)                                       \
  {                                                       \
    spins = 0;                                            \
    mismatch = 0;                                         \
    recvStep[i] += SLICESTEPS;                            \
    if (tid == i) {                                       \
      while (LOAD(waitPtr) < recvStep[i]) {               \
        CHECKABORT(recvConn[i]->opCountRem);              \
        if (abort) break;                                 \
      }                                                   \
    }                                                     \
  }
#endif

#define POSTSENDSIZE(i, size)                                           \
  {                                                                     \
    if (sendConn[i]->fifo) STORE(sendConn[i]->fifo+((sendStep[i]-SLICESTEPS)%NCCL_STEPS), size); \
  }

#define POSTSEND(i)                                                     \
  {                                                                     \
    if (sendConn[i]->next_hdp_reg) STORE(sendConn[i]->next_hdp_reg, 0x1); \
    STORE(sendConn[i]->tail, sendStep[i]);                              \
  }

#define POSTRECV(i)                             \
  {                                             \
    STORE(recvConn[i]->head, recvStep[i]);      \
  }

#define GENERICOP(DIRECTRECV, DIRECTSEND, RECV, SEND, SRC, DST, srcPtr, dstPtr, nelem, directOffset) \
  {                                                                     \
    int offset = 0;                                                     \
    int sliceSize = stepSize * SLICESTEPS;                              \
                                                                        \
    const T* srcs[RECV*NRECV+SRC];                                      \
    srcs[0] = SRC ? srcPtr : DIRECTRECVPTR(DIRECTRECV, 0, directOffset); \
    if (RECV) {                                                         \
      if (SRC) srcs[1] = RECVPTR(0);                                    \
      for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = RECVPTR(i);  \
    }                                                                   \
                                                                        \
    T* dsts[SEND*NSEND+DST];                                            \
    dsts[0] = DST ? dstPtr : DIRECTSENDPTR(DIRECTSEND, 0, directOffset); \
    if (SEND) {                                                         \
      if (DST) dsts[1] = DIRECTSENDPTR(DIRECTSEND, 0, directOffset);   \
      for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = DIRECTSENDPTR(DIRECTSEND, i, directOffset); \
    }                                                                   \
                                                                        \
    /*    _Pragma("unroll 1")                                             */ \
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {                  \
      int realSize = max(0, min(sliceSize, nelem-offset));              \
      /*FOR_SEND(waitSend);*/                                           \
      do {                                                              \
        if (SEND) {                                                     \
          /* Send to far first, then close */                           \
          for (int i=1; i<NSEND && i<nsend; i++) WAITSEND(i);           \
          WAITSEND(0);                                                  \
        }                                                               \
      } while (0);                                                      \
      /*FOR_RECV(waitRecv);*/                                           \
      do {                                                              \
        if (RECV) {                                                     \
          /* Recv from close first, then far */                         \
          WAITRECV(0);                                                  \
          for (int i=1; i<NRECV && i<nrecv; i++) WAITRECV(i);           \
        }                                                               \
      } while (0);                                                      \
      if (realSize > 0) {                                               \
        /*barrier();*/                                                  \
        __syncthreads();                                                \
        if (DIRECTRECV && recvDirectBuff[0]) {                          \
          /* We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy */ \
          if (SEND) {                                                   \
            ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, NSEND>(tid, nthreads, 1, srcs, nsend, dsts+1, realSize); \
          }                                                             \
        } else {                                                        \
          ReduceOrCopyMulti<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST>(tid, nthreads, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize); \
        }                                                               \
      }                                                                 \
      if (abort) __atomic_fetch_add(abortCount, 1, __ATOMIC_SEQ_CST);   \
      __syncthreads();                                                  \
      if (LOAD(abortCount)) {                                           \
        asm volatile ("s_endpgm");                                      \
        /*return; */                                                    \
      }                                                                 \
      /*exitIfAbortBarrier(abort, abortCount);*/                        \
      if (tid == 0) {                                                   \
        /*FOR_SEND(postSendSize, realSize*sizeof(T));*/                 \
        do {                                                            \
          if (SEND) {                                                   \
            /* Send to far first, then close */                         \
            for (int i=1; i<NSEND && i<nsend; i++) POSTSENDSIZE(i, realSize*sizeof(T)); \
            POSTSENDSIZE(0, realSize*sizeof(T));                        \
          }                                                             \
        } while (0);                                                    \
      }                                                                 \
      if (SEND) __threadfence_system();                                 \
      if (tid == 0) {                                                   \
        /* FOR_SEND(postSend);*/                                        \
        do {                                                            \
          if (SEND) {                                                   \
            /* Send to far first, then close */                         \
            for (int i=1; i<NSEND && i<nsend; i++) POSTSEND(i);         \
            POSTSEND(0);                                                \
          }                                                             \
        } while (0);                                                    \
      }                                                                 \
      if (tid == 0) {                                                   \
        /*FOR_RECV(postRecv);*/                                         \
        do {                                                            \
          if (RECV) {                                                   \
            /* Recv from close first, then far */                       \
            POSTRECV(0);                                                \
            for (int i=1; i<NRECV && i<nrecv; i++) POSTRECV(i);         \
          }                                                             \
        } while (0);                                                    \
      }                                                                 \
      for (int i=0; i<RECV*NRECV+SRC; i++) srcs[i] += sliceSize;        \
      for (int i=0; i<SEND*NSEND+DST; i++) dsts[i] += sliceSize;        \
      offset += sliceSize;                                              \
    }                                                                   \
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

#ifdef ENABLE_PROFILING
  /*
  auto devProf = comm->devProf;
  uint64_t clk, t0 = 0ULL, ws, wr;
  if (tid == 0) clk = clock64();
  */
#endif

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

//#define USE_PRIMITIVE 1
  int flag = -1;

#if USE_PRIMITIVE
  ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, FUNC>
    prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm, args->opCount);
#else

  #define SLICESPERCHUNK (ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS)
  #define SLICESTEPS     (ALLREDUCE_SLICESTEPS)
  #define NRECV          (1)
  #define NSEND          (1)

  int nrecv = 0;
  int nsend = 0;
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
  uint32_t* abortCount;
  const uint64_t opCount = (args->opCount);
  uint32_t mismatch = 0;
  uint32_t spins = 0;
  uint32_t abort = 0;

  {
    int* recvPeers = &ring->prev;
    int* sendPeers = &ring->next;
    // Make sure step is updated before we read it
    abortCount = channel->abortCount;
    __syncthreads();

    // disable directBuff
    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++)
    {
      LOADRECVCONN(&channel->devPeers[recvPeers[i]].recv.conn, i, 0);
    }
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++)
    {
      LOADSENDCONN(&channel->devPeers[sendPeers[i]].send.conn, i, 0);
    }
  }
#endif


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
#if USE_PRIMITIVE
    prims.send(thisInput+offset, nelem);
#else
    {
      const T* src = (thisInput+offset);
    //GenericOp<0, 0, 0, 1, 1, 0>(src, NULL, nelem, 0);
      GENERICOP(0, 0, 0, 1, 1, 0, src, NULL, nelem, 0);
    }
#endif
    ACCUMULATE_COUNTER(send);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      INIT_COUNTER;
#if USE_PRIMITIVE
      prims.recvReduceSend(thisInput+offset, nelem);
#else
      {
        const T* src = (thisInput+offset);
//      GenericOp<0, 0, 1, 1, 1, 0>(src, NULL, nelem, 0);
        GENERICOP(0, 0, 1, 1, 1, 0, src, NULL, nelem, 0);
      }
#endif

      ACCUMULATE_COUNTER(recvReduceSend);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    slice = ring->devUserRanks[0];
    offset = chunkOffset + slice * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    INIT_COUNTER;
#if USE_PRIMITIVE
    prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);
#else
    {
      const T* src = (thisInput+offset);
      T* dst = (thisOutput+offset);
      int directOffset = offset;
//    GenericOp<0, 1, 1, 1, 1, 1>(src, dst, nelem, directOffset);
      GENERICOP(0, 1, 1, 1, 1, 1, src, dst, nelem, directOffset);
    }
#endif

    ACCUMULATE_COUNTER(directRecvReduceCopySend);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      INIT_COUNTER;
#if USE_PRIMITIVE
      prims.directRecvCopySend(thisOutput+offset, offset, nelem);
#else
      {
        T* dst = (thisOutput+offset);
        int directOffset = offset;
//      GenericOp<1, 1, 1, 1, 0, 1>(NULL, dst, nelem, directOffset);
        GENERICOP(1, 1, 1, 1, 0, 1, NULL, dst, nelem, directOffset);
      }
#endif
      ACCUMULATE_COUNTER(directRecvCopySend);
    }

    // Make final copy from buffer to dest.
    slice = ring->devUserRanks[1];
    offset = chunkOffset + slice * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    // Final wait/copy.
    INIT_COUNTER;
#if USE_PRIMITIVE
    prims.directRecv(thisOutput+offset, offset, nelem);
#else
    {
      T* dst = (thisOutput+offset);
      int directOffset = offset;
//    GenericOp<1, 0, 1, 0, 0, 1>(NULL, dst, nelem, directOffset);
      GENERICOP(1, 0, 1, 0, 0, 1, NULL, dst, nelem, directOffset);
    }
#endif
    ACCUMULATE_COUNTER(directRecv);
  }

#if USE_PRIMITIVE

#else
  //__device__ ~ncclPrimitives2() {
  {
    // Save steps for next collective. Have thread 0 do it to be compatible
    // with the way LL works.
    for (int i=0; i<NRECV && i<nrecv; i++)
    {
      SAVERECVCONN(i);
    }
    for (int i=0; i<NSEND && i<nsend; i++)
    {
      SAVESENDCONN(i);
    }
  }
#endif
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

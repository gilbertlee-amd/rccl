// Implementation of primitive types
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, class FUNC>
class ncclPrimitives2 {
 private:
  const int tid;
  const int nthreads;
  int nrecv = 0;
  int nsend = 0;
  const int stepSize;
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
  struct ncclDevComm* comm;
  uint32_t* abortCount;

  __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
  __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
  __device__ const T* recvPtr(int i) { return ((const T*)recvBuff[i])+recvOffset(i); }
  __device__ T* sendPtr(int i) { return ((T*)sendBuff[i])+sendOffset(i); }

  __device__ void barrier() {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    __syncthreads();
#else
    asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
#endif
  }

  uint32_t mismatch = 0;
  const uint64_t opCount;

  __device__ void checkMismatch(volatile uint64_t* remoteOpCount) {
    if (mismatch) {
      // In non-LL, we use _threadfence_system before incrementing opCount, yet we are still waiting for credits here, so there must be a size mismatch
      STORE(comm->fatalDevError, ncclDevAssertedMismatch);
    } else if (remoteOpCount && LOAD(remoteOpCount) > opCount) {
      mismatch += 1;
    }
  }

  uint32_t spins = 0;
  uint32_t abort = 0;

  __device__ int checkAbort(volatile uint64_t* remoteOpCount) {
    spins++;
    if (spins == SPINS_BEFORE_CHECK_ABORT) {
      abort = LOAD(comm->abortFlag);
      checkMismatch(remoteOpCount);
      spins = 0;
    }
    return abort;
  }

  __device__ void waitRecv(int i) {
    spins = 0;
    mismatch = 0;
    recvStep[i] += SLICESTEPS;
    if (tid == i) {
#ifdef ENABLE_PROFILING
      auto devProf = comm->devProf;
      uint64_t t0 = clock64();
#endif
      while (LOAD(waitPtr) < recvStep[i]) {
        if (checkAbort(recvConn[i]->opCountRem)) break;
      }
#ifdef ENABLE_PROFILING
      __atomic_fetch_add(&devProf->wait_recv_cycle[blockIdx.x], clock64() - t0, __ATOMIC_SEQ_CST);
#endif
    }
  }

  __device__ void waitSend(int i) {
    spins = 0;
    mismatch = 0;
    sendStep[i] += SLICESTEPS;
    if (tid == WARP_SIZE+i) {
#ifdef ENABLE_PROFILING
      auto devProf = comm->devProf;
      uint64_t t0 = clock64();
#endif
      while (sendConnHead[i] + NCCL_STEPS < sendStep[i]) {
        sendConnHead[i] = LOAD(waitPtr);
        if (checkAbort(sendConn[i]->opCountRem)) break;
      }
#ifdef ENABLE_PROFILING
      __atomic_fetch_add(&devProf->wait_send_cycle[blockIdx.x], clock64() - t0, __ATOMIC_SEQ_CST);
#endif
    }
  }

  inline __device__ void postRecv(int i) {
    STORE(recvConn[i]->head, recvStep[i]);
  }

  inline __device__ void postSend(int i) {
    if (sendConn[i]->next_hdp_reg) STORE(sendConn[i]->next_hdp_reg, 0x1);
    STORE(sendConn[i]->tail, sendStep[i]);
  }

  __device__ void postSendSize(int i, int size) {
    if (sendConn[i]->fifo) STORE(sendConn[i]->fifo+((sendStep[i]-SLICESTEPS)%NCCL_STEPS), size);
  }

  template <int DIRECTRECV>
  __device__ const T* directRecvPtr(int i, int directOffset) {
    return DIRECTRECV && recvDirectBuff[i] ? recvDirectBuff[i]+directOffset : recvPtr(i);
  }

  template <int DIRECTSEND>
  __device__ T* directSendPtr(int i, int directOffset) {
    return DIRECTSEND && sendDirectBuff[i] ? sendDirectBuff[i]+directOffset : sendPtr(i);
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST>
  __device__ void
  GenericOp(const T* srcPtr, T* dstPtr, int nelem, int directOffset) {
    int offset = 0;
    int sliceSize = stepSize * SLICESTEPS;

    const T* srcs[RECV*NRECV+SRC];
    srcs[0] = SRC ? srcPtr : directRecvPtr<DIRECTRECV>(0, directOffset);
    if (RECV) {
      if (SRC) srcs[1] = recvPtr(0);
      for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = recvPtr(i);
    }

    T* dsts[SEND*NSEND+DST];
    dsts[0] = DST ? dstPtr : directSendPtr<DIRECTSEND>(0, directOffset);
    if (SEND) {
      if (DST) dsts[1] = directSendPtr<DIRECTSEND>(0, directOffset);
      for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = directSendPtr<DIRECTSEND>(i, directOffset);
    }

    #pragma unroll 1
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(sliceSize, nelem-offset));
      FOR_SEND(waitSend);
      FOR_RECV(waitRecv);
      if (realSize > 0) {
        barrier();
        if (DIRECTRECV && recvDirectBuff[0]) {
          // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
          if (SEND) {
            ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, NSEND>(tid, nthreads, 1, srcs, nsend, dsts+1, realSize);
          }
        } else {
          ReduceOrCopyMulti<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST>(tid, nthreads, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize);
        }
      }
      exitIfAbortBarrier(abort, abortCount);
      if (tid == 0) FOR_SEND(postSendSize, realSize*sizeof(T));
      if (SEND) __threadfence_system();
      if (tid == 0) FOR_SEND(postSend);
      if (tid == 0) FOR_RECV(postRecv);
    }
    for (int i=0; i<RECV*NRECV+SRC; i++) srcs[i] += sliceSize;
    for (int i=0; i<SEND*NSEND+DST; i++) dsts[i] += sliceSize;
    offset += sliceSize;
  }

  __device__ void loadRecvConn(struct ncclConnInfo* conn, int i, T* directBuff) {
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

  __device__ void loadSendConn(struct ncclConnInfo* conn, int i, T* directBuff) {
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

  __device__ void saveRecvConn(int i) {
    if (tid == i) {
      STORE(&recvConn[i]->step, recvStep[i]);
      __threadfence_system();
      __atomic_fetch_add(recvConn[i]->opCountLoc, 1, __ATOMIC_SEQ_CST);
    }
  }

  __device__ void saveSendConn(int i) {
    if (tid == WARP_SIZE+i) {
      STORE(&sendConn[i]->step, sendStep[i]);
      __threadfence_system();
      __atomic_fetch_add(sendConn[i]->opCountLoc, 1, __ATOMIC_SEQ_CST);
    }
  }

 public:
  __device__
  ncclPrimitives2(const int tid, const int nthreads, int* recvPeers, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount)
    : comm(comm), tid(tid), nthreads(nthreads), stepSize(stepSize), opCount(opCount) {
    // Make sure step is updated before we read it
    abortCount = channel->abortCount;
    __syncthreads();

    // disable directBuff
    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i, 0);
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i, 0);
  }

  __device__ void
  send(const T* src, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 0>(src, NULL, nelem, 0);
  }
  __device__ void
  directSend(const T* src, int directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 0>(src, NULL, nelem, directOffset);
  }

  __device__ void
  recv(T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ void
  directRecv(T* dst, int directOffset, int nelem) {
    GenericOp<1, 0, 1, 0, 0, 1>(NULL, dst, nelem, directOffset);
  }

  __device__ void
  copySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 1>(src, dst, nelem, 0);
  }
  __device__ void
  directCopySend(const T* src, T* dst, int directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 1>(src, dst, nelem, directOffset);
  }

  __device__ void
  recvCopySend(T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ void
  directRecvCopySend(T* dst, int directOffset, int nelem) {
    GenericOp<1, 1, 1, 1, 0, 1>(NULL, dst, nelem, directOffset);
  }

  __device__ void
  recvReduceCopy(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 1, 1>(src, dst, nelem, 0);
  }

  __device__ void
  recvReduceSend(const T* src, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 0>(src, NULL, nelem, 0);
  }

  __device__ void
  recvReduceCopySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 1>(src, dst, nelem, 0);
  }
  __device__ void
  directRecvReduceCopySend(const T* src, T* dst, int directOffset, int nelem) {
    // Direct is only for the send part
    GenericOp<0, 1, 1, 1, 1, 1>(src, dst, nelem, directOffset);
  }

  __device__ ~ncclPrimitives2() {
    // Save steps for next collective. Have thread 0 do it to be compatible
    // with the way LL works.
    for (int i=0; i<NRECV && i<nrecv; i++) saveRecvConn(i);
    for (int i=0; i<NSEND && i<nsend; i++) saveSendConn(i);
  }
};


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

  ncclPrimitives2<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, FUNC>
    prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm, args->opCount);

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
    prims.send(thisInput+offset, nelem);
    ACCUMULATE_COUNTER(send);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      INIT_COUNTER;
      prims.recvReduceSend(thisInput+offset, nelem);
      ACCUMULATE_COUNTER(recvReduceSend);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    slice = ring->devUserRanks[0];
    offset = chunkOffset + slice * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    INIT_COUNTER;
    prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);
    ACCUMULATE_COUNTER(directRecvReduceCopySend);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      INIT_COUNTER;
      prims.directRecvCopySend(thisOutput+offset, offset, nelem);
      ACCUMULATE_COUNTER(directRecvCopySend);
    }

    // Make final copy from buffer to dest.
    slice = ring->devUserRanks[1];
    offset = chunkOffset + slice * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    // Final wait/copy.
    INIT_COUNTER;
    prims.directRecv(thisOutput+offset, offset, nelem);
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

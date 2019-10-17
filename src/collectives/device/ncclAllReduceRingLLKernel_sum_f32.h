template <int UNROLL, class FUNC, typename T>
__attribute__((noinline)) __device__ void
ncclAllReduceRingKernel2(struct CollectiveArgs *args) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int bid = args->bid;
  struct ncclDevComm *comm = args->comm;
  struct ncclChannel *channel = comm->channels + blockIdx.x;
  struct ncclRing *ring = &channel->ring;
  const ssize_t size = args->N;
  const int nranks = comm->nRanks;
  const int stepSize = channel->buffSize / (sizeof(T) * NCCL_STEPS);
  const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const ssize_t loopSize = args->nChannels * (ssize_t)chunkSize;
  const T *__restrict__ thisInput = (const T *)args->ThisInput;
  T *__restrict__ thisOutput = (T *)args->ThisOutput;

  int nrecv = 0;
  int nsend = 0;
  struct ncclConnInfo *recvConn[(1)];
  struct ncclConnInfo *sendConn[(1)];
  volatile uint64_t *waitPtr;
  uint64_t recvStep[(1)];
  uint64_t sendStep[(1)];
  uint64_t sendConnHead[(1)];
  const T *recvDirectBuff[(1)];
  T *sendDirectBuff[(1)];
  const T *recvBuff[(1)];
  T *sendBuff[(1)];
  uint32_t *abortCount;
  const uint64_t opCount = (args->opCount);
  uint32_t mismatch = 0;
  uint32_t spins = 0;
  uint32_t abort = 0;

  {
    int *recvPeers = &ring->prev;
    int *sendPeers = &ring->next;

    abortCount = channel->abortCount;
    __syncthreads();

    for (int i = 0; i < (1) && recvPeers[i] >= 0; i++) {
      {
        recvConn[i] = &channel->devPeers[recvPeers[i]].recv.conn;
        recvBuff[i] = (const T *)LOAD(&recvConn[i]->buff);
        recvStep[i] = LOAD(&recvConn[i]->step);
        recvStep[i] =
            ROUNDUP(recvStep[i], (ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS) *
                                     (ALLREDUCE_SLICESTEPS));
        if (tid == 0)
          STORE(recvConn[i]->head, recvStep[i]);
        if (tid == i) {
          waitPtr = LOAD(&recvConn[i]->tail);
          STORE(recvConn[i]->opCountLoc, opCount);
        }
        recvDirectBuff[i] = NULL;
        if (0 && recvConn[i]->direct) {
          recvDirectBuff[i] = 0;
          if (tid == 0)
            STORE(recvConn[i]->ptrExchange, 0);
        }
        nrecv++;
      };
    }
    for (int i = 0; i < (1) && sendPeers[i] >= 0; i++) {
      {
        sendConn[i] = &channel->devPeers[sendPeers[i]].send.conn;
        sendBuff[i] = (T *)LOAD(&sendConn[i]->buff);
        sendStep[i] = LOAD(&sendConn[i]->step);
        sendStep[i] =
            ROUNDUP(sendStep[i], (ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS) *
                                     (ALLREDUCE_SLICESTEPS));
        if (tid == WARP_SIZE + i) {
          waitPtr = LOAD(&sendConn[i]->head);
          sendConnHead[i] = LOAD(waitPtr);
          STORE(sendConn[i]->opCountLoc, opCount);
        }
        sendDirectBuff[i] = NULL;
        if (0 && sendConn[i]->direct) {
          void *volatile *ptr = sendConn[i]->ptrExchange;
          while ((sendDirectBuff[i] = (T *)(LOAD(ptr))) == NULL)
            ;
          __syncthreads();
          if (tid == 0)
            STORE(ptr, NULL);
        }
        nsend++;
      };
    }
  }

  for (ssize_t gridOffset = 0; gridOffset < size;
       gridOffset += nranks * loopSize) {
    int realChunkSize =
        min(chunkSize, DIVUP(size - gridOffset, nranks * args->nChannels));
    ALIGN_SIZE(realChunkSize, nthreads * sizeof(uint64_t) / sizeof(T));
    ssize_t chunkOffset = gridOffset + bid * nranks * realChunkSize;

    ssize_t offset;
    int nelem;
    int slice;

    slice = ring->devUserRanks[nranks - 1];
    offset = chunkOffset + slice * realChunkSize;
    nelem = min(realChunkSize, size - offset);

    ;

    {
      const T *src = (thisInput + offset);

      {
        int offset = 0;
        int sliceSize = stepSize * (ALLREDUCE_SLICESTEPS);
        const T *srcs[0 * (1) + 1];
        srcs[0] = 1 ? src : (0 && recvDirectBuff[0]
                                 ? recvDirectBuff[0] + 0
                                 : (((const T *)recvBuff[0]) +
                                    ((recvStep[0] % NCCL_STEPS) * stepSize)));
        if (0) {
          if (1)
            srcs[1] = (((const T *)recvBuff[0]) +
                       ((recvStep[0] % NCCL_STEPS) * stepSize));
          for (int i = 1; i < (1) && i < nrecv; i++)
            srcs[1 + i] = (((const T *)recvBuff[i]) +
                           ((recvStep[i] % NCCL_STEPS) * stepSize));
        }
        T *dsts[1 * (1) + 0];
        dsts[0] = 0 ? NULL : (0 && sendDirectBuff[0]
                                  ? sendDirectBuff[0] + 0
                                  : (((T *)sendBuff[0]) +
                                     ((sendStep[0] % NCCL_STEPS) * stepSize)));
        if (1) {
          if (0)
            dsts[1] = (0 && sendDirectBuff[0]
                           ? sendDirectBuff[0] + 0
                           : (((T *)sendBuff[0]) +
                              ((sendStep[0] % NCCL_STEPS) * stepSize)));
          for (int i = 1; i < (1) && i < nsend; i++)
            dsts[0 + i] = (0 && sendDirectBuff[i]
                               ? sendDirectBuff[i] + 0
                               : (((T *)sendBuff[i]) +
                                  ((sendStep[i] % NCCL_STEPS) * stepSize)));
        }
        for (int slice = 0;
             slice < (ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS); ++slice) {
          int realSize = max(0, min(sliceSize, nelem - offset));
          do {
            if (1) {
              for (int i = 1; i < (1) && i < nsend; i++) {
                spins = 0;
                mismatch = 0;
                sendStep[i] += (ALLREDUCE_SLICESTEPS);
                if (tid == WARP_SIZE + i) {
                  while (sendConnHead[i] + NCCL_STEPS < sendStep[i]) {
                    sendConnHead[i] = LOAD(waitPtr);
                    {
                      spins++;
                      if (spins == SPINS_BEFORE_CHECK_ABORT) {
                        abort = LOAD(comm->abortFlag);
                        {
                          volatile uint64_t *remoteOpCount =
                              sendConn[i]->opCountRem;
                          if (mismatch) {
                            STORE(comm->fatalDevError, ncclDevAssertedMismatch);
                          } else if (remoteOpCount &&
                                     LOAD(remoteOpCount) > opCount) {
                            mismatch += 1;
                          }
                        };
                        spins = 0;
                      }
                    };
                    if (abort)
                      break;
                  }
                }
              };
              {
                spins = 0;
                mismatch = 0;
                sendStep[0] += (ALLREDUCE_SLICESTEPS);
                if (tid == WARP_SIZE + 0) {
                  while (sendConnHead[0] + NCCL_STEPS < sendStep[0]) {
                    sendConnHead[0] = LOAD(waitPtr);
                    {
                      spins++;
                      if (spins == SPINS_BEFORE_CHECK_ABORT) {
                        abort = LOAD(comm->abortFlag);
                        {
                          volatile uint64_t *remoteOpCount =
                              sendConn[0]->opCountRem;
                          if (mismatch) {
                            STORE(comm->fatalDevError, ncclDevAssertedMismatch);
                          } else if (remoteOpCount &&
                                     LOAD(remoteOpCount) > opCount) {
                            mismatch += 1;
                          }
                        };
                        spins = 0;
                      }
                    };
                    if (abort)
                      break;
                  }
                }
              };
            }
          } while (0);
          do {
            if (0) {
              {
                spins = 0;
                mismatch = 0;
                recvStep[0] += (ALLREDUCE_SLICESTEPS);
                if (tid == 0) {
                  while (LOAD(waitPtr) < recvStep[0]) {
                    {
                      spins++;
                      if (spins == SPINS_BEFORE_CHECK_ABORT) {
                        abort = LOAD(comm->abortFlag);
                        {
                          volatile uint64_t *remoteOpCount =
                              recvConn[0]->opCountRem;
                          if (mismatch) {
                            STORE(comm->fatalDevError, ncclDevAssertedMismatch);
                          } else if (remoteOpCount &&
                                     LOAD(remoteOpCount) > opCount) {
                            mismatch += 1;
                          }
                        };
                        spins = 0;
                      }
                    };
                    if (abort)
                      break;
                  }
                }
              };
              for (int i = 1; i < (1) && i < nrecv; i++) {
                spins = 0;
                mismatch = 0;
                recvStep[i] += (ALLREDUCE_SLICESTEPS);
                if (tid == i) {
                  while (LOAD(waitPtr) < recvStep[i]) {
                    {
                      spins++;
                      if (spins == SPINS_BEFORE_CHECK_ABORT) {
                        abort = LOAD(comm->abortFlag);
                        {
                          volatile uint64_t *remoteOpCount =
                              recvConn[i]->opCountRem;
                          if (mismatch) {
                            STORE(comm->fatalDevError, ncclDevAssertedMismatch);
                          } else if (remoteOpCount &&
                                     LOAD(remoteOpCount) > opCount) {
                            mismatch += 1;
                          }
                        };
                        spins = 0;
                      }
                    };
                    if (abort)
                      break;
                  }
                }
              };
            }
          } while (0);
          if (realSize > 0) {
            __syncthreads();
            if (0 && recvDirectBuff[0]) {
              if (1) {
                ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, (1)>(
                    tid, nthreads, 1, srcs, nsend, dsts + 1, realSize);
              }
            } else {
              ReduceOrCopyMulti<UNROLL, FUNC, T, 0 + 1, 0 * (1) + 1, 1 + 0,
                                1 * (1) + 0>(tid, nthreads, 0 * nrecv + 1, srcs,
                                             1 * nsend + 0, dsts, realSize);
            }
          }
          if (abort)
            __atomic_fetch_add(abortCount, 1, 5);
          __syncthreads();
          if (LOAD(abortCount)) {
            asm volatile("s_endpgm");
          }
          if (tid == 0) {
            do {
              if (1) {
                for (int i = 1; i < (1) && i < nsend; i++) {
                  if (sendConn[i]->fifo)
                    STORE(sendConn[i]->fifo +
                              ((sendStep[i] - (ALLREDUCE_SLICESTEPS)) %
                               NCCL_STEPS),
                          realSize * sizeof(T));
                };
                {
                  if (sendConn[0]->fifo)
                    STORE(sendConn[0]->fifo +
                              ((sendStep[0] - (ALLREDUCE_SLICESTEPS)) %
                               NCCL_STEPS),
                          realSize * sizeof(T));
                };
              }
            } while (0);
          }
          if (1)
            __threadfence_system();
          if (tid == 0) {
            do {
              if (1) {
                for (int i = 1; i < (1) && i < nsend; i++) {
                  if (sendConn[i]->next_hdp_reg)
                    STORE(sendConn[i]->next_hdp_reg, 0x1);
                  STORE(sendConn[i]->tail, sendStep[i]);
                };
                {
                  if (sendConn[0]->next_hdp_reg)
                    STORE(sendConn[0]->next_hdp_reg, 0x1);
                  STORE(sendConn[0]->tail, sendStep[0]);
                };
              }
            } while (0);
          }
          if (tid == 0) {
            do {
              if (0) {
                { STORE(recvConn[0]->head, recvStep[0]); };
                for (int i = 1; i < (1) && i < nrecv; i++) {
                  STORE(recvConn[i]->head, recvStep[i]);
                };
              }
            } while (0);
          }
          for (int i = 0; i < 0 * (1) + 1; i++)
            srcs[i] += sliceSize;
          for (int i = 0; i < 1 * (1) + 0; i++)
            dsts[i] += sliceSize;
          offset += sliceSize;
        }
      };
    }

    ;

    for (int j = 2; j < nranks; ++j) {
      slice = ring->devUserRanks[nranks - j];
      offset = chunkOffset + slice * realChunkSize;
      nelem = min(realChunkSize, size - offset);

      ;

      {
        const T *src = (thisInput + offset);

        {
          int offset = 0;
          int sliceSize = stepSize * (ALLREDUCE_SLICESTEPS);
          const T *srcs[1 * (1) + 1];
          srcs[0] = 1 ? src : (0 && recvDirectBuff[0]
                                   ? recvDirectBuff[0] + 0
                                   : (((const T *)recvBuff[0]) +
                                      ((recvStep[0] % NCCL_STEPS) * stepSize)));
          if (1) {
            if (1)
              srcs[1] = (((const T *)recvBuff[0]) +
                         ((recvStep[0] % NCCL_STEPS) * stepSize));
            for (int i = 1; i < (1) && i < nrecv; i++)
              srcs[1 + i] = (((const T *)recvBuff[i]) +
                             ((recvStep[i] % NCCL_STEPS) * stepSize));
          }
          T *dsts[1 * (1) + 0];
          dsts[0] =
              0 ? NULL : (0 && sendDirectBuff[0]
                              ? sendDirectBuff[0] + 0
                              : (((T *)sendBuff[0]) +
                                 ((sendStep[0] % NCCL_STEPS) * stepSize)));
          if (1) {
            if (0)
              dsts[1] = (0 && sendDirectBuff[0]
                             ? sendDirectBuff[0] + 0
                             : (((T *)sendBuff[0]) +
                                ((sendStep[0] % NCCL_STEPS) * stepSize)));
            for (int i = 1; i < (1) && i < nsend; i++)
              dsts[0 + i] = (0 && sendDirectBuff[i]
                                 ? sendDirectBuff[i] + 0
                                 : (((T *)sendBuff[i]) +
                                    ((sendStep[i] % NCCL_STEPS) * stepSize)));
          }
          for (int slice = 0;
               slice < (ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS); ++slice) {
            int realSize = max(0, min(sliceSize, nelem - offset));
            do {
              if (1) {
                for (int i = 1; i < (1) && i < nsend; i++) {
                  spins = 0;
                  mismatch = 0;
                  sendStep[i] += (ALLREDUCE_SLICESTEPS);
                  if (tid == WARP_SIZE + i) {
                    while (sendConnHead[i] + NCCL_STEPS < sendStep[i]) {
                      sendConnHead[i] = LOAD(waitPtr);
                      {
                        spins++;
                        if (spins == SPINS_BEFORE_CHECK_ABORT) {
                          abort = LOAD(comm->abortFlag);
                          {
                            volatile uint64_t *remoteOpCount =
                                sendConn[i]->opCountRem;
                            if (mismatch) {
                              STORE(comm->fatalDevError,
                                    ncclDevAssertedMismatch);
                            } else if (remoteOpCount &&
                                       LOAD(remoteOpCount) > opCount) {
                              mismatch += 1;
                            }
                          };
                          spins = 0;
                        }
                      };
                      if (abort)
                        break;
                    }
                  }
                };
                {
                  spins = 0;
                  mismatch = 0;
                  sendStep[0] += (ALLREDUCE_SLICESTEPS);
                  if (tid == WARP_SIZE + 0) {
                    while (sendConnHead[0] + NCCL_STEPS < sendStep[0]) {
                      sendConnHead[0] = LOAD(waitPtr);
                      {
                        spins++;
                        if (spins == SPINS_BEFORE_CHECK_ABORT) {
                          abort = LOAD(comm->abortFlag);
                          {
                            volatile uint64_t *remoteOpCount =
                                sendConn[0]->opCountRem;
                            if (mismatch) {
                              STORE(comm->fatalDevError,
                                    ncclDevAssertedMismatch);
                            } else if (remoteOpCount &&
                                       LOAD(remoteOpCount) > opCount) {
                              mismatch += 1;
                            }
                          };
                          spins = 0;
                        }
                      };
                      if (abort)
                        break;
                    }
                  }
                };
              }
            } while (0);
            do {
              if (1) {
                {
                  spins = 0;
                  mismatch = 0;
                  recvStep[0] += (ALLREDUCE_SLICESTEPS);
                  if (tid == 0) {
                    while (LOAD(waitPtr) < recvStep[0]) {
                      {
                        spins++;
                        if (spins == SPINS_BEFORE_CHECK_ABORT) {
                          abort = LOAD(comm->abortFlag);
                          {
                            volatile uint64_t *remoteOpCount =
                                recvConn[0]->opCountRem;
                            if (mismatch) {
                              STORE(comm->fatalDevError,
                                    ncclDevAssertedMismatch);
                            } else if (remoteOpCount &&
                                       LOAD(remoteOpCount) > opCount) {
                              mismatch += 1;
                            }
                          };
                          spins = 0;
                        }
                      };
                      if (abort)
                        break;
                    }
                  }
                };
                for (int i = 1; i < (1) && i < nrecv; i++) {
                  spins = 0;
                  mismatch = 0;
                  recvStep[i] += (ALLREDUCE_SLICESTEPS);
                  if (tid == i) {
                    while (LOAD(waitPtr) < recvStep[i]) {
                      {
                        spins++;
                        if (spins == SPINS_BEFORE_CHECK_ABORT) {
                          abort = LOAD(comm->abortFlag);
                          {
                            volatile uint64_t *remoteOpCount =
                                recvConn[i]->opCountRem;
                            if (mismatch) {
                              STORE(comm->fatalDevError,
                                    ncclDevAssertedMismatch);
                            } else if (remoteOpCount &&
                                       LOAD(remoteOpCount) > opCount) {
                              mismatch += 1;
                            }
                          };
                          spins = 0;
                        }
                      };
                      if (abort)
                        break;
                    }
                  }
                };
              }
            } while (0);
            if (realSize > 0) {
              __syncthreads();
              if (0 && recvDirectBuff[0]) {
                if (1) {
                  ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, (1)>(
                      tid, nthreads, 1, srcs, nsend, dsts + 1, realSize);
                }
              } else {
                ReduceOrCopyMulti<UNROLL, FUNC, T, 1 + 1, 1 * (1) + 1, 1 + 0,
                                  1 * (1) + 0>(tid, nthreads, 1 * nrecv + 1,
                                               srcs, 1 * nsend + 0, dsts,
                                               realSize);
              }
            }
            if (abort)
              __atomic_fetch_add(abortCount, 1, 5);
            __syncthreads();
            if (LOAD(abortCount)) {
              asm volatile("s_endpgm");
            }
            if (tid == 0) {
              do {
                if (1) {
                  for (int i = 1; i < (1) && i < nsend; i++) {
                    if (sendConn[i]->fifo)
                      STORE(sendConn[i]->fifo +
                                ((sendStep[i] - (ALLREDUCE_SLICESTEPS)) %
                                 NCCL_STEPS),
                            realSize * sizeof(T));
                  };
                  {
                    if (sendConn[0]->fifo)
                      STORE(sendConn[0]->fifo +
                                ((sendStep[0] - (ALLREDUCE_SLICESTEPS)) %
                                 NCCL_STEPS),
                            realSize * sizeof(T));
                  };
                }
              } while (0);
            }
            if (1)
              __threadfence_system();
            if (tid == 0) {
              do {
                if (1) {
                  for (int i = 1; i < (1) && i < nsend; i++) {
                    if (sendConn[i]->next_hdp_reg)
                      STORE(sendConn[i]->next_hdp_reg, 0x1);
                    STORE(sendConn[i]->tail, sendStep[i]);
                  };
                  {
                    if (sendConn[0]->next_hdp_reg)
                      STORE(sendConn[0]->next_hdp_reg, 0x1);
                    STORE(sendConn[0]->tail, sendStep[0]);
                  };
                }
              } while (0);
            }
            if (tid == 0) {
              do {
                if (1) {
                  { STORE(recvConn[0]->head, recvStep[0]); };
                  for (int i = 1; i < (1) && i < nrecv; i++) {
                    STORE(recvConn[i]->head, recvStep[i]);
                  };
                }
              } while (0);
            }
            for (int i = 0; i < 1 * (1) + 1; i++)
              srcs[i] += sliceSize;
            for (int i = 0; i < 1 * (1) + 0; i++)
              dsts[i] += sliceSize;
            offset += sliceSize;
          }
        };
      }

      ;
    }

    slice = ring->devUserRanks[0];
    offset = chunkOffset + slice * realChunkSize;
    nelem = min(realChunkSize, size - offset);

    ;

    {
      const T *src = (thisInput + offset);
      T *dst = (thisOutput + offset);
      int directOffset = offset;

      {
        int offset = 0;
        int sliceSize = stepSize * (ALLREDUCE_SLICESTEPS);
        const T *srcs[1 * (1) + 1];
        srcs[0] = 1 ? src : (0 && recvDirectBuff[0]
                                 ? recvDirectBuff[0] + directOffset
                                 : (((const T *)recvBuff[0]) +
                                    ((recvStep[0] % NCCL_STEPS) * stepSize)));
        if (1) {
          if (1)
            srcs[1] = (((const T *)recvBuff[0]) +
                       ((recvStep[0] % NCCL_STEPS) * stepSize));
          for (int i = 1; i < (1) && i < nrecv; i++)
            srcs[1 + i] = (((const T *)recvBuff[i]) +
                           ((recvStep[i] % NCCL_STEPS) * stepSize));
        }
        T *dsts[1 * (1) + 1];
        dsts[0] = 1 ? dst : (1 && sendDirectBuff[0]
                                 ? sendDirectBuff[0] + directOffset
                                 : (((T *)sendBuff[0]) +
                                    ((sendStep[0] % NCCL_STEPS) * stepSize)));
        if (1) {
          if (1)
            dsts[1] = (1 && sendDirectBuff[0]
                           ? sendDirectBuff[0] + directOffset
                           : (((T *)sendBuff[0]) +
                              ((sendStep[0] % NCCL_STEPS) * stepSize)));
          for (int i = 1; i < (1) && i < nsend; i++)
            dsts[1 + i] = (1 && sendDirectBuff[i]
                               ? sendDirectBuff[i] + directOffset
                               : (((T *)sendBuff[i]) +
                                  ((sendStep[i] % NCCL_STEPS) * stepSize)));
        }
        for (int slice = 0;
             slice < (ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS); ++slice) {
          int realSize = max(0, min(sliceSize, nelem - offset));
          do {
            if (1) {
              for (int i = 1; i < (1) && i < nsend; i++) {
                spins = 0;
                mismatch = 0;
                sendStep[i] += (ALLREDUCE_SLICESTEPS);
                if (tid == WARP_SIZE + i) {
                  while (sendConnHead[i] + NCCL_STEPS < sendStep[i]) {
                    sendConnHead[i] = LOAD(waitPtr);
                    {
                      spins++;
                      if (spins == SPINS_BEFORE_CHECK_ABORT) {
                        abort = LOAD(comm->abortFlag);
                        {
                          volatile uint64_t *remoteOpCount =
                              sendConn[i]->opCountRem;
                          if (mismatch) {
                            STORE(comm->fatalDevError, ncclDevAssertedMismatch);
                          } else if (remoteOpCount &&
                                     LOAD(remoteOpCount) > opCount) {
                            mismatch += 1;
                          }
                        };
                        spins = 0;
                      }
                    };
                    if (abort)
                      break;
                  }
                }
              };
              {
                spins = 0;
                mismatch = 0;
                sendStep[0] += (ALLREDUCE_SLICESTEPS);
                if (tid == WARP_SIZE + 0) {
                  while (sendConnHead[0] + NCCL_STEPS < sendStep[0]) {
                    sendConnHead[0] = LOAD(waitPtr);
                    {
                      spins++;
                      if (spins == SPINS_BEFORE_CHECK_ABORT) {
                        abort = LOAD(comm->abortFlag);
                        {
                          volatile uint64_t *remoteOpCount =
                              sendConn[0]->opCountRem;
                          if (mismatch) {
                            STORE(comm->fatalDevError, ncclDevAssertedMismatch);
                          } else if (remoteOpCount &&
                                     LOAD(remoteOpCount) > opCount) {
                            mismatch += 1;
                          }
                        };
                        spins = 0;
                      }
                    };
                    if (abort)
                      break;
                  }
                }
              };
            }
          } while (0);
          do {
            if (1) {
              {
                spins = 0;
                mismatch = 0;
                recvStep[0] += (ALLREDUCE_SLICESTEPS);
                if (tid == 0) {
                  while (LOAD(waitPtr) < recvStep[0]) {
                    {
                      spins++;
                      if (spins == SPINS_BEFORE_CHECK_ABORT) {
                        abort = LOAD(comm->abortFlag);
                        {
                          volatile uint64_t *remoteOpCount =
                              recvConn[0]->opCountRem;
                          if (mismatch) {
                            STORE(comm->fatalDevError, ncclDevAssertedMismatch);
                          } else if (remoteOpCount &&
                                     LOAD(remoteOpCount) > opCount) {
                            mismatch += 1;
                          }
                        };
                        spins = 0;
                      }
                    };
                    if (abort)
                      break;
                  }
                }
              };
              for (int i = 1; i < (1) && i < nrecv; i++) {
                spins = 0;
                mismatch = 0;
                recvStep[i] += (ALLREDUCE_SLICESTEPS);
                if (tid == i) {
                  while (LOAD(waitPtr) < recvStep[i]) {
                    {
                      spins++;
                      if (spins == SPINS_BEFORE_CHECK_ABORT) {
                        abort = LOAD(comm->abortFlag);
                        {
                          volatile uint64_t *remoteOpCount =
                              recvConn[i]->opCountRem;
                          if (mismatch) {
                            STORE(comm->fatalDevError, ncclDevAssertedMismatch);
                          } else if (remoteOpCount &&
                                     LOAD(remoteOpCount) > opCount) {
                            mismatch += 1;
                          }
                        };
                        spins = 0;
                      }
                    };
                    if (abort)
                      break;
                  }
                }
              };
            }
          } while (0);
          if (realSize > 0) {
            __syncthreads();
            if (0 && recvDirectBuff[0]) {
              if (1) {
                ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, (1)>(
                    tid, nthreads, 1, srcs, nsend, dsts + 1, realSize);
              }
            } else {
              ReduceOrCopyMulti<UNROLL, FUNC, T, 1 + 1, 1 * (1) + 1, 1 + 1,
                                1 * (1) + 1>(tid, nthreads, 1 * nrecv + 1, srcs,
                                             1 * nsend + 1, dsts, realSize);
            }
          }
          if (abort)
            __atomic_fetch_add(abortCount, 1, 5);
          __syncthreads();
          if (LOAD(abortCount)) {
            asm volatile("s_endpgm");
          }
          if (tid == 0) {
            do {
              if (1) {
                for (int i = 1; i < (1) && i < nsend; i++) {
                  if (sendConn[i]->fifo)
                    STORE(sendConn[i]->fifo +
                              ((sendStep[i] - (ALLREDUCE_SLICESTEPS)) %
                               NCCL_STEPS),
                          realSize * sizeof(T));
                };
                {
                  if (sendConn[0]->fifo)
                    STORE(sendConn[0]->fifo +
                              ((sendStep[0] - (ALLREDUCE_SLICESTEPS)) %
                               NCCL_STEPS),
                          realSize * sizeof(T));
                };
              }
            } while (0);
          }
          if (1)
            __threadfence_system();
          if (tid == 0) {
            do {
              if (1) {
                for (int i = 1; i < (1) && i < nsend; i++) {
                  if (sendConn[i]->next_hdp_reg)
                    STORE(sendConn[i]->next_hdp_reg, 0x1);
                  STORE(sendConn[i]->tail, sendStep[i]);
                };
                {
                  if (sendConn[0]->next_hdp_reg)
                    STORE(sendConn[0]->next_hdp_reg, 0x1);
                  STORE(sendConn[0]->tail, sendStep[0]);
                };
              }
            } while (0);
          }
          if (tid == 0) {
            do {
              if (1) {
                { STORE(recvConn[0]->head, recvStep[0]); };
                for (int i = 1; i < (1) && i < nrecv; i++) {
                  STORE(recvConn[i]->head, recvStep[i]);
                };
              }
            } while (0);
          }
          for (int i = 0; i < 1 * (1) + 1; i++)
            srcs[i] += sliceSize;
          for (int i = 0; i < 1 * (1) + 1; i++)
            dsts[i] += sliceSize;
          offset += sliceSize;
        }
      };
    }

    ;

    for (int j = 1; j < nranks - 1; ++j) {
      slice = ring->devUserRanks[nranks - j];
      offset = chunkOffset + slice * realChunkSize;
      nelem = min(realChunkSize, size - offset);

      ;

      {
        T *dst = (thisOutput + offset);
        int directOffset = offset;

        {
          int offset = 0;
          int sliceSize = stepSize * (ALLREDUCE_SLICESTEPS);
          const T *srcs[1 * (1) + 0];
          srcs[0] =
              0 ? NULL : (1 && recvDirectBuff[0]
                              ? recvDirectBuff[0] + directOffset
                              : (((const T *)recvBuff[0]) +
                                 ((recvStep[0] % NCCL_STEPS) * stepSize)));
          if (1) {
            if (0)
              srcs[1] = (((const T *)recvBuff[0]) +
                         ((recvStep[0] % NCCL_STEPS) * stepSize));
            for (int i = 1; i < (1) && i < nrecv; i++)
              srcs[0 + i] = (((const T *)recvBuff[i]) +
                             ((recvStep[i] % NCCL_STEPS) * stepSize));
          }
          T *dsts[1 * (1) + 1];
          dsts[0] = 1 ? dst : (1 && sendDirectBuff[0]
                                   ? sendDirectBuff[0] + directOffset
                                   : (((T *)sendBuff[0]) +
                                      ((sendStep[0] % NCCL_STEPS) * stepSize)));
          if (1) {
            if (1)
              dsts[1] = (1 && sendDirectBuff[0]
                             ? sendDirectBuff[0] + directOffset
                             : (((T *)sendBuff[0]) +
                                ((sendStep[0] % NCCL_STEPS) * stepSize)));
            for (int i = 1; i < (1) && i < nsend; i++)
              dsts[1 + i] = (1 && sendDirectBuff[i]
                                 ? sendDirectBuff[i] + directOffset
                                 : (((T *)sendBuff[i]) +
                                    ((sendStep[i] % NCCL_STEPS) * stepSize)));
          }
          for (int slice = 0;
               slice < (ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS); ++slice) {
            int realSize = max(0, min(sliceSize, nelem - offset));
            do {
              if (1) {
                for (int i = 1; i < (1) && i < nsend; i++) {
                  spins = 0;
                  mismatch = 0;
                  sendStep[i] += (ALLREDUCE_SLICESTEPS);
                  if (tid == WARP_SIZE + i) {
                    while (sendConnHead[i] + NCCL_STEPS < sendStep[i]) {
                      sendConnHead[i] = LOAD(waitPtr);
                      {
                        spins++;
                        if (spins == SPINS_BEFORE_CHECK_ABORT) {
                          abort = LOAD(comm->abortFlag);
                          {
                            volatile uint64_t *remoteOpCount =
                                sendConn[i]->opCountRem;
                            if (mismatch) {
                              STORE(comm->fatalDevError,
                                    ncclDevAssertedMismatch);
                            } else if (remoteOpCount &&
                                       LOAD(remoteOpCount) > opCount) {
                              mismatch += 1;
                            }
                          };
                          spins = 0;
                        }
                      };
                      if (abort)
                        break;
                    }
                  }
                };
                {
                  spins = 0;
                  mismatch = 0;
                  sendStep[0] += (ALLREDUCE_SLICESTEPS);
                  if (tid == WARP_SIZE + 0) {
                    while (sendConnHead[0] + NCCL_STEPS < sendStep[0]) {
                      sendConnHead[0] = LOAD(waitPtr);
                      {
                        spins++;
                        if (spins == SPINS_BEFORE_CHECK_ABORT) {
                          abort = LOAD(comm->abortFlag);
                          {
                            volatile uint64_t *remoteOpCount =
                                sendConn[0]->opCountRem;
                            if (mismatch) {
                              STORE(comm->fatalDevError,
                                    ncclDevAssertedMismatch);
                            } else if (remoteOpCount &&
                                       LOAD(remoteOpCount) > opCount) {
                              mismatch += 1;
                            }
                          };
                          spins = 0;
                        }
                      };
                      if (abort)
                        break;
                    }
                  }
                };
              }
            } while (0);
            do {
              if (1) {
                {
                  spins = 0;
                  mismatch = 0;
                  recvStep[0] += (ALLREDUCE_SLICESTEPS);
                  if (tid == 0) {
                    while (LOAD(waitPtr) < recvStep[0]) {
                      {
                        spins++;
                        if (spins == SPINS_BEFORE_CHECK_ABORT) {
                          abort = LOAD(comm->abortFlag);
                          {
                            volatile uint64_t *remoteOpCount =
                                recvConn[0]->opCountRem;
                            if (mismatch) {
                              STORE(comm->fatalDevError,
                                    ncclDevAssertedMismatch);
                            } else if (remoteOpCount &&
                                       LOAD(remoteOpCount) > opCount) {
                              mismatch += 1;
                            }
                          };
                          spins = 0;
                        }
                      };
                      if (abort)
                        break;
                    }
                  }
                };
                for (int i = 1; i < (1) && i < nrecv; i++) {
                  spins = 0;
                  mismatch = 0;
                  recvStep[i] += (ALLREDUCE_SLICESTEPS);
                  if (tid == i) {
                    while (LOAD(waitPtr) < recvStep[i]) {
                      {
                        spins++;
                        if (spins == SPINS_BEFORE_CHECK_ABORT) {
                          abort = LOAD(comm->abortFlag);
                          {
                            volatile uint64_t *remoteOpCount =
                                recvConn[i]->opCountRem;
                            if (mismatch) {
                              STORE(comm->fatalDevError,
                                    ncclDevAssertedMismatch);
                            } else if (remoteOpCount &&
                                       LOAD(remoteOpCount) > opCount) {
                              mismatch += 1;
                            }
                          };
                          spins = 0;
                        }
                      };
                      if (abort)
                        break;
                    }
                  }
                };
              }
            } while (0);
            if (realSize > 0) {
              __syncthreads();
              if (1 && recvDirectBuff[0]) {
                if (1) {
                  ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, (1)>(
                      tid, nthreads, 1, srcs, nsend, dsts + 1, realSize);
                }
              } else {
                ReduceOrCopyMulti<UNROLL, FUNC, T, 1 + 0, 1 * (1) + 0, 1 + 1,
                                  1 * (1) + 1>(tid, nthreads, 1 * nrecv + 0,
                                               srcs, 1 * nsend + 1, dsts,
                                               realSize);
              }
            }
            if (abort)
              __atomic_fetch_add(abortCount, 1, 5);
            __syncthreads();
            if (LOAD(abortCount)) {
              asm volatile("s_endpgm");
            }
            if (tid == 0) {
              do {
                if (1) {
                  for (int i = 1; i < (1) && i < nsend; i++) {
                    if (sendConn[i]->fifo)
                      STORE(sendConn[i]->fifo +
                                ((sendStep[i] - (ALLREDUCE_SLICESTEPS)) %
                                 NCCL_STEPS),
                            realSize * sizeof(T));
                  };
                  {
                    if (sendConn[0]->fifo)
                      STORE(sendConn[0]->fifo +
                                ((sendStep[0] - (ALLREDUCE_SLICESTEPS)) %
                                 NCCL_STEPS),
                            realSize * sizeof(T));
                  };
                }
              } while (0);
            }
            if (1)
              __threadfence_system();
            if (tid == 0) {
              do {
                if (1) {
                  for (int i = 1; i < (1) && i < nsend; i++) {
                    if (sendConn[i]->next_hdp_reg)
                      STORE(sendConn[i]->next_hdp_reg, 0x1);
                    STORE(sendConn[i]->tail, sendStep[i]);
                  };
                  {
                    if (sendConn[0]->next_hdp_reg)
                      STORE(sendConn[0]->next_hdp_reg, 0x1);
                    STORE(sendConn[0]->tail, sendStep[0]);
                  };
                }
              } while (0);
            }
            if (tid == 0) {
              do {
                if (1) {
                  { STORE(recvConn[0]->head, recvStep[0]); };
                  for (int i = 1; i < (1) && i < nrecv; i++) {
                    STORE(recvConn[i]->head, recvStep[i]);
                  };
                }
              } while (0);
            }
            for (int i = 0; i < 1 * (1) + 0; i++)
              srcs[i] += sliceSize;
            for (int i = 0; i < 1 * (1) + 1; i++)
              dsts[i] += sliceSize;
            offset += sliceSize;
          }
        };
      }

      ;
    }

    slice = ring->devUserRanks[1];
    offset = chunkOffset + slice * realChunkSize;
    nelem = min(realChunkSize, size - offset);

    ;

    {
      T *dst = (thisOutput + offset);
      int directOffset = offset;

      {
        int offset = 0;
        int sliceSize = stepSize * (ALLREDUCE_SLICESTEPS);
        const T *srcs[1 * (1) + 0];
        srcs[0] = 0 ? NULL : (1 && recvDirectBuff[0]
                                  ? recvDirectBuff[0] + directOffset
                                  : (((const T *)recvBuff[0]) +
                                     ((recvStep[0] % NCCL_STEPS) * stepSize)));
        if (1) {
          if (0)
            srcs[1] = (((const T *)recvBuff[0]) +
                       ((recvStep[0] % NCCL_STEPS) * stepSize));
          for (int i = 1; i < (1) && i < nrecv; i++)
            srcs[0 + i] = (((const T *)recvBuff[i]) +
                           ((recvStep[i] % NCCL_STEPS) * stepSize));
        }
        T *dsts[0 * (1) + 1];
        dsts[0] = 1 ? dst : (0 && sendDirectBuff[0]
                                 ? sendDirectBuff[0] + directOffset
                                 : (((T *)sendBuff[0]) +
                                    ((sendStep[0] % NCCL_STEPS) * stepSize)));
        if (0) {
          if (1)
            dsts[1] = (0 && sendDirectBuff[0]
                           ? sendDirectBuff[0] + directOffset
                           : (((T *)sendBuff[0]) +
                              ((sendStep[0] % NCCL_STEPS) * stepSize)));
          for (int i = 1; i < (1) && i < nsend; i++)
            dsts[1 + i] = (0 && sendDirectBuff[i]
                               ? sendDirectBuff[i] + directOffset
                               : (((T *)sendBuff[i]) +
                                  ((sendStep[i] % NCCL_STEPS) * stepSize)));
        }
        for (int slice = 0;
             slice < (ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS); ++slice) {
          int realSize = max(0, min(sliceSize, nelem - offset));
          do {
            if (0) {
              for (int i = 1; i < (1) && i < nsend; i++) {
                spins = 0;
                mismatch = 0;
                sendStep[i] += (ALLREDUCE_SLICESTEPS);
                if (tid == WARP_SIZE + i) {
                  while (sendConnHead[i] + NCCL_STEPS < sendStep[i]) {
                    sendConnHead[i] = LOAD(waitPtr);
                    {
                      spins++;
                      if (spins == SPINS_BEFORE_CHECK_ABORT) {
                        abort = LOAD(comm->abortFlag);
                        {
                          volatile uint64_t *remoteOpCount =
                              sendConn[i]->opCountRem;
                          if (mismatch) {
                            STORE(comm->fatalDevError, ncclDevAssertedMismatch);
                          } else if (remoteOpCount &&
                                     LOAD(remoteOpCount) > opCount) {
                            mismatch += 1;
                          }
                        };
                        spins = 0;
                      }
                    };
                    if (abort)
                      break;
                  }
                }
              };
              {
                spins = 0;
                mismatch = 0;
                sendStep[0] += (ALLREDUCE_SLICESTEPS);
                if (tid == WARP_SIZE + 0) {
                  while (sendConnHead[0] + NCCL_STEPS < sendStep[0]) {
                    sendConnHead[0] = LOAD(waitPtr);
                    {
                      spins++;
                      if (spins == SPINS_BEFORE_CHECK_ABORT) {
                        abort = LOAD(comm->abortFlag);
                        {
                          volatile uint64_t *remoteOpCount =
                              sendConn[0]->opCountRem;
                          if (mismatch) {
                            STORE(comm->fatalDevError, ncclDevAssertedMismatch);
                          } else if (remoteOpCount &&
                                     LOAD(remoteOpCount) > opCount) {
                            mismatch += 1;
                          }
                        };
                        spins = 0;
                      }
                    };
                    if (abort)
                      break;
                  }
                }
              };
            }
          } while (0);
          do {
            if (1) {
              {
                spins = 0;
                mismatch = 0;
                recvStep[0] += (ALLREDUCE_SLICESTEPS);
                if (tid == 0) {
                  while (LOAD(waitPtr) < recvStep[0]) {
                    {
                      spins++;
                      if (spins == SPINS_BEFORE_CHECK_ABORT) {
                        abort = LOAD(comm->abortFlag);
                        {
                          volatile uint64_t *remoteOpCount =
                              recvConn[0]->opCountRem;
                          if (mismatch) {
                            STORE(comm->fatalDevError, ncclDevAssertedMismatch);
                          } else if (remoteOpCount &&
                                     LOAD(remoteOpCount) > opCount) {
                            mismatch += 1;
                          }
                        };
                        spins = 0;
                      }
                    };
                    if (abort)
                      break;
                  }
                }
              };
              for (int i = 1; i < (1) && i < nrecv; i++) {
                spins = 0;
                mismatch = 0;
                recvStep[i] += (ALLREDUCE_SLICESTEPS);
                if (tid == i) {
                  while (LOAD(waitPtr) < recvStep[i]) {
                    {
                      spins++;
                      if (spins == SPINS_BEFORE_CHECK_ABORT) {
                        abort = LOAD(comm->abortFlag);
                        {
                          volatile uint64_t *remoteOpCount =
                              recvConn[i]->opCountRem;
                          if (mismatch) {
                            STORE(comm->fatalDevError, ncclDevAssertedMismatch);
                          } else if (remoteOpCount &&
                                     LOAD(remoteOpCount) > opCount) {
                            mismatch += 1;
                          }
                        };
                        spins = 0;
                      }
                    };
                    if (abort)
                      break;
                  }
                }
              };
            }
          } while (0);
          if (realSize > 0) {
            __syncthreads();
            if (1 && recvDirectBuff[0]) {
              if (0) {
                ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, (1)>(
                    tid, nthreads, 1, srcs, nsend, dsts + 1, realSize);
              }
            } else {
              ReduceOrCopyMulti<UNROLL, FUNC, T, 1 + 0, 1 * (1) + 0, 0 + 1,
                                0 * (1) + 1>(tid, nthreads, 1 * nrecv + 0, srcs,
                                             0 * nsend + 1, dsts, realSize);
            }
          }
          if (abort)
            __atomic_fetch_add(abortCount, 1, 5);
          __syncthreads();
          if (LOAD(abortCount)) {
            asm volatile("s_endpgm");
          }
          if (tid == 0) {
            do {
              if (0) {
                for (int i = 1; i < (1) && i < nsend; i++) {
                  if (sendConn[i]->fifo)
                    STORE(sendConn[i]->fifo +
                              ((sendStep[i] - (ALLREDUCE_SLICESTEPS)) %
                               NCCL_STEPS),
                          realSize * sizeof(T));
                };
                {
                  if (sendConn[0]->fifo)
                    STORE(sendConn[0]->fifo +
                              ((sendStep[0] - (ALLREDUCE_SLICESTEPS)) %
                               NCCL_STEPS),
                          realSize * sizeof(T));
                };
              }
            } while (0);
          }
          if (0)
            __threadfence_system();
          if (tid == 0) {
            do {
              if (0) {
                for (int i = 1; i < (1) && i < nsend; i++) {
                  if (sendConn[i]->next_hdp_reg)
                    STORE(sendConn[i]->next_hdp_reg, 0x1);
                  STORE(sendConn[i]->tail, sendStep[i]);
                };
                {
                  if (sendConn[0]->next_hdp_reg)
                    STORE(sendConn[0]->next_hdp_reg, 0x1);
                  STORE(sendConn[0]->tail, sendStep[0]);
                };
              }
            } while (0);
          }
          if (tid == 0) {
            do {
              if (1) {
                { STORE(recvConn[0]->head, recvStep[0]); };
                for (int i = 1; i < (1) && i < nrecv; i++) {
                  STORE(recvConn[i]->head, recvStep[i]);
                };
              }
            } while (0);
          }
          for (int i = 0; i < 1 * (1) + 0; i++)
            srcs[i] += sliceSize;
          for (int i = 0; i < 0 * (1) + 1; i++)
            dsts[i] += sliceSize;
          offset += sliceSize;
        }
      };
    }

    ;
  }

  {

    for (int i = 0; i < (1) && i < nrecv; i++) {
      {
        if (tid == i) {
          STORE(&recvConn[i]->step, recvStep[i]);
          __threadfence_system();
          __atomic_fetch_add(recvConn[i]->opCountLoc, 1, 5);
        }
      };
    }
    for (int i = 0; i < (1) && i < nsend; i++) {
      {
        if (tid == WARP_SIZE + i) {
          STORE(&sendConn[i]->step, sendStep[i]);
          __threadfence_system();
          __atomic_fetch_add(sendConn[i]->opCountLoc, 1, 5);
        }
      };
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

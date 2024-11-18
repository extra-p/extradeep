# IMDB Benchmark

This repository contains a deep learning benchmark that trains a DNN on the IMDB data set.
It is written in TensorFlow and uses Horovod for distributed deep learning.
Data parallelism is used to distribute the model to the set number of mpi ranks.
Each rank has its dedicated GPU.
The data set is sharded into the exact same number of pieces as mpi processes.
Experiments can be done using only one parameter p the number of MPI ranks.

For full dataset and instructions see [Link](https://developer.imdb.com/non-commercial-datasets/).
For tiny dataset to get started see [Link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## Network architectures for training

* https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2

## Command Line Arguments

```
usage: main.py [-h] [-p RANKS] [-r REPETITION] [-b BATCHSIZE] [-e EPOCHS] [-verbose {0,1,2}] [--tfprofiler] [--determinism] [--memorygrowth] [--mixedprecision] [--tensor32]

IMDB Benchmark.

optional arguments:
  -h, --help        show this help message and exit
  -p RANKS          set the number of mpi ranks
  -r REPETITION     set the repetition number for this experiment
  -b BATCHSIZE      set the batch size
  -e EPOCHS         set the number of training epochs
  -verbose {0,1,2}  set tensorflow verbose mode option
  --tfprofiler      enable the tf-profiler
  --determinism     enable determinism for tf operations
  --memorygrowth    enable the memory growth for tf
  --mixedprecision  enable mixed precision computing
  --tensor32        enable tensor32 computation
```

## Profiling command for Nsight Systems

```
srun nsys profile -t nvtx,cuda,mpi,cublas,cudnn --mpi-impl openmpi -b none --cpuctxsw none -f true -x true -o imdb.p2.r1.mpi%q{SLURM_PROCID} python -u main.py -p 2 -r 1
```

## Profiling with Darshan

```
export LD_PRELOAD=$HOME/darshan-runtime/lib/libdarshan.so
export DARSHAN_LOG_PATH=darshanlogs
export DARSHAN_LOGFILE=darshanlogs/imdb_p1_r1.log
export DXT_ENABLE_IO_TRACE=1
srun python -u main.py -p 1 -r 1
```

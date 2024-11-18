# CIFAR-10 Benchmark

This repository contains a deep learning benchmark that trains a DNN on the CIFAR-10 data set.
It is written in TensorFlow and uses Horovod for distributed deep learning.
Data parallelism is used to distribute the model to the set number of mpi ranks.
Each rank has its dedicated GPU.
The data set is sharded into the exact same number of pieces as mpi processes.
Experiments can be done using only one parameter p the number of MPI ranks. Or two parameters
p and n the network architecture.

For data set see following link [Link](https://www.cs.utoronto.ca/~kriz/cifar.html).

## Available network architectures for training

* ResNet (18,34,50,101,110,152,164,1001)

## Command Line Arguments

```
usage: main.py [-h] [-p RANKS] [-n {resnet18,resnet34,resnet50,resnet101,resnet110,resnet152,resnet164,resnet1001}] [-r REPETITION] [-b BATCHSIZE]
               [-e EPOCHS] [-nrparameters {1,2}] [-verbose {0,1,2}] [--tfprofiler] [--determinism] [--memorygrowth] [--mixedprecision] [--tensor32]

CIFAR-10 Benchmark.

optional arguments:
  -h, --help            show this help message and exit
  -p RANKS              set the number of mpi ranks
  -n {resnet18,resnet34,resnet50,resnet101,resnet110,resnet152,resnet164,resnet1001}
                        set the network configuration
  -r REPETITION         set the repetition number for this experiment
  -b BATCHSIZE          set the batch size
  -e EPOCHS             set the number of training epochs
  -nrparameters {1,2}   set the number of parameters considered for analysis
  -verbose {0,1,2}      set tensorflow verbose mode option
  --tfprofiler          enable the tf-profiler
  --determinism         enable determinism for tf operations
  --memorygrowth        enable the memory growth for tf
  --mixedprecision      enable mixed precision computing
  --tensor32            enable tensor32 computation
```

## Profiling command for Nsight Systems

```
srun nsys profile -t nvtx,cuda,mpi,cublas,cudnn --mpi-impl openmpi -b none --cpuctxsw none -f true -x true -o cifar10.p2.r1.mpi%q{SLURM_PROCID} python -u main.py -p 2 -n resnet18 -r 1
```

## Profiling with Darshan

```
export LD_PRELOAD=$HOME/darshan-runtime/lib/libdarshan.so
export DARSHAN_LOG_PATH=darshanlogs
export DARSHAN_LOGFILE=darshanlogs/cifar10_p1_r1.log
export DXT_ENABLE_IO_TRACE=1
srun python -u main.py -p 1 -n resnet18 -r 1
``` 

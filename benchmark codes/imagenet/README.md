# ImageNet Benchmark

This repository contains a deep learning benchmark that trains a DNN on the ImageNet data set.
It is written in TensorFlow and uses Horovod for distributed deep learning.
Data parallelism is used to distribute the model to the set number of mpi ranks.
Each rank has its dedicated GPU.
The data set is sharded into the exact same number of pieces as mpi processes.
Experiments can be done using only one parameter p the number of MPI ranks. Or two parameters
p and n the network architecture. Or three parameters p, n and s the problem size i.e.
the number of classes to train the network for.

For the dataset please see the following [Link](https://www.image-net.org/download.php).

## Available network architectures for training

* EfficientNet (b0,b1,b2,b3,b4,b5,b6,b7)

## Command Line Arguments

```
usage: main.py [-h] [-p RANKS] [-n {efficientnetb0,efficientnetb1,efficientnetb2,efficientnetb3,efficientnetb4,efficientnetb5,efficientnetb6,efficientnetb7}] [-r REPETITION] [-b BATCHSIZE] [-e EPOCHS]
               [-nrparameters {1,2,3}] [-s PROBLEMSIZE] [-verbose {0,1,2}] [--tfprofiler] [--determinism] [--memorygrowth] [--mixedprecision] [--tensor32]

ImageNet Benchmark.

optional arguments:
  -h, --help            show this help message and exit
  -p RANKS              set the number of mpi ranks
  -n {efficientnetb0,efficientnetb1,efficientnetb2,efficientnetb3,efficientnetb4,efficientnetb5,efficientnetb6,efficientnetb7}
                        set the network configuration
  -r REPETITION         set the repetition number for this experiment
  -b BATCHSIZE          set the batch size
  -e EPOCHS             set the number of training epochs
  -nrparameters {1,2,3}
                        set the number of parameters considered for analysis
  -s PROBLEMSIZE        set the problem size, the number of classes to train the model for
  -verbose {0,1,2}      set tensorflow verbose mode option
  --tfprofiler          enable the tf-profiler
  --determinism         enable determinism for tf operations
  --memorygrowth        enable the memory growth for tf
  --mixedprecision      enable mixed precision computing
  --tensor32            enable tensor32 computation
```

## Profiling command for Nsight Systems

```
srun nsys profile -t nvtx,cuda,mpi,cublas,cudnn --mpi-impl openmpi -b none --cpuctxsw none -f true -x true -o imagenet.p2.r1.mpi%q{SLURM_PROCID} python -u main.py -p 2 -n efficientnetb1 -r 1
```

## Profiling with Darshan

```
export LD_PRELOAD=$HOME/darshan-runtime/lib/libdarshan.so
export DARSHAN_LOG_PATH=darshanlogs
export DARSHAN_LOGFILE=darshanlogs/imagenet_p1_r1.log
export DXT_ENABLE_IO_TRACE=1
srun python -u main.py -p 1 -n efficientnetb1 -r 1
```

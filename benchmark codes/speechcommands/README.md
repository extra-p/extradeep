# Speech Commands Benchmark

This repository contains a deep learning benchmark that trains a DNN on the Speech Commands data set.
It is written in TensorFlow and uses Horovod for distributed deep learning.
Data parallelism is used to distribute the model to the set number of mpi ranks.
Each rank has its dedicated GPU.
The data set is sharded into the exact same number of pieces as mpi processes.
Experiments can be done using only one parameter p the number of MPI ranks.

For the dataset please see [Link](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data).

## Network architectures for training

* CNN

```
norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

model = models.Sequential([
    layers.Input(shape=input_shape),
    preprocessing.Resizing(32, 32),
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])
```

## Command Line Arguments

```
usage: main.py [-h] [-p RANKS] [-r REPETITION] [-b BATCHSIZE] [-e EPOCHS] [-verbose {0,1,2}] [--tfprofiler] [--determinism] [--memorygrowth] [--mixedprecision] [--tensor32] [--mini]

SpeechCommands Benchmark.

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
  --mini            use mini speech commands data set for training
```

## Profiling command for Nsight Systems

```
srun nsys profile -t nvtx,cuda,mpi,cublas,cudnn --mpi-impl openmpi -b none --cpuctxsw none -f true -x true -o speech.p2.r1.mpi%q{SLURM_PROCID} python -u main.py -p 2 -r 1
```

## Profiling with Darshan

```
export LD_PRELOAD=$HOME/darshan-runtime/lib/libdarshan.so
export DARSHAN_LOG_PATH=darshanlogs
export DARSHAN_LOGFILE=darshanlogs/speech_p1_r1.log
export DXT_ENABLE_IO_TRACE=1
srun python -u main.py -p 1 -r 1
```

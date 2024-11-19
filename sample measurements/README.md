This directory contains some sample measurements from the CIFAR-10 benchmark. It contains samples for:

* I/O analysis measurements done with Darshan
* Full measurements runs with NsightSystems where cudnn, cublas, cuda api, nvtx, mpi, etc. was profiled
* NVTX only measurement runs with NsightSystems where only nvtx was profiled

The provided samples need to be converted into .sqlite format in order to read them with Extra-Deep for model creation.
Since a full set of measurement for a large benchmark as presented in the paper is very large in size (~1TB or more), we can only provide sample source data sets.
We advice to convert the data into Extra-P objects after they have been loaded initially to speedup analysis afterwards.
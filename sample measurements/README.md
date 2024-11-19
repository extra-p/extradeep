
## Measurements

This directory contains some sample measurements from the CIFAR-10 benchmark. It contains samples for:

* I/O analysis measurements done with Darshan
* Full measurements runs with NsightSystems where cudnn, cublas, cuda api, nvtx, mpi, etc. was profiled
* NVTX only measurement runs with NsightSystems where only nvtx was profiled

The larger data sample sets can be obtained at [Link](https://zenodo.org/records/14183393).

The provided samples need to be converted into .sqlite format in order to read them with Extra-Deep for model creation.

Since a full set of measurement for a large benchmark as presented in the paper is very large in size (~1TB or more), we can only provide sample source data sets.

We advice to convert the data into Extra-P objects after they have been loaded initially to speedup analysis afterwards.

## Used Software Packages

1. Stages/2022  
2. GCCcore/.11.2.0  
3. zlib/.1.2.11  
4. binutils/.2.37  
5. GCC  
6. numactl/2.0.14  
7. nvidia-driver/.default  
8. CUDA/11.5  
9. UCX/default  
10. pscom/.5.4-default  
11. XZ/.5.2.5  
12. libxml2/.2.9.10  
13. ParaStationMPI  
14. imkl/.2021.4.0  
15. cuDNN/8.3.1.22-CUDA-11.5  
16. NCCL/2.12.7-1-CUDA-11.5  
17. bzip2/.1.0.8  
18. ncurses/.6.2  
19. libreadline/.8.1  
20. Tcl/8.6.11  
21. SQLite/.3.36  
22. GMP/6.2.1  
23. libffi/.3.4.2  
24. OpenSSL/1.1  
25. libxslt/.1.1.34  
26. libyaml/.0.2.5  
27. PostgreSQL/13.4  
28. gflags/.2.2.2  
29. libspatialindex/.1.9.3  
30. NASM/.2.15.05  
31. libjpeg-turbo/.2.1.1  
32. Python/3.9.6  
33. pybind11/.2.7.1  
34. SciPy-bundle/2021.10  
35. Szip/.2.1.1  
36. HDF5/1.12.1-serial  
37. h5py/3.5.0-serial  
38. cURL/7.78.0  
39. double-conversion/3.1.6  
40. flatbuffers/.2.0.0  
41. giflib/.5.2.1  
42. libpciaccess/.0.16  
43. hwloc/2.5.0  
44. ICU/.70.1  
45. JsonCpp/.1.9.4  
46. LMDB/.0.9.29  
47. nsync/.1.24.0  
48. protobuf/.3.17.3  
49. protobuf-python/.3.17.3  
50. flatbuffers-python/2.0  
51. typing-extensions/3.10.0.0  
52. libpng/.1.6.37  
53. snappy/.1.1.9  
54. TensorFlow/2.6.0-CUDA-11.5  
55. libarchive/3.5.1  
56. CMake

For horovod installation: 
`HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_MPI=1 pip install --no-cache-dir horovod[tensorflow,keras]`.
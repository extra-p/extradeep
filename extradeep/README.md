# Extra-Deep

Extra-Deep is a performance analysis tool for distirbuted deep learning applications written in Python. It enables the users and developers to model the training performance of their deep learning application as a function of one or several input/configuration parameters such as the number of MPI ranks the application is executed with, the batch size per worker (rank), or the trained network configuration (architecture). Though Extra-Deep offers a variety of usefull tools and features for performance analysis:

* Automatic instrumentation of user written code
* GUI tool for performance analysis with visualizations
* Command line tool for performance analysis on clusters (Some of the features are for now only available in command line tool)

For further information please see our paper: [1. Installing Extra-Deep](#link) TODO: add link to my paper.

## Index:

[1. Installing Extra-Deep](#anchor1)

[2. Instrument your code/application](#anchor2)

[3. Profile your applications training performance](#anchor3)

## 1. Installing Extra-Deep
<a name="anchor1"></a>

If you are on Linux and do install Extra-Deep into `~/.local/bin` don't forget to add it to your to your `$PATH` with `export PATH="$HOME/.local/bin:$PATH"`. Otherwise the `extradeep` command is not found.

### 1. Building the package locally with symlink

Installation option one is to build Extra-Deep from source by cloning our GitHub repository and the build and install it using the following commands:

* `git clone https://github.com/extra-p/extradeep`
* `cd extradeep/extradeep`
* `python setup.py sdist bdist_wheel`
* `cd ..`
* `pip install -e extradeep/`

## 2. Instrument your code/application
<a name="anchor2"></a>

Extra-Deep features a instrumentation tool to instrument your deep learning code. To instrument your code execute the following command in the folder where the main method of your application is. The instrumenter will create a new folder containing the instrumented version of the code. Specifying an output path is optional.

`extradeep_instrumenter <main_method_file_path> --out <output_folder_path>`

The instrumenter will only instrument user defined code and no library function from tensorflow etc. For improved coverage and a better performance analysis you should write your code in an object oriented style using functions and classes. Currently the instrumenter can only instrument the code inside user defined functions. To properly instrument the callstack of the user defined functions all your code files of the application have to be in the same root dir and as defined in the import statements in your python code. The test folder `test/sample_code/` features a sample code of a deep learning application written in python using tensorflow that can be instrumented. The automatic and manual instrumentation shown here works only for code written in python. The DL framework does not matter. You can use TensorFlow, PyTorch or similar frameworks.

You can run `extradeep_instrumenter test/main.py --out test/` in the root dir of extradeep repository to try it out and see how the sample code in `test/sample_code/` is instrumented automatically.

Depending on how your code is written the automatic instrumenter might not be able to cover all of the code regions of interest. Therefore, you can instrument sections of your code not covered by the automated instrumenter, or regions of interest for you, manually using the following methods:

```python
@nvtx.annotate(message="my_message", color="blue")
def my_func():
    pass
```

```python
rng = nvtx.start_range(message="my_message", color="blue")
# ... do something ... #
nvtx.end_range(rng)
```

```python
with nvtx.annotate(message="my_message", color="green"):
    pass
```

```python
try:
    something()
except SomeError():
    nvtx.mark(message="some error occurred", color="red")
    # ... do something else ...
```

For the instrumentation of user defined code the nvtx python package is required. It can be installed with:

`python -m pip install nvtx`

For further documentation please see [nvtx](https://nvtx.readthedocs.io/en/latest/index.html).

## 3. Profile your applications training performance
<a name="anchor3"></a>

Please also see the sample job scripts provided in the benchmark codes folder.

### 3.a. Profiling runtime, cuda, MPI, etc. with NsightSystems

`nsys profile -t nvtx,cuda,mpi,cublas,cudnn --mpi-impl openmpi -b none --cpuctxsw none -f true -x true -o cifar10.p"+str(tasks[i])+".r1.mpi%q{SLURM_PROCID} python -u main.py -p "+str(tasks[i])+" -n resnet18 -r 1`

Depending on your job sheduler or cluster you want to run, e.g., `srun` before the `nsys` command and specfiy all parameter for parallel execution.

The `mpi%q{SLURM_PROCID}` is important, so that you get the data from different MPI ranks separately for analysis. Make sure also to set correct MPI implementation depending on what you use on your system.

### 3.b. Profiling hardware counters with NsightCompute

The following shows an example that profiles only nvtx marks and events in the code for a quick high-level analysis.

`nsys profile -t nvtx --mpi-impl openmpi -b none --cpuctxsw none -f true -x true -o cifar10.p"+str(tasks[i])+".r1.mpi%q{SLURM_PROCID} python -u main.py -p "+str(tasks[i])+" -n resnet18 -r 1`

### 3.c. Profiling I/O counters with Darshan

To profile for I/O analysis you can use, e.g., Darshan and run your codes similar like this:

```
export LD_PRELOAD=$HOME/darshan-runtime/lib/libdarshan.so
export DARSHAN_LOG_PATH=darshanlogs
export DARSHAN_LOGFILE=darshanlogs/cifar10_p"+str(tasks[i])+"_r1.log
export DXT_ENABLE_IO_TRACE=1
srun python -u main.py -p "+str(tasks[i])+" -n resnet18 -r 1
```

### 3.d. Profiling hardware counters with NsightCompute

To measure gpu hardware counters, for which Extra-Deep can also create models, you need to run your code like follows:

`ncu --replay-mode kernel --app-replay-buffer memory --target-processes all --metrics """+str(counter_string)+""" -o cifar10_"""+str(metric)+""".p"""+str(t)+""".mpi%q{SLURM_PROCID}.r$SLURM_ARRAY_TASK_ID -f python main.py`

The `counter_string` describes the counter that you want to measure. The metrics.csv file contains a list of counters that can be measured. These counters vary from GPU to GPU and the software versions used!
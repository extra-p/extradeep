import argparse
import nvtx
import horovod.tensorflow.keras as hvd
from imagenetbenchmark import ImageNetBenchmark

parser = argparse.ArgumentParser(description='ImageNet Benchmark.')

parser.add_argument('-p', dest='ranks', action='store',
                    default=1,type=int,
                    help='set the number of mpi ranks')

parser.add_argument('-n', dest='network', action='store',
                    default="resnet18",type=str,choices=["resnet18","resnet34","resnet50","resnet101","resnet110","resnet152","resnet164","resnet1001"],
                    help='set the network configuration')

parser.add_argument('-r', dest='repetition', action='store',
                    default=1,type=int,
                    help='set the repetition number for this experiment')

parser.add_argument('-b', dest='batchsize', action='store',
                    default=32,type=int,
                    help='set the batch size')

parser.add_argument('-e', dest='epochs', action='store',
                    default=2,type=int,
                    help='set the number of training epochs')

parser.add_argument('-nrparameters', dest='nrparameters', action='store',
                    default=1,type=int,choices=[1,2,3],
                    help='set the number of parameters considered for analysis')

parser.add_argument('-s', dest='problemsize', action='store',
                    default=200,type=int,
                    help='set the problem size, the number of classes to train the model for')

parser.add_argument('-verbose', dest='verbose', action='store',
                    default=1,type=int,choices=[0,1,2],
                    help='set tensorflow verbose mode option')

parser.add_argument('--tfprofiler', dest='tfprofiler', action='store_true', help='enable the tf-profiler', default=False)

parser.add_argument('--determinism', dest='determinism', action='store_true', help='enable determinism for tf operations', default=False)

parser.add_argument('--memorygrowth', dest='memory_growth', action='store_true', help='enable the memory growth for tf', default=False)

parser.add_argument('--mixedprecision', dest='mixed_precision', action='store_true', help='enable mixed precision computing', default=False)

parser.add_argument('--tensor32', dest='tensor32', action='store_true', help='enable tensor32 computation', default=False)

parser.add_argument('--mode', dest='mode', action='store', help='switch between running full epochs and only a few steps for sampling', default='nvtx', choices=['nvtx', 'all'])


@nvtx.annotate("main", color="black")
def main():

    # get the command line arguments
    args = parser.parse_args()
    NR_PROCESSES = args.ranks
    NETWORK = args.network
    REPETITION = args.repetition
    TF_PROFILER = args.tfprofiler
    EPOCHS = args.epochs
    BATCH_SIZE = args.batchsize
    NR_PARAMETERS = args.nrparameters
    VERBOSE = args.verbose
    DETERMINISM = args.determinism
    MEMORY_GROWTH = args.memory_growth
    MIXED_PRECISION = args.mixed_precision
    TENSOR32_EXECUTION = args.tensor32
    PROBLEM_SIZE = args.problemsize
    MODE = args.mode

    # initialize ImageNet benchmark
    benchmark = ImageNetBenchmark(REPETITION, NR_PROCESSES, NETWORK, EPOCHS, BATCH_SIZE, TF_PROFILER, NR_PARAMETERS, VERBOSE, DETERMINISM, MEMORY_GROWTH, MIXED_PRECISION, TENSOR32_EXECUTION, PROBLEM_SIZE, MODE)

    # load the data
    benchmark.load_data()

    # do the data preprocessing
    benchmark.preprocessing()

    # create callback for training
    benchmark.create_callbacks_training()

    # define the network for training
    benchmark.define_network()

    # train the model
    benchmark.train()

    # final evaluation of the model
    benchmark.evaluation()

    # log the experiment data
    if hvd.rank() == 0:
        benchmark.log_experiment_data()

if __name__ == '__main__':
    main()

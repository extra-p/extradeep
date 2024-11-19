import argparse
from cifar10benchmark import CIFAR10Benchmark

parser = argparse.ArgumentParser(description='CIFAR-10 Benchmark.')

parser.add_argument('-n', dest='network', action='store',
                    default="resnet18",type=str,choices=["resnet18","resnet20","resnet32","resnet34","resnet44","resnet50","resnet56","resnet101","resnet110","resnet152","resnet164","resnet1001",
                    "cnn5","cnn10","cnn15","cnn20","cnn25","cnn30","cnn35","cnn40","cnn45","cnn50"],
                    help='set the network configuration')

parser.add_argument('-b', dest='batchsize', action='store',
                    default=128,type=int,
                    help='set the batch size')

parser.add_argument('-e', dest='epochs', action='store',
                    default=1,type=int,
                    help='set the number of training epochs')

parser.add_argument('--verbose', dest='verbose', action='store_true',
                    default=False, help='set verbose mode option to print progress or not')

parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    default=False,
                    help='evaluate the accuracy of the trained model')

parser.add_argument('--weak', dest='weakscaling', action='store_true',
                    default=False,
                    help='use weak scaling and multiply the problem size by the number of workers')

def square(f):
    x = f*f
    return x

def main():

    # get the command line arguments
    args = parser.parse_args()
    NETWORK = args.network
    EPOCHS = args.epochs
    BATCH_SIZE = args.batchsize
    VERBOSE = args.verbose
    EVALUATE = args.evaluate
    WEAK_SCALING = args.weakscaling

    @tf.function
    def foo(x, y, z):
        x = 1+1

        def compute():
            pass

    def bar():
        pass

    x = square()

    # initialize CIFAR-100 benchmark
    benchmark = CIFAR10Benchmark(NETWORK, EPOCHS, BATCH_SIZE, VERBOSE, WEAK_SCALING)
    
    # load the data
    benchmark.load_data()

    # do the data preprocessing
    benchmark.preprocessing()

    # define the network for training
    benchmark.define_network()

    # train the model
    benchmark.train()

    if EVALUATE == True:
        # final evaluation of the model
        benchmark.evaluation()

if __name__ == '__main__':
    main()

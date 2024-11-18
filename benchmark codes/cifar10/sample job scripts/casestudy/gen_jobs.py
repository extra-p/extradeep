tasks = [4,8,12,16,20,24,28]
batch = [32,64,128,256,512,1024,2048]

for i in range(len(tasks)):

    for j in range(len(batch)):

        file = open("../../cifar10_p"+str(tasks[i])+"_b"+str(batch[j])+"_job.sh","w")

        text = "#!/bin/sh\n\n"
        text += "#SBATCH -J cifar10_p"+str(tasks[i])+"_b"+str(batch[j])+"\n"
        text += "#SBATCH --partition=dp-esb\n"
        text += "#SBATCH -A <replace>\n"
        text += "#SBATCH -N "+str(tasks[i])+"\n"
        text += "#SBATCH -n "+str(tasks[i])+"\n"
        text += "#SBATCH -o /<replace>/cifar10_benchmark/output_p"+str(tasks[i])+"_b"+str(batch[j])+".out\n"
        text += "#SBATCH -e /<replace>/cifar10_benchmark/error_p"+str(tasks[i])+"_b"+str(batch[j])+".er\n"
        text += "#SBATCH --time=00:30:00\n"
        text += "#SBATCH --gres=gpu:1\n"
        text += "\n"
        text += "module purge\n"
        text += "module restore extradeep\n"
        text += "ml -f unload nvidia-driver/.default\n"
        text += "\n"
        text += "export TMPDIR=/scratch\n"
        text += "export CUDA_VISIBLE_DEVICES=0\n"
        text += "\n"
        text += "PSP_OPENIB=1 PSP_UCP=0 srun /usr/local/software/skylake/Stages/2020/software/CUDA/11.3/bin/nsys profile -t nvtx,cuda,mpi,cublas,cudnn --mpi-impl mpich -b none --cpuctxsw none -f true -x true -o /p/project/deepsea/ritter2/cifar10_benchmark/cifar10.p"+str(tasks[i])+".b"+str(batch[j])+".r1.mpi%q{SLURM_PROCID} python -u main.py -p "+str(tasks[i])+" -b "+str(batch[j])+" -nrparameters 2 -r 1\n"
        file.write(text)

        file.close()

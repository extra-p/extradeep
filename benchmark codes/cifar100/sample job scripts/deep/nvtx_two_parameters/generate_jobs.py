tasks = [1,2,4,6,8,10,12,14,16]
nets = [18,34,50,101,110,164]

for i in range(len(tasks)):

    for j in range(len(nets)):

        file = open("../../../cifar100_p"+str(tasks[i])+"_n"+str(nets[j])+"_job_nvtx2.sh","w")

        text = "#!/bin/sh\n\n"
        text += "#SBATCH -J cifar100\n"
        text += "#SBATCH --partition=dp-esb\n"
        text += "#SBATCH -A <replace\n"
        text += "#SBATCH -N "+str(tasks[i])+"\n"
        text += "#SBATCH -n "+str(tasks[i])+"\n"
        text += "#SBATCH -o /<replace/cifar100-benchmark/nvtx2/output_p"+str(tasks[i])+"_n"+str(nets[j])+".out\n"
        text += "#SBATCH -e /<replace/cifar100-benchmark/nvtx2/error_p"+str(tasks[i])+"_n"+str(nets[j])+".er\n"
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
        text += "PSP_OPENIB=1 PSP_UCP=0 srun /usr/local/software/skylake/Stages/2020/software/CUDA/11.3/bin/nsys profile -t nvtx --mpi-impl mpich -b none --cpuctxsw none -f true -x true -o /p/project/deepsea/ritter2/cifar100-benchmark/nvtx2/cifar100.p"+str(tasks[i])+".n"+str(nets[j])+".r1.mpi%q{SLURM_PROCID} python -u main.py -p "+str(tasks[i])+" -r 1 -n resnet"+str(nets[j])+" -nrparameters 2 -e 2 -mode nvtx\n"
        file.write(text)

        file.close()

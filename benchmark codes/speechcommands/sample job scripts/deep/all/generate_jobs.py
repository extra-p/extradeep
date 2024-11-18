tasks = [1,2,4,6,8,10,12,14,16]

for i in range(len(tasks)):

    file = open("../../../speech"+str(tasks[i])+"_job.sh","w")

    text = "#!/bin/sh\n\n"
    text += "#SBATCH -J speech\n"
    text += "#SBATCH --partition=dp-esb\n"
    text += "#SBATCH -A <replace>\n"
    text += "#SBATCH -N "+str(tasks[i])+"\n"
    text += "#SBATCH -n "+str(tasks[i])+"\n"
    text += "#SBATCH -o /<replace>/speechcommands-benchmark/output_p"+str(tasks[i])+".out\n"
    text += "#SBATCH -e /<replace>/speechcommands-benchmark/error_p"+str(tasks[i])+".er\n"
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
    text += "PSP_OPENIB=1 PSP_UCP=0 srun /usr/local/software/skylake/Stages/2020/software/CUDA/11.3/bin/nsys profile -t nvtx,cuda,mpi,cublas,cudnn --mpi-impl mpich -b none --cpuctxsw none -f true -x true -o /p/project/deepsea/ritter2/speechcommands-benchmark/speech.p"+str(tasks[i])+".r1.mpi%q{SLURM_PROCID} python -u main.py -p "+str(tasks[i])+" -r 1\n"
    file.write(text)

    file.close()

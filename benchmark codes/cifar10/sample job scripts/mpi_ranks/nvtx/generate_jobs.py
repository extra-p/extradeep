nodes = [1,1,1,2,2,3,3,4,4]
tasks = [1,2,4,6,8,10,12,14,16]
gpu_per_node = [1,2,4,3,4,4,4,4,4]

for i in range(len(nodes)):

    file = open("../../cifar10_p"+str(tasks[i])+"_job.sh","w")

    text = "#!/bin/sh\n\n"
    text += "# Slurm job configuration\n"
    text += "#SBATCH --reservation=\"<replace>\""
    text += "#SBATCH -A \"<replace>\"\n"
    text += "#SBATCH -N "+str(nodes[i])+"\n"
    text += "#SBATCH -n "+str(tasks[i])+"\n"
    text += "#SBATCH -c 24\n"
    text += "#SBATCH --mem-per-cpu=3600\n"
    text += "#SBATCH -o output_p"+str(tasks[i])+".out\n"
    text += "#SBATCH -e error_p"+str(tasks[i])+".er\n"
    text += "#SBATCH --time=00:30:00\n"
    text += "#SBATCH -J cifar10\n"
    text += "#SBATCH --gres=gpu:v100:"+str(gpu_per_node[i])+"\n"
    text += "#SBATCH --mail-type=ALL\n"
    text += "#SBATCH --mail-user=<replace>\n"
    text += "#SBATCH --exclusive\n"
    text += "#SBATCH -C avx512\n"
    text += "\n"
    text += "# Load the required modules\n"
    text += "module purge\n"
    text += "module restore test\n"
    text += "\n"
    text += "# expose gpus\n"
    text += "export CUDA_VISIBLE_DEVICES=0,1,2,3\n"
    text += "\n"
    text += "# Run the program\n"
    text += "srun nsys profile -t nvtx --mpi-impl openmpi -b none --cpuctxsw none -f true -x true -o cifar10.p"+str(tasks[i])+".r1.mpi%q{SLURM_PROCID} python -u main.py -p "+str(tasks[i])+" -n resnet18 -r 1\n"

    file.write(text)

    file.close()

networks = [18,34,50,101,110,152,164]

for i in range(len(networks)):

    file = open("../../../cifar10_n"+str(networks[i])+"_job.sh","w")

    text = "#!/bin/sh\n\n"
    text += "# Slurm job configuration\n"
    text += "#SBATCH --reservation=\"<replace>\"\n"
    text += "#SBATCH -A \"<replace>\"\n"
    text += "#SBATCH -N 1\n"
    text += "#SBATCH -n 4\n"
    text += "#SBATCH -c 24\n"
    text += "#SBATCH --mem-per-cpu=3600\n"
    text += "#SBATCH -o output_n"+str(networks[i])+".out\n"
    text += "#SBATCH -e error_n"+str(networks[i])+".er\n"
    text += "#SBATCH --time=01:00:00\n"
    text += "#SBATCH -J cifar10\n"
    text += "#SBATCH --gres=gpu:v100:4\n"
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
    text += "srun nsys profile -t nvtx --mpi-impl openmpi -b none --cpuctxsw none -f true -x true -o cifar10.n"+str(networks[i])+".r1.mpi%q{SLURM_PROCID} python -u main.py -p 4 -n resnet"+str(networks[i])+" -r 1 -nrparameters 2\n"

    file.write(text)

    file.close()

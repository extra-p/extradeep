nodes = [1,1,2,2,3,3,4,4]
tasks = [2,4,6,8,10,12,14,16]
gpu_per_node = [2,4,3,4,4,4,4,4]
networks = ["resnet18","resnet34","resnet50","resnet101","resnet110","resnet152","resnet164"]
nets = [18,34,50,101,110,152,164]
problem_size = [10,20,30,40,50,60,70]

file = open("../../submit_jobs.sh","w")

for i in range(len(nodes)):

    for j in range(len(nets)):

        for k in range(len(problem_size)):

            text = "echo Submitting job script cifar100 with mpi ranks "+str(tasks[i])+" and network "+str(networks[j])+".\n"
            text += "sbatch cifar100_p"+str(tasks[i])+"_n"+str(nets[j])+"_s"+str(problem_size[k])+"_job.sh\n"

            file.write(text)

file.close()

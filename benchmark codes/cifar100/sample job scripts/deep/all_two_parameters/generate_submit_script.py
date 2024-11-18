tasks = [1,2,4,6,8,10,12,14,16]
nets = [18,34,50,101,110,164]

file = open("../../../submit_jobs_all2.sh","w")

for i in range(len(tasks)):

    for j in range(len(nets)):

        text = "echo Submitting job script cifar100 with mpi ranks "+str(tasks[i])+".\n"
        text += "sbatch cifar100_p"+str(tasks[i])+"_n"+str(nets[j])+"_job_all2.sh\n"

        file.write(text)

file.close()

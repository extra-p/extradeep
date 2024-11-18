tasks = [1,2,4,6,8,10,12,14,16]
nets = [18,34,50,101,110,164]
batches = [8,16,32,64,128,256,512,1024]

file = open("../../../submit_jobs_all3.sh","w")

for i in range(len(tasks)):

    for j in range(len(nets)):

        for k in range(len(batches)):

            text = "echo Submitting job script cifar100 with mpi ranks "+str(tasks[i])+".\n"
            text += "sbatch cifar100_p"+str(tasks[i])+"_n"+str(nets[j])+"_b"+str(batches[k])+"_job_all3.sh\n"

            file.write(text)

file.close()

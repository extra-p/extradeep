networks = [18,34,50,101,110,152,164]

file = open("../../../submit_jobs.sh","w")

for i in range(len(networks)):

    text = "echo Submitting job script cifar10 with network "+str(networks[i])+".\n"
    text += "sbatch cifar10_n"+str(networks[i])+"_job.sh\n"

    file.write(text)

file.close()

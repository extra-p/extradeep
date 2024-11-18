nodes = [1,1,1,2,2,3,3,4,4]
tasks = [1,2,4,6,8,10,12,14,16]
gpu_per_node = [1,2,4,3,4,4,4,4,4]

file = open("../../submit_jobs.sh","w")

for i in range(len(nodes)):

    text = "echo Submitting job script speech with mpi ranks "+str(tasks[i])+".\n"
    text += "sbatch speech_p"+str(tasks[i])+"_job.sh\n"

    file.write(text)

file.close()

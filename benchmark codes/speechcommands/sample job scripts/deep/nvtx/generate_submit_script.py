tasks = [1,2,4,6,8,10,12,14,16]

file = open("../../../submit_jobs.sh","w")

for i in range(len(tasks)):

    text = "echo Submitting job script speech with mpi ranks "+str(tasks[i])+".\n"
    text += "sbatch speech_p"+str(tasks[i])+"_job.sh\n"

    file.write(text)

file.close()

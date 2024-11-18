tasks = [1,2,4,6,8,10,12,14,16]

file = open("../../../convert2.sh","w")

for i in range(len(tasks)):

    for j in range(tasks[i]):

        text = "echo Converting qdrep from speech.p"+str(tasks[i])+".r1.mpi"+str(j)+".qdrep to .sqlite\n"
        text += "nsys export --type sqlite --force-overwrite true speech.p"+str(tasks[i])+".r1.mpi"+str(j)+".qdrep\n"

        file.write(text)

file.close()

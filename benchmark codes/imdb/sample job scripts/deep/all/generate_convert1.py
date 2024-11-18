tasks = [1,2,4,6,8,10,12,14,16]

file = open("../../../convert1.sh","w")

for i in range(len(tasks)):

    for j in range(tasks[i]):

        text = "echo Converting qdstrm from imdb.p"+str(tasks[i])+".r1.mpi"+str(j)+".qdstrm to .qdrep\n"
        text += "/usr/local/software/skylake/Stages/2020/software/CUDA/11.3/nsight-systems-2021.1.3/host-linux-x64/QdstrmImporter -i imdb.p"+str(tasks[i])+".r1.mpi"+str(j)+".qdstrm\n"

        file.write(text)

file.close()

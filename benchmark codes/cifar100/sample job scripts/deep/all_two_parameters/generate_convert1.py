tasks = [1,2,4,6,8,10,12,14,16]
nets = [18,34,50,101,110,164]

file = open("../../../convert1.sh","w")

for i in range(len(tasks)):

    for j in range(tasks[i]):

        for l in range(len(nets)):

            text = "echo Converting qdstrm from cifar100.p"+str(tasks[i])+".r1.mpi"+str(j)+".qdstrm to .qdrep\n"
            text += "/usr/local/software/skylake/Stages/2020/software/CUDA/11.3/nsight-systems-2021.1.3/host-linux-x64/QdstrmImporter -i cifar100.p"+str(tasks[i])+".n"+str(nets[l])+".r1.mpi"+str(j)+".qdstrm\n"

            file.write(text)

file.close()

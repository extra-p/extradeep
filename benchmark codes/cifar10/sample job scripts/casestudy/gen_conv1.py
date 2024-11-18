tasks = [4,8,12,16,20,24,28]
batch = [32,64,128,256,512,1024,2048]

file = open("../../../convert1.sh","w")

for i in range(len(tasks)):

    for k in range(len(batch)):

        for j in range(tasks[i]):

            text = "echo Converting qdstrm from cifar10.p"+str(tasks[i])+".b"+str(batch[k])+".r1.mpi"+str(j)+".qdstrm to .qdrep\n"
            text += "/usr/local/software/skylake/Stages/2020/software/CUDA/11.3/nsight-systems-2021.1.3/host-linux-x64/QdstrmImporter -i cifar10.p"+str(tasks[i])+".b"+str(batch[k])+".r1.mpi"+str(j)+".qdstrm\n"


            file.write(text)

file.close()

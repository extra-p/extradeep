tasks = [4,8,12,16,20,24,28]
batch = [32,64,128,256,512,1024,2048]

file = open("../../../convert2.sh","w")

for i in range(len(tasks)):

    for k in range(len(batch)):

        for j in range(tasks[i]):

            text = "echo Converting qdrep from cifar10.p"+str(tasks[i])+".b"+str(batch[k])+".r1.mpi"+str(j)+".qdrep to .sqlite\n"
            text += "nsys export --type sqlite --force-overwrite true cifar10.p"+str(tasks[i])+".b"+str(batch[k])+".r1.mpi"+str(j)+".qdrep\n"

            file.write(text)

file.close()

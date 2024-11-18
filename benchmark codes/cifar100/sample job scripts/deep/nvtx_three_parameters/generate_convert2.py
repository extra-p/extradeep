tasks = [1,2,4,6,8,10,12,14,16]
nets = [18,34,50,101,110,164]
batches = [8,16,32,64,128,256,512,1024]

file = open("../../../convert2.sh","w")

for i in range(len(tasks)):

    for j in range(tasks[i]):

        for l in range(len(nets)):

            for k in range(len(batches)):

                text = "echo Converting qdrep from cifar100.p"+str(tasks[i])+".r1.mpi"+str(j)+".qdrep to .sqlite\n"
                text += "nsys export --type sqlite --force-overwrite true cifar100.p"+str(tasks[i])+".n"+str(nets[l])+".b"+str(batches[k])+".r1.mpi"+str(j)+".qdrep\n"

                file.write(text)

file.close()

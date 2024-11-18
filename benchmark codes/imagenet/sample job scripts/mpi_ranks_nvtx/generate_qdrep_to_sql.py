nodes = [1,1,2,2,3,3,4,4]
tasks = [2,4,6,8,10,12,14,16]
gpu_per_node = [2,4,3,4,4,4,4,4]
repetition = [1,2,3,4,5]

#nsys export --type sqlite cifar100.r10.r1.qdrep

file = open("../../qdrep_to_sql.sh","w")

text = ""

for i in range(len(tasks)):
    for j in range(len(repetition)):
        for k in range(tasks[i]):

            text += "nsys export --type sqlite imagenet.p"+str(tasks[i])+".r"+str(repetition[j])+".mpi"+str(k)+".nsys-rep\n"


file.write(text)

file.close()

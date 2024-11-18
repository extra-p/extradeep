nodes = [1,1,2,2,3,3,4,4]
tasks = [2,4,6,8,10,12,14,16]
gpu_per_node = [2,4,3,4,4,4,4,4]
repetition = [1,2,3,4,5]
networks = ["resnet18","resnet34","resnet50","resnet101","resnet110","resnet152","resnet164"]
nets = [18,34,50,101,110,152,164]
problem_size = [10,20,30,40,50,60,70]

#nsys export --type sqlite cifar100.r10.r1.qdrep

file = open("../../qdrep_to_sql.sh","w")

text = ""

for i in range(len(tasks)):
    for l in range(len(nets)):
        for p in range(len(problem_size)):
            for j in range(len(repetition)):
                for k in range(tasks[i]):

                    text += "nsys export --type sqlite cifar100.p"+str(tasks[i])+".n"+str(nets[l])+".s"+str(problem_size[p])+".r"+str(repetition[j])+".mpi"+str(k)+".nsys-rep\n"

file.write(text)

file.close()

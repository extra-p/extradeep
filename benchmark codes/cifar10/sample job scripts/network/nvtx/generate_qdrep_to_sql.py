networks = [18,34,50,101,110,152,164]
repetition = [1,2,3,4,5]

#nsys export --type sqlite cifar100.r10.r1.qdrep

file = open("../../../qdrep_to_sql.sh","w")

text = ""

for i in range(len(networks)):
    for j in range(len(repetition)):
        for k in range(4):

            text += "nsys export --type sqlite cifar10.n"+str(networks[i])+".r"+str(repetition[j])+".mpi"+str(k)+".nsys-rep\n"


file.write(text)

file.close()

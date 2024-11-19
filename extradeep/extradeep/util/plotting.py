import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_epochs_runtime(data, eval_experiment):

    labels = []
    for count, value in enumerate(data):
        labels.append(str(value))

    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars
    spacer = 0.05

    fig, ax = plt.subplots()

    facecolors = ["orange", "blue"]
    colors = ["black", "black"]

    type = eval_experiment.analysistypes[0]
    callpath_list = eval_experiment.callpaths[type]

    legends = []
    for j in range(len(eval_experiment.coordinates)):
        for i in range(len(callpath_list)):
            if j == 0:
                legends.append(str(callpath_list[i].name))
            for l in range(len(eval_experiment.metrics)):
                pass

    values = []
    for count, value in enumerate(data):
        callpaths = []
        for key, value2 in enumerate(data[value]):
            for count3, value3 in enumerate(data[value][value2]):
                callpaths.append(data[value][value2][value3])
        values.append(callpaths)

    temp = []
    for i in range(len(values)):
        for j in range(len(values[i])):
            temp.append([])

    for i in range(len(values)):
        for j in range(len(values[i])):
            temp[j].append(values[i][j])

    rects1 = ax.bar(x, temp[0], width, label=legends[0])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Median Prediction Error [%]')
    ax.set_xlabel('Evaluation Point')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ymax = max(temp[0]) + (((max(temp[0]))/100)*10)
    #print(ymax)
    #ax.set_ylim([0, ymax])
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    title_string = "Prediction power at evaluation points\n"

    plt.title(title_string, fontsize=12)

    # adding horizontal grid lines
    ax.yaxis.grid(True)

    plt.subplots_adjust(bottom=0.15)

    def autolabel(rects):
        #Attach a text label above each bar in *rects*, displaying its height
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.3f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', weight="bold", fontsize=10)

    autolabel(rects1)
    plt.ylim(0,ymax)
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()

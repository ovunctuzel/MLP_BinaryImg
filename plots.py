import visualizations
import numpy as np
import matplotlib.pyplot as plt


class data(object):
    def __init__(self, l, b, h, m, p):
        self.lrate = l
        self.batch = b
        self.hidden = h
        self.momentum = m
        self.performance = p

def get_perf(dataset, sample):
    m, b, h, l = sample
    for data in dataset:
        if data.momentum == m and data.batch == b and data.hidden == h and data.lrate == l:
            return data.performance

def bar_graph(N, y1, y2, y3, y4):
    fig, ax = plt.subplots(figsize=(10, 5))

    ind = np.arange(N)  # the x locations for the groups
    width = 0.2  # the width of the bars
    p1 = ax.bar(ind, y1, width, color="#223388", bottom=0)
    p2 = ax.bar(ind + width, y2, width, color="#d9e888", bottom=0)
    p3 = ax.bar(ind + 2*width, y3, width, color="#22cc88", bottom=0)
    p4 = ax.bar(ind + 3*width, y4, width, color="#e74c3c", bottom=0)

    ax.set_title('Test Success Rate - Batch Size = 1')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('3', '12', '48', '192'))
    ax.set_xlabel(('Hidden Units'))


    ax.legend((p1[0], p2[0], p3[0], p4[0]), ('LR=10', 'LR=0.01', 'LR=0.001', 'LR=0.00001'))
    ax.autoscale_view()

    plt.ylim([0, 1])
    plt.show()



hidden_units = [3, 12, 48, 192]
batch_size = [1, 4, 16, 64]
learning_rate = [10, 0.01, 0.001, 0.00001]

dataset = []
lst = list(open('Results/results.txt'))
for i in range(len(lst)):
    r = lst[i].split(",")
    dataset.append(data(float(r[2]), float(r[1]), float(r[3]), float(r[0]), float(r[4])))

b = 64
p1 = [get_perf(dataset, (0.2, b, i, 10)) for i in hidden_units]
p2 = [get_perf(dataset, (0.2, b, i, 0.01)) for i in hidden_units]
p3 = [get_perf(dataset, (0.2, b, i, 0.001)) for i in hidden_units]
p4 = [get_perf(dataset, (0.2, b, i, 0.00001)) for i in hidden_units]
print p1
print p2
print p3
print p4
bar_graph(len(p1), p1, p2, p3, p4)




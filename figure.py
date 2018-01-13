import numpy
import matplotlib.pyplot as plt

###############
# Two sinusoids
###############

# x1 = numpy.linspace(1, 10, 1000)
# y1 = numpy.cos(2*x1-2)
# plt.plot(x1, y1)

# x2 = numpy.linspace(1, 10, 1000)
# y2 = numpy.cos(2*x2-1)
# plt.plot(x2, y2)

# x3 = numpy.linspace(1, 10, 1000)
# y3 = [0 for dummy_x in x3]
# plt.plot(x3, y3)

# plt.show()

# x1 = [-5, -4, -3, 3, 4, 5]
# x2 = [-2, -1, 0, 1 ,2]
# y1_original = [0, 0, 0, 0, 0, 0]
# y2_original = [0, 0, 0, 0, 0]

# y1_projeted = [5, 4, 3, 3, 4,  5]
# y2_projeted = [2, 1, 0, 1, 2]

# plt.plot(x1, y1_original, 'o' + 'b')
# plt.plot(x2, y2_original, '+' + 'r')
# plt.show()

# plt.plot(x1, y1_projeted, 'o' + 'b')
# plt.plot(x2, y2_projeted, '+' + 'r')
# plt.show()

###################
# Original clusters
###################

f = open('result/data/cinq_classes1/cinq_classes1.txt', 'r')
Data = []
for line in f:
    line = line.split('\t')
    line = [line[i] for i in range(len(line)) if line[i] != '\n']
    line = [float(line[i]) for i in range(len(line))]
    Data.append(line)
f.close()

plt.figure()
x = [Data[i][0] for i in range(len(Data))]
y = [Data[i][1] for i in range(len(Data))]
plt.plot(x, y, '.', color = 'black')
plt.show()

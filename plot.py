from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

name = "Full_brake"
my_data0 = genfromtxt("difplots/0.csv", delimiter=',')
my_data1 = genfromtxt("difplots/1.csv", delimiter=',')
my_data2 = genfromtxt("difplots/2.csv", delimiter=',')
my_data3 = genfromtxt("difplots/3.csv", delimiter=',')
my_data4 = genfromtxt("difplots/4.csv", delimiter=',')
data = [my_data0, my_data1, my_data2, my_data3, my_data4]

plt.boxplot([my_data0[:,0],my_data1[:,0],my_data2[:,0],my_data3[:,0],my_data4[:,0]])
plt.xlabel("Action")
plt.ylabel("Distance (m)")
plt.title("1 Difference in distance between prediction and actual situation")
plt.show()
plt.boxplot([my_data0[:,1],my_data1[:,1],my_data2[:,1],my_data3[:,1],my_data4[:,1] ])
plt.xlabel("Action")
plt.ylabel("Distance (m)")
plt.title("2 Difference in speed between prediction and actual situation")
plt.show()
plt.boxplot([my_data0[:,2],my_data1[:,2],my_data2[:,2],my_data3[:,2],my_data4[:,2] ])
plt.xlabel("Action")
plt.ylabel("Speed")
plt.title("Difference in acceleration between prediction and actual situation")
plt.show()


#for set in data:
#    x = set[:, 0]
#    y = set[:, 3]
#    plt.scatter(x, y)
#    z = np.polyfit(x, y, 2)
#    p = np.poly1d(z)
#    print(p)
#    plt.plot(np.sort(x), p(np.sort(x)), 'r--')
#     plt.show()
# the line equation:


#plt.boxplot([distance, speed, acc])
#plt.show()
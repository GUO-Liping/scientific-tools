import numpy as np
import matplotlib.pyplot as plt

num_pieces = 10
num_points = 100
t  = np.zeros((num_pieces,num_points))
ft = np.zeros((num_pieces,num_points))
delta_t = 2*np.pi/(num_pieces+1)
for i in range(num_pieces):
    sum = np.zeros((1,num_points))
    t[i] = np.linspace(i*delta_t, (i+1)*delta_t, num_points)
    for j in range(i+1):
        sum = sum + np.sin(t[i]-j*delta_t)
    ft[i] = sum

    print(t)
    print(ft)
    plt.plot(t[i], ft[i],'-*')
    plt.xlabel('Time (s)')
    plt.ylabel('Function value')
    plt.title('Function value over time')
plt.show()

import numpy as np
from matplotlib import animation as animation, pyplot as plt, cm
plt.rcParams["figure.figsize"] = [10, 5]
plt.rcParams["figure.autolayout"] = True
BAR_PLOT = plt.figure()

num_elements = 100

def make_list():
       b_list = []
       for i in range(num_elements):
              b_list.append(i+1)
       return b_list
def rand_data():
       rand_list = []
       for i in range(num_elements):
              n = np.random.randint(1,num_elements+1)
              rand_list.append(n)
       return rand_list
data = make_list()
data2 = rand_data()
def bubble_sort(s_data):
       for i in range(0,num_elements-1):
              if s_data[i+1] > s_data[i]:
                     s_data[i], s_data[i+1] = s_data[i+1], s_data[i]
       data2 = s_data
       return data2
def insertion_sort(s_data):
       for i in range(1, num_elements):
              key_item = s_data[i]
              j = i - 1
              while j >= 0 and s_data[j] > key_item:
                     s_data[j + 1] = s_data[j]
                     j -= 1
              s_data[j + 1] = key_item
       data2 = s_data
       return data2
def animate(frame):
       global bars
       list_data = bubble_sort(data2)
       for i in range(num_elements):
              bars[i].set_height(list_data[i])                
plt.yticks(np.arange(0, num_elements+1, num_elements/10))
plt.xticks(np.arange(0, num_elements+1, num_elements/10))
plt.xlim(0,num_elements+1)
plt.ylim(0,num_elements+1)
bars = plt.bar(x=data, height=data, width=0.8, color='black')
ani = animation.FuncAnimation(BAR_PLOT, animate, frames=None, interval=1)
plt.show()
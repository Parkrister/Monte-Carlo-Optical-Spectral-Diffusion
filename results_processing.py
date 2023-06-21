import numpy as np
import matplotlib.pyplot as plt
import random
import math
import json
import os

directory = 'results/'
for filename in os.listdir(directory):
  f = os.path.join(directory, filename)
  if os.path.isfile(f) and filename.endswith('.txt'):
    with open(filename, 'r') as fr:
      weight_mas = json.load(fr)
      n = 100
    w_mas_copy = weight_mas.copy()
    mid_distance = [0] * (n + 1)
    for distance in range(len(w_mas_copy)):
      sum = 0  # сумма всех весов на расстонии i
      sum_d = 0  # значение глубины * вес
      for i in range(len(w_mas_copy[distance])):
        if w_mas_copy[distance][i] > 0:
          sum_d += i * w_mas_copy[distance][i]
          sum += w_mas_copy[distance][i]
      if sum > 0:
        result = sum_d / sum
        mid_distance[distance] = result

    # усредненная половина графика
    mid_distance_half = [0] * (51)
    for i in range(len(mid_distance_half)):
      mid_distance_half[i] = (mid_distance[i] + mid_distance[n - i]) / 2

    plt.minorticks_on()
    plt.bar(np.arange(len(mid_distance_half)), mid_distance_half)
    plt.xlabel('x мм')
    plt.ylabel('medium z мм')
    x = np.arange(51)
    #plift = np.polyfit(x, mid_distance_half, 2)
    #print(filename)
    p = np.poly1d(np.polyfit(x, mid_distance_half, 2))
    print(filename, ': ', p, '\n')
    t = np.linspace(0, 51, 50)
    #plt.plot(p(t), '-', color='red')
    save_name = 'result_approximate/' + filename + '_approximate.png'
    #plt.savefig(save_name)
    #plt.show()

'''
file_name = 'test.txt'
with open(file_name, 'r') as fr:
    weight_mas = json.load(fr)

n = 100


weight_mas_transpose = [0]*(n+1)

for i in range(n+1):
    weight_mas_transpose[i] = [-9]*(n+1)

for i in range(len(weight_mas)):
  for j in range(len(weight_mas[i])):
    if weight_mas[i][j] > 0:
      weight_mas_transpose[j][i] = weight_mas[i][j]

w_mas_log = weight_mas_transpose.copy()

for i in range(len(w_mas_log)):
  for j in range(len(w_mas_log[i])):
    if w_mas_log[i][j] > 0:
      w_mas_log[i][j] = math.log(w_mas_log[i][j])

plt.pcolormesh(np.linspace(0, 101, 101), np.linspace(0, 101, 101), w_mas_log)
clb = plt.colorbar()
plt.minorticks_on()
clb.ax.set_title('log() вес')
plt.xlabel('x мм')
plt.ylabel('max z мм')
plt.show()


w_mas_copy = weight_mas.copy()
mid_distance = [0]*(n+1)
for distance in range(len(w_mas_copy)):
  sum = 0 # сумма всех весов на расстонии i
  sum_d = 0 # значение глубины * вес
  for i in range(len(w_mas_copy[distance])):
    if w_mas_copy[distance][i] > 0:
      sum_d += i*w_mas_copy[distance][i]
      sum += w_mas_copy[distance][i]
  if sum > 0:
    result = sum_d/sum
    mid_distance[distance] = result

# усредненная половина графика
mid_distance_half = [0]*(51)
for i in range(len(mid_distance_half)):
    mid_distance_half[i] = (mid_distance[i]+mid_distance[n - i])/2


plt.minorticks_on()
plt.bar(np.arange(len(mid_distance_half)), mid_distance_half)
plt.xlabel('x мм')
plt.ylabel('medium z мм')
x = np.arange(51)
print(len(x), ' ', len(mid_distance_half))
p = np.poly1d(np.polyfit(x, mid_distance_half,3))
t = np.linspace(0,51,50)
plt.plot(p(t), '-', color = 'red')
plt.show()
'''
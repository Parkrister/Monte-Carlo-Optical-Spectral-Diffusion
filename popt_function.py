import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import json
import os

'''
mu_a=0.01, mu_s=10.0, g=0.9.txt :             2
-0.007209 x + 0.03714 x + 35.71 

mu_a=0.01, mu_s=20.0, g=0.9.txt :             2
-0.006295 x - 0.02136 x + 31.21 

mu_a=0.01, mu_s=30.0, g=0.9.txt :            2
-0.00653 x - 0.02611 x + 29.67 

mu_a=0.05, mu_s=10.0, g=0.9.txt :             2
-0.005723 x - 0.01662 x + 27.19 

mu_a=0.05, mu_s=20.0, g=0.9.txt :             2
-0.004634 x - 0.06585 x + 23.59 

mu_a=0.05, mu_s=30.0, g=0.9.txt :             2
-0.004491 x - 0.07716 x + 22.11 

mu_a=0.1, mu_s=10.0, g=0.9.txt :             2
-0.004918 x - 0.03486 x + 22.7 

mu_a=0.1, mu_s=20.0, g=0.9.txt :             2
-0.003655 x - 0.08752 x + 19.95 

mu_a=0.1, mu_s=30.0, g=0.9.txt :             2
-0.003502 x - 0.09622 x + 18.78 


Process finished with exit code 0

'''

mas_coef = [0]*9
'''
mas_coef[0] = -0.007209
mas_coef[1] = -0.006295
mas_coef[2] = -0.00653
mas_coef[3] = -0.005723
mas_coef[4] = -0.004634
mas_coef[5] = -0.004491
mas_coef[6] = -0.004918
mas_coef[7] = -0.003655
mas_coef[8] = -0.003502
'''
mas_coef[0] = 0.03714
mas_coef[1] = -0.02136
mas_coef[2] = -0.02611
mas_coef[3] = -0.01662
mas_coef[4] = -0.06585
mas_coef[5] = -0.07716
mas_coef[6] = -0.03486
mas_coef[7] = -0.08752
mas_coef[8] = -0.09622
'''

x_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8]
x_labels = [r'$\mu a=0.01$, $\mu s=10.0$',
            r'$\mu a=0.01$, $\mu s=20.0$',
            r'$\mu a=0.01$, $\mu s=30.0$',
            r'$\mu a=0.05$, $\mu s=10.0$',
            r'$\mu a=0.05$, $\mu s=20.0$',
            r'$\mu a=0.05$, $\mu s=30.0$',
            r'$\mu a=0.1$, $\mu s=10.0$',
            r'$\mu a=0.1$, $\mu s=20.0$',
            r'$\mu a=0.1$, $\mu s=30.0$']
plt.yticks (ticks=x_ticks, labels=x_labels)
plt.minorticks_on()
plt.xlabel('коэффицент аппроксимирующей функции')
plt.scatter(mas_coef, np.arange(9))
plt.show()
'''
## a
'''
mua_001 = [-0.007209, -0.006295, -0.00653]
mua_005 = [-0.005723, -0.004634, -0.004491]
mua_01 = [-0.004918, -0.003655, -0.003502]
'''
''' 
## b
mua_001 = [0.03714, -0.02136, -0.02611]
mua_005 = [-0.01662, -0.06585, -0.07716]
mua_01 = [-0.03486, -0.08752, -0.09622]
'''
## c
mua_001 = [35.71, 31.21, 29.67]
mua_005 = [27.19, 23.59, 22.11 ]
mua_01 = [22.7, 19.95, 18.78 ]

x_ticks = [0, 1, 2]
x_labels = ['10', '20', '30']
plt.xticks (ticks=x_ticks, labels=x_labels)

plt.plot(np.arange(3), mua_001, ':b', label=r'$\mu a=0.01$')
plt.plot(np.arange(3), mua_005, '--r', label=r'$\mu a=0.05$')
plt.plot(np.arange(3), mua_01, 'k', label=r'$\mu a=0.1$')

plt.xlabel(r'$\mu s$')
#plt.title('коэффицент аппроксимирующей функции')

plt.legend(fontsize=14)
plt.show()
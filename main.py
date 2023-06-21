import numpy as np
import matplotlib.pyplot as plt
import random
import math
import json

COSZERO = 1.0 - 1.0e-12
CHANCE = 0.1
WEIGHT = 1e-4

class PhotonStruct:
    def __init__(self, x, y, z, ux, uy, uz, w, dead, layer, s, sleft):
        self.x = x # Cartesian coordinates.[cm]
        self.y = y
        self.z = z
        self.ux = ux # directional cosines of a photon
        self.uy = uy
        self.uz = uz
        self.w = w # weight
        self.dead = dead #  1 if photon is terminated
        self.layer = layer # index to layer where the photon packet resides
        self.s = s # current step size. [cm]
        self.sleft = sleft # step size left. dimensionless [-]

class LayerStruct:
    def __init__(self, z0, z1, mu_a, mu_s, g):
        self.z0 = z0  # z coordinates of a layer [cm]
        self.z1 = z1
        self.n = None  # refractive index of a layer
        self.mu_a = mu_a  # absorption coefficient [1/cm]
        self.mu_s = mu_s  # scattering coefficient [1/cm]
        self.g = g  # anisotropy
        self.cos_crit0 = None
        self.cos_crit1 = None

def Rspecular(n1, n2, mu_a, mu_s):
  temp = (n1 - n2)/(n1 + n2)
  r1 = temp**2
  return r1

def Spin_theta(g):
  if g == 0:
    return 2*random.uniform(0,1) - 1
  else:
    temp = (1-g**2)/(1 - g + 2*g*random.uniform(0,1))
    return (1 + g**2 - temp**2)/(2*g)

def Spin(g, photon):
  ux = photon.ux
  uy = photon.uy
  uz = photon.uz
  cost = Spin_theta(g)
  sint = math.sqrt(1 - cost**2)
  psi = 2*np.pi*random.uniform(0,1)
  cosp = math.cos(psi)
  if(psi < np.pi):
    sinp = math.sqrt(1.0 - cosp**2)
  else:
    sinp = -math.sqrt(1.0 - cosp**2)
  if(math.fabs(photon.uz > COSZERO)):
    photon.ux = sint*cosp
    photon.uy = sint*sinp
    photon.uz = cost*np.sign(uz)
  else:
    temp = math.sqrt(1.0 - photon.uz**2)
    photon.ux = sint*(ux*uz*cosp - uy*sinp)/temp + ux*cost
    photon.uy = sint*(uy*uz*cosp + ux*sinp)/temp + uy*cost
    photon.uz = -sint*cosp*temp + uz*cost

def Hop(photon): # Move the photon s away in the current layer of medium.
  s = photon.s
  photon.x += s*photon.ux
  photon.y += s*photon.uy
  photon.z += s*photon.uz
  if abs(photon.x) > 5:
    if np.random.uniform(0,1) > 0.5:
      photon.dead = True
    else:
      photon.ux *= - 1

def StepSize(photon, layer):
  mu_t = layer.mu_a + layer.mu_s
  if photon.sleft == 0:
    photon.s = - np.log(np.random.uniform(0,1)) / mu_t
  else:
    photon.s = photon.sleft / mu_t
    photon.sleft = 0


def HitBoundary(photon, layer):
    dl_b = 0
    if photon.uz > 0:
        dl_b = (layer.z1 - photon.z) / photon.uz
    elif photon.uz < 0:
        dl_b = (layer.z0 - photon.z) / photon.uz

    if photon.uz != 0 and photon.s > dl_b:
        mu_t = layer.mu_a + layer.mu_s
        photon.sleft = (photon.s - dl_b) * mu_t
        photon.s = dl_b
        return True
    else:
        return False

def Drop(photon, layer):
  #update photon weight
  dwa = photon.w * layer.mu_a/(layer.mu_a + layer.mu_s)
  photon.w -= dwa

def Roulette(photon):
  if photon.w == 0:
    photon.dead = 1
  else:
    if np.random.uniform(0,1) < CHANCE:
      photon.w /= CHANCE
    else:
      photon.dead = 1

def HopDropSpin(photon, layer):
  StepSize(photon, layer)
  if HitBoundary(photon, layer):
    Hop(photon)
    photon.uz *= -1
    if np.random.uniform(0,1) < 0.03:
      photon.dead = 1
  else:
    Hop(photon)
    Drop(photon, layer)
    Spin(layer.g, photon)
  if photon.w < WEIGHT and photon.dead != 1:
    Roulette(photon)

#########
# колличество фотонов
N = 400000

# шаг
n = 100
my_mu_a = float(input('введите mu_a: '))
my_mu_s = float(input('введите mu_s: '))
name = f'mu_a={my_mu_a}, mu_s={my_mu_s}, g=0.9'
layer = LayerStruct(z0=0, z1=10, mu_a=my_mu_a, mu_s=my_mu_s, g=0.95)
a = -5
b = 5

weight_mas = [0] * (n + 1)
for i in range(n + 1):
    weight_mas[i] = [0] * (n + 1)

max_depth = [0] * N
###------------- mu_s = 30, mu_a = 0.01, g = 0.9 -------------###

def HitBoundary_v2(photon, layer, i):
    dl_b = 0
    hit_floor = False
    if photon.uz > 0:
        dl_b = (layer.z1 - photon.z) / photon.uz
    elif photon.uz < 0:
        hit_floor = True
        dl_b = (layer.z0 - photon.z) / photon.uz

    if photon.uz != 0 and photon.s > dl_b:
        if hit_floor == True:
            try:
                x = math.floor(((photon.x - a) / (b - a)) * n)
                w = math.floor(((max_depth[i] - layer.z0) / (layer.z1 - layer.z0)) * n)
                weight_mas[x][w] += photon.w
            except:
                print(photon.x, i)
        mu_t = layer.mu_a + layer.mu_s
        photon.sleft = (photon.s - dl_b) * mu_t
        photon.s = dl_b
        return True
    else:
        return False

def HopDropSpin_v2(photon, layer, i):
  StepSize(photon, layer)
  if HitBoundary_v2(photon, layer, i):
    Hop(photon)
    photon.uz *= -1
    if np.random.uniform(0,1) < 0.03:
      photon.dead = 1
  else:
    Hop(photon)
    Drop(photon, layer)
    Spin(layer.g, photon)
  if photon.w < WEIGHT and photon.dead != 1:
    Roulette(photon)

###------------- mu_s = 30, mu_a = 0.01, g = 0.9 -------------###

for i in range(N):
    x_list = [0, ]
    z_list = [0, ]
    weights = [1, ]
    photon = PhotonStruct(x=0, y=0, z=0, ux=0, uy=0, uz=1, w=1, dead=0, s=0, sleft=0, layer=1)
    while photon.dead == 0:
        HopDropSpin_v2(photon, layer, i)
        if photon.z > max_depth[i]:
            max_depth[i] = photon.z

# сохраняем массив weight_mas в файл
file_name = name + '.txt'
with open(file_name, 'w') as f:
    f.write(str(weight_mas))


weight_mas_transpose = [0]*(n+1)
w_mas_copy = [0]*(n+1)

for i in range(n+1):
    w_mas_copy[i] = [0] * (n + 1)
    weight_mas_transpose[i] = [-9]*(n+1)

for i in range(len(weight_mas)):
  for j in range(len(weight_mas[i])):
    if weight_mas[i][j] > 0:
      weight_mas_transpose[j][i] = weight_mas[i][j]
      w_mas_copy[i][j] = weight_mas[i][j]

w_mas_log = weight_mas_transpose.copy()

for i in range(len(w_mas_log)):
  for j in range(len(w_mas_log[i])):
    if(w_mas_log[i][j] > 0):
      w_mas_log[i][j] = math.log(w_mas_log[i][j])


plt.pcolormesh(np.linspace(0, 101, 101), np.linspace(0, 101, 101), w_mas_log)
clb = plt.colorbar()
plt.minorticks_on()
clb.ax.set_title('log() вес')
plt.xlabel('x мм')
plt.ylabel('max z мм')
save_name = name + '.png'
plt.savefig(save_name)
plt.show()


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
save_name = name + '_medium_distance.png'
plt.savefig(save_name)
plt.show()
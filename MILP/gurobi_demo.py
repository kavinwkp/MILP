from gurobipy import *
import matplotlib.pyplot as plt
import numpy as np

Distance = 1800
Time_total = 100
N = 19
dd = Distance/(N-1)
Acc_max = 1.2
mass = 177800
A = 2.0895
B = 0.0098
C = 0.0065
F_max = 200000
P_max = 5000000
P_ESD = 500000
n_cat = 0.8
n_ESD = 0.9
g = 9.8
i = list(range(1, N))
ii = list(range(0, N))

# PWL point for speed
K = 11
Vmin = 1
Vmax = 45
delk = (Vmax-Vmin)/(K-1)
PWL_SPE = [Vmin]
spv = Vmin
for index in range(K-1):
    spv = spv + delk  # PWL accuracy
    PWL_SPE.append(spv)
S = len(PWL_SPE)


v_limit = 45

delH = 0
E_ini = 0
E_cap = 30000000
L1 = 999999999
L2 = 999999999

m = Model('ESD')

delTime = m.addVars(i, lb=0.0, vtype=GRB.CONTINUOUS, name="delTime")
a_i = m.addVars(i, lb=-Acc_max, ub=Acc_max, vtype=GRB.CONTINUOUS, name="a_i")   # 4 5
v_point = m.addVars(ii, lb=0.0, vtype=GRB.CONTINUOUS, name="v_point")
v_point_squ = m.addVars(ii, lb=0.0, vtype=GRB.CONTINUOUS, name="v_point_squ")

v_ave = m.addVars(i, lb=0.0, vtype=GRB.CONTINUOUS, name="v_ave")
v_ave_squ = m.addVars(i, lb=0.0, vtype=GRB.CONTINUOUS, name="v_ave_squ")
v_ave_div = m.addVars(i, lb=0.0, vtype=GRB.CONTINUOUS, name="v_ave_div")

f_drag = m.addVars(i, vtype=GRB.CONTINUOUS, name="drag")

alpha = m.addVars(N, S, lb=0, ub=1,  vtype=GRB.CONTINUOUS, name='a')
beta = m.addVars(N-1, S, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='b')

E_m = m.addVars(i, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="E_im")
E_cat = m.addVars(i, lb=0.0, vtype=GRB.CONTINUOUS, name="E_cat")
E_dch = m.addVars(i, lb=0.0, vtype=GRB.CONTINUOUS, name="E_dch")
E_ch = m.addVars(i, lb=0.0, vtype=GRB.CONTINUOUS, name="E_ch")
E_i = m.addVars(i, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="E_i")
E_tmp = m.addVars(i, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="E_tmp")
E_smp = m.addVars(i, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="E_smp")
SOE = m.addVars(i, vtype=GRB.CONTINUOUS, name="SOE")

lambda1 = m.addVars(i, vtype=GRB.BINARY, name='lambda1')

for index in range(N-1):
    m.addSOS(GRB.SOS_TYPE2, [beta[index, j] for j in range(S)], )

for index in range(N):
    m.addSOS(GRB.SOS_TYPE2, [alpha[index, j] for j in range(S)])
# SOS2
m.addConstrs((alpha.sum(index, '*') == 1 for index in range(N)), name="alpha")
m.addConstrs((beta.sum(index, '*') == 1 for index in range(N-1)), name="beta")

for index in range(0, N-1):
    m.addConstr(v_point_squ[index] <= v_limit**2)
m.addConstr(v_point[0] == 1, name="start")
m.addConstr(v_point[N-1] == 1, name='end')

for index in range(0, N):
    m.addConstr(v_point[index] == (quicksum(PWL_SPE[j]*alpha[index, j] for j in range(S))), name="for v_point")
for index in range(0, N):
    m.addConstr(v_point_squ[index] == (quicksum(PWL_SPE[j]*PWL_SPE[j]*alpha[index, j] for j in range(S))), name="for v_squ")

m.addConstrs((v_point[index+1]+v_point[index] - 2*v_ave[index+1] == 0 for index in range(0, N-1)), name="for v_ave")

for index in range(0, N-1):
    m.addConstr(v_ave[index+1] == (quicksum(PWL_SPE[j]*beta[index, j] for j in range(S))), name="for beta")
for index in range(0, N-1):
    m.addConstr(v_ave_div[index+1] == (quicksum(1/PWL_SPE[j]*beta[index, j] for j in range(S))), name="for v_ave_div")
for index in range(0, N-1):
    m.addConstr(v_ave_squ[index+1] == (quicksum(PWL_SPE[j]*PWL_SPE[j] * beta[index, j] for j in range(S))), name="for v_ave_squ")
for index in range(0, N-1):
    m.addConstr((v_point_squ[index+1]-v_point_squ[index]) == 2*a_i[index+1]*dd, name="get a_i")  # 2
for index in range(1, N):
    m.addConstr(delTime[index] == dd*v_ave_div[index], name="get delTime")  # 6
    m.addConstr(f_drag[index] == 1000*(A+B*v_ave[index]+C*v_ave_squ[index]), name="Davis formula")

m.addConstr(quicksum(delTime) <= Time_total)    # 8

for index in range(1, N):
    m.addConstr(E_m[index] == (lambda1[index]*(E_cat[index]*n_cat+E_dch[index]*n_ESD) + (1-lambda1[index])*(-E_ch[index]/n_ESD)))  # 30

for index in range(1, N):
    m.addConstr(E_m[index] - 0.5*mass*(v_point_squ[index] - v_point_squ[index-1]) - f_drag[index]*dd - mass*g*delH >= 0, name="E_m")   # 11
    m.addConstr(E_cat[index]*n_cat + E_dch[index]*n_ESD <= P_max*dd*v_ave_div[index])   # 12
    m.addConstr(E_ch[index]/n_ESD <= P_max*dd*v_ave_div[index])    # 13
    m.addConstr(E_dch[index] <= P_ESD*dd*v_ave_div[index])  # 14
    m.addConstr(E_ch[index] <= P_ESD*dd*v_ave_div[index])  # 15
    m.addConstr(E_cat[index]*n_cat + E_dch[index]*n_ESD <= F_max*dd)   # 16
    m.addConstr(E_ch[index]/n_ESD <= F_max*dd)  # 17
    m.addConstr(E_cat[index] <= lambda1[index]*L1)   # 31
    m.addConstr(E_dch[index] <= lambda1[index]*L1)   # 32
    m.addConstr(E_ch[index] <= (1-lambda1[index])*L1)   # 33
    m.addConstr(E_i[index] == E_cat[index] + E_dch[index] - E_ch[index])    # 21


for index in range(1, N):
    m.addConstr(E_tmp[index] == -E_dch[index] + E_ch[index], name='get E_tmp')
m.addConstr(E_smp[1] == E_tmp[1])

for index in range(2, N):
    m.addConstr(E_smp[index] == E_smp[index-1] + E_tmp[index], name='get E_smp')

for index in range(1, N):
    m.addRange(E_ini + E_smp[index], 0, E_cap, "soe")  # 20

for index in range(1, N):
    m.addConstr(SOE[index] == (E_ini+E_smp[index])/E_cap)

obj = quicksum(E_i)
m.setObjective(obj, GRB.MINIMIZE)
# m.write("ESD.lp")

m.optimize()

# Data
# speed curve
plt.grid()
plt.xticks(np.linspace(0, 1800, 10))
plt.yticks(np.linspace(0, 100, 11))

vpoint = []
for index in range(N):
    vpoint.append(v_point[index].x)
v_x = []
for index in range(N):
    v_x.append(index*dd)
plt.plot(v_x, vpoint, 'g')
print(vpoint)

# [1.0, 14.9, 20.9, 25.6, 25.8, 25.6, 25.5, 25.3, 24.7, 24.1, 23.4, 22.7, 21.9, 21.0, 20.2, 19.3, 18.3, 15.4, 1.0]
# [1.0, 14.9, 21.0, 25.6, 25.4, 25.3, 25.1, 25.0, 24.8, 24.2, 23.5, 22.8, 22.0, 21.2, 20.3, 19.4, 18.4, 15.4, 1.0]
# plt.xlim(0, 1800)
# plt.ylim(0, 30)

# battery
# SOE
# SOEy = [0]
# for index in range(1, N):
#     SOEy.append(SOE[index].x*100)
# plt.plot(v_x, SOEy, 'b')
# plt.ylim(0, 100)
# plt.xlim(0, 1800)
#
# # power
# Echy = []
# for index in range(1, N):
#     Echy.append(E_ch[index].x)
# print(Echy)
# vavediv = []
# for index in range(1, N):
#     vavediv.append(v_ave_div[index].x)
# ax1 = plt.twinx()
# Power = []
# for index in range(N-1):
#     Power.append(Echy[index]/(dd*vavediv[index])/1000)
# vxx = [0]
# for index in range(1, N):
#     vxx.append(dd*index)
#     vxx.append(dd*index)
# vxx.pop()
# Powery = []
# for index in range(N-1):
#     Powery.append(Power[index])
#     Powery.append(Power[index])
# ax1.plot(vxx, Powery, 'b--')
# plt.ylim(0, 550)
plt.show()
# plt.savefig('11.svg')
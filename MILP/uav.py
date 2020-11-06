from gurobipy import *
import matplotlib.pyplot as plt
import numpy as np

Distance = 1000
Time_total = 100
N = 19
dt = Time_total/(N-1)
Acc_max = 1.2
dAcc_max = -0.5
mass = 5
rho = 1.225
Sfp = 0.0151

F_max = 24000
P_max = 100
P_BAT = 2200
n_BAT = 0.9
g = 9.8
i = list(range(1, N))
ii = list(range(0, N))

delH = 0
K = 11
Vmin = 1
Vmax = 21
delk = (Vmax-Vmin)/(K-1)
PWL_SPE = [Vmin]
spv = Vmin
for index in range(K-1):
    spv = spv + delk  # PWL accuracy
    PWL_SPE.append(spv)
S = len(PWL_SPE)
# [1, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]

v_limit = 20

m = Model('UAV')

delDistance = m.addVars(i, lb=0.0, vtype=GRB.CONTINUOUS, name="delDistance")
a_i = m.addVars(i, lb=dAcc_max, ub=Acc_max, vtype=GRB.CONTINUOUS, name="a_i")   # 4 5
v_point_x = m.addVars(ii, lb=0.0, vtype=GRB.CONTINUOUS, name="v_point")
v_point_squ_x = m.addVars(ii, lb=0.0, vtype=GRB.CONTINUOUS, name="v_point_squ")

v_ave_x = m.addVars(i, lb=0.0, vtype=GRB.CONTINUOUS, name="v_ave")
v_ave_squ_x = m.addVars(i, lb=0.0, vtype=GRB.CONTINUOUS, name="v_ave_squ")
v_ave_div_x = m.addVars(i, lb=0.0, vtype=GRB.CONTINUOUS, name="v_ave_div")

f_drag = m.addVars(i, vtype=GRB.CONTINUOUS, name="drag")

alpha_x = m.addVars(N, S, lb=0, ub=1,  vtype=GRB.CONTINUOUS, name='a')
beta_x = m.addVars(N - 1, S, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='b')

E_m = m.addVars(i, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="E_im")
E_dch = m.addVars(i, lb=0.0, vtype=GRB.CONTINUOUS, name="E_dch")
E_i = m.addVars(i, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="E_i")
E_tmp = m.addVars(i, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="E_tmp")
E_smp = m.addVars(i, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="E_smp")

for index in range(N-1):
    m.addSOS(GRB.SOS_TYPE2, [beta_x[index, j] for j in range(S)])

for index in range(N):
    m.addSOS(GRB.SOS_TYPE2, [alpha_x[index, j] for j in range(S)])

m.addConstrs((alpha_x.sum(index, '*') == 1 for index in range(N)), name="alpha")
m.addConstrs((beta_x.sum(index, '*') == 1 for index in range(N - 1)), name="beta")

for index in range(0, N-1):
    m.addConstr(v_point_squ_x[index] <= v_limit ** 2)
m.addConstr(v_point_x[0] == 1, name="start")
m.addConstr(v_point_x[N - 1] == 1, name='end')

for index in range(0, N):
    m.addConstr(v_point_x[index] == (quicksum(PWL_SPE[j] * alpha_x[index, j] for j in range(S))), name="for v_point")
for index in range(0, N):
    m.addConstr(v_point_squ_x[index] == (quicksum(PWL_SPE[j] * PWL_SPE[j] * alpha_x[index, j] for j in range(S))), name="for v_squ")

m.addConstrs((v_point_x[index + 1] + v_point_x[index] - 2 * v_ave_x[index + 1] == 0 for index in range(0, N - 1)), name="for v_ave")

for index in range(0, N-1):
    m.addConstr(v_ave_x[index + 1] == (quicksum(PWL_SPE[j] * beta_x[index, j] for j in range(S))), name="for beta")
for index in range(0, N-1):
    m.addConstr(v_ave_div_x[index + 1] == (quicksum(1 / PWL_SPE[j] * beta_x[index, j] for j in range(S))), name="for v_ave_div")
for index in range(0, N-1):
    m.addConstr(v_ave_squ_x[index + 1] == (quicksum(PWL_SPE[j] * PWL_SPE[j] * beta_x[index, j] for j in range(S))), name="for v_ave_squ")

for index in range(0, N-1):
    m.addConstr((v_point_x[index + 1] - v_point_x[index]) == a_i[index + 1] * dt, name="get a_i")  # 2

for index in range(1, N):
    m.addConstr(delDistance[index] == v_ave_div_x[index] * dt, name="get delTime")  # 6
    m.addConstr(f_drag[index] == 0.5 * rho * Sfp * v_ave_squ_x[index], name="drag formula")

m.addConstr(quicksum(delDistance) <= Distance)    # 8

for index in range(1, N):
    m.addConstr(E_m[index] - 0.5 * mass * (v_point_squ_x[index] - v_point_squ_x[index - 1]) - f_drag[index] * delDistance[index] - mass * g * delH >= 0, name="E_m")   # 11
    m.addConstr(E_i[index] <= P_BAT * delDistance[index] * v_ave_div_x[index])  # 14
    m.addConstr(E_i[index] * n_BAT <= F_max * delDistance[index])   # 16


# for index in range(1, N):
#     m.addConstr(E_tmp[index] == -E_dch[index], name='get E_tmp')
# m.addConstr(E_smp[1] == E_tmp[1])
#
# for index in range(2, N):
#     m.addConstr(E_smp[index] == E_smp[index-1] + E_tmp[index], name='get E_smp')

obj = quicksum(E_m)
m.setObjective(obj, GRB.MINIMIZE)

m.optimize()

# Data
# speed curve
plt.grid()
# plt.xticks(np.linspace(0, 100, 11))
plt.xticks(np.linspace(0, 1000, 10))
plt.yticks(np.linspace(0, 32, 9))

vpoint = []
for index in range(N):
    vpoint.append(v_point_x[index].x)

v_x = []
for index in range(N):
    v_x.append(index*dt)
# [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0]
plt.plot(v_x, vpoint, 'g')
# [1.0, 15.5, 21.9, 26.8, 28, 28, 28.0, 28, 28, 28, 28, 28, 28, 28, 28, 26.8, 21.9, 15.5, 1.0]
# v_t = []
# for index in range(1, N):
#     v_t.append(dd / v_ave_x[index].x)
# v_t.insert(0, 0)
# for i in range(1, N):
#     v_t[i] = v_t[i-1] + v_t[i]
# for i in range(1, N):
#     v_t[i] = math.ceil(v_t[i])
# print(v_t)
# plt.plot(v_t, vpoint, 'g')
plt.show()
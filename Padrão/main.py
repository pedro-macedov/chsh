import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import data_generation as data
import train

#########################################################
#                        PARAMETERS                     #
#########################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch = 100000

lossfunc = nn.KLDivLoss(reduction="batchmean", log_target=False)
normal2 = nn.MSELoss(reduction='sum')

patience = 100 # epochs without cost evolution that the NN waits before finishing the training

nu = np.linspace(0,1,40)

inputs = data.data_chsh(batch,device) # Inputs generation

validation = data.data_chsh(1000000,device) # Validation inputs

losses = [] # list of cost evolution for all nu values


#########################################################
#                     TRAINING LOOP                     #
#########################################################

option = 0

costs = train.train(device,nu,inputs,validation,lossfunc,normal2,
                    patience)

np.save("costs.npy", costs)

#########################################################
#                     GRAPH GENERATION                  #
#########################################################
    
p_target = data.p_ab_given_xy_T(0.5,device) # distribuição isotrópica p/ nu=0.5

distance = []

for i in range(len(nu)):
    
    p = data.p_ab_given_xy_T(nu[i],device) # distribuição isotrópica p/ cada ponto
    
    if nu[i] < 0.5:
        distance.append(0)
        
    if nu[i] > 0.5:
        
        d = normal2(p,p_target).item()
        d = d**(1/2)
        distance.append(d)

x = nu[20:]
y = costs[20:]

coef = np.polyfit(x, y, deg=1)
x0 = -coef[1]/coef[0]
x = np.append(x0,x)
x = np.sort(x)
print()
print("Inclinação:", coef[0])
print("Intercepto:", x0)


plt.figure()
plt.plot(nu,costs,label='Resultado Numérico',linestyle='--', linewidth=2, color='blue')
plt.plot(x, coef[0]*x + coef[1], label='Regressão linear',linewidth=1,alpha=0.7, color='green')
plt.plot(nu,distance, label='Resultado Teórico',linewidth=3, color='red',alpha=0.5)
#plt.axvline(x=0.5, color='r', linestyle='--', linewidth=2, label=r'$\nu$ = 0.5')
plt.xlabel("visibility")
plt.ylabel("cost")
plt.legend()
plt.savefig("cost x visibility.png")
plt.show()

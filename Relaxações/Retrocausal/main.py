import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
import data_generation as data
import train
import torch
from scipy.optimize import minimize_scalar
from pathlib import Path

#########################################################
#                     MUTUAL INFORMATION                #
#########################################################

def MI(inputs,r):  
    
    p_00 = len(inputs[(inputs[:,0] == 0) & (inputs[:,2] == 0)])/len(inputs)
    p_01 = len(inputs[(inputs[:,0] == 0) & (inputs[:,2] == 1)])/len(inputs)
    p_10 = len(inputs[(inputs[:,0] == 1) & (inputs[:,2] == 0)])/len(inputs)
    p_11 = len(inputs[(inputs[:,0] == 1) & (inputs[:,2] == 1)])/len(inputs)
    
    p_00_r = len(inputs[(inputs[:,3] == 0) & (inputs[:,4] == 0)])/len(inputs)
    p_01_r = len(inputs[(inputs[:,3] == 0) & (inputs[:,4] == 1)])/len(inputs)
    p_10_r = len(inputs[(inputs[:,3] == 1) & (inputs[:,4] == 0)])/len(inputs)
    p_11_r = len(inputs[(inputs[:,3] == 1) & (inputs[:,4] == 1)])/len(inputs)
    
    
    p_conj = np.zeros((4,4))
    
    combinacoes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    i=0

    for a, b in combinacoes:
        j=0
        for c, d in combinacoes:
            p_conj[i][j] = (len(inputs[(inputs[:,0] == a) & (inputs[:,2] == b) & 
                                (inputs[:,3] == c) & (inputs[:,4] == d)])/len(inputs))
            j=j+1
        i=i+1
    
    p = np.array([p_00,p_01,p_10,p_11])
    p_r = np.array([p_00_r,p_01_r,p_10_r,p_11_r])
        
    mi = 0
    
    for i in range(4):
        for j in range(4):
            
            mi = mi + p_conj[i][j]*(np.log2(p_conj[i][j]) - np.log2(p[i]*p_r[j]))
         
    return mi

#########################################################
#                        PARAMETERS                     #
#########################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#I(x,x') 1.06  0.89  0.64  0.56  0.44  0.38  0.24  0.18  0.10  0.06
flips = [0.10, 0.13, 0.18, 0.20, 0.23, 0.25, 0.30, 0.33, 0.37, 0.40]

lossfunc = nn.KLDivLoss(reduction="batchmean", log_target=True)
normal2 = nn.MSELoss(reduction='sum')

nu = np.concat((np.linspace(0,0.4,5),np.linspace(0.5,1,51)))
# nu = np.linspace(0.5,1,51)


#########################################################
#                     TRAINING LOOP                     #
#########################################################

mis = []
for r in flips:
    
    batch = 100000
    
    inputs = data.data_chsh_relaxed(batch,device,r)
    
    validation = data.data_chsh_relaxed(1000000,device,r) # Validation inputs

    mi = MI(validation, r)
    
    mis.append(mi)
    print(f'I: {mi: .2f}')

    costs = train.train(device,nu,r,mi,inputs,validation,lossfunc,normal2)
    
    np.save(f"costs_mi_{mi:.2f}", costs)

#########################################################
#                     GRAPH GENERATION                  #
#########################################################
    
    
    # Cost X Visibility
    
    plt.figure()
    plt.scatter(nu,costs,label='Numerical result')
    plt.axvline(x=0.5, color='r', linestyle='--', linewidth=2, label=r'$\nu$ = 0.5')
    plt.title(f"Mutual Information = {mi: .2f}")
    plt.xlabel("visibility")
    plt.ylabel("cost")
    plt.legend()
    plt.savefig(f"MI = {mi: .2f} cost x visibility.png")
    plt.show()
        
def uma_reta(nu,costs):
    x = nu[0:7]
    y = costs[0:7]
    coef = np.polyfit(x, y, deg=1)
    
    f = coef[0]*nu + coef[1]
    
    mse = (f - costs)**2
    mse = np.sqrt(mse)
    
    if (mse >= 3*mse.mean()).sum() > 0: return False
    else: return True

def two_line_fit_error(v_break, visibilities, costs):
    v = np.array(visibilities)
    c = np.array(costs)
    
    # Separa em dois segmentos
    mask1 = v <= v_break
    mask2 = v > v_break
    
    if mask1.sum() < 2 or mask2.sum() < 2:
        return float('inf')
    
    # Ajusta reta em cada segmento
    p1 = np.polyfit(v[mask1], c[mask1], 1)
    p2 = np.polyfit(v[mask2], c[mask2], 1)
    
    # Erro total
    err1 = np.sum((np.polyval(p1, v[mask1]) - c[mask1])**2)
    err2 = np.sum((np.polyval(p2, v[mask2]) - c[mask2])**2)
    
    return err1 + err2

mis = sorted(mis)

costs = [[np.load(f"costs_mi_{mis[0]:.2f}.npy"), np.load(f"costs_mi_{mis[1]:.2f}.npy"), np.load(f"costs_mi_{mis[2]:.2f}.npy"),np.load(f"costs_mi_{mis[3]:.2f}.npy"), np.load(f"costs_mi_{mis[4]:.2f}.npy")],
         [np.load(f"costs_mi_{mis[5]:.2f}.npy"), np.load(f"costs_mi_{mis[6]:.2f}.npy"), np.load(f"costs_mi_{mis[7]:.2f}.npy"), np.load(f"costs_mi_{mis[8]:.2f}.npy"), np.load(f"costs_mi_{mis[9]:.2f}.npy")]]

rotulos = [[f"I = {mis[0]:.2f}", f"I = {mis[1]:.2f}", f"I = {mis[2]:.2f}", f"I = {mis[3]:.2f}", f"I = {mis[4]:.2f}"],
         [f"I = {mis[5]:.2f}", f"I = {mis[6]:.2f}", f"I = {mis[7]:.2f}", f"I = {mis[8]:.2f}", f"I = {mis[9]:.2f}"]]

results = []

# Criar a figura e os eixos (2 linhas x 2 colunas)
fig, axs = plt.subplots(2, 5, figsize=(20,6))

for i in range(2):
    for j in range(5):

        costs_i = costs[i][j]
        
        if uma_reta(nu, costs_i): v_transition = 1.0
        else:
        
            # Otimiza o ponto de quebra
            result = minimize_scalar(two_line_fit_error,bounds=(nu[5], nu[-2]),
                                     method='bounded',args=(nu, costs_i))
    
            v_transition = result.x
        
        results.append(v_transition)
        axs[i, j].scatter(nu, costs_i, label='Numerical result')
        axs[i, j].axvline(x=v_transition,color='gray',linestyle='--',linewidth=1,label=rf'$\nu$ = {v_transition:.2f}')
        axs[i, j].set_title(rotulos[i][j])
        axs[i, j].legend(loc="upper left")
        axs[i, j].set_ylim(-0.01,0.5)

# Ajustar layout para evitar sobreposição
plt.tight_layout()

# Salvar imagem final
plt.savefig("relaxação.png", dpi=300)

I = mis

acumulado = np.array((results,I))
acumulado = acumulado.transpose()
np.save("results.npy", acumulado)

plt.show()

if not Path("acumulado.npy").exists():
    np.save("acumulado.npy", acumulado)
else:
    # Atualiza o arquivo acumulado.npy
    pontos_acumulado = np.load("acumulado.npy")
    pontos_acumulado = np.vstack((pontos_acumulado,acumulado))
    np.save("acumulado.npy", pontos_acumulado)


def hall_result(S):
    p = (4 - S)/8
    h = -p*np.log2(p) - (1 - p)*np.log2(1-p)

    return 2 - h - ((4 + S)/8)*np.log2(3)

results = acumulado
v = np.zeros(len(results) + 1)

v[0] = 0.5

for i in range(len(results[:,0])):
    v[i+1] = results[i,0]

S = v*4

s = np.linspace(2, 4,101)

I = np.zeros(len(results[:,1])+1)
for i in range(len(results[:,1])):
    I[i+1] = results[i,1]

I_teo = []
for s_val in s[:-1]:
    I_teo.append(hall_result(s_val))
I_teo[100] = 2 - np.log2(3)

plt.figure()
plt.scatter(S,I,label='Resultado Numérico')
plt.plot(s,I_teo, label='Hall e Branciard (2020)', color='r')
plt.axvline(x=2*np.sqrt(2), color='gray', linestyle='--', linewidth=1, label=r'$S$ = $2\sqrt{2}$')
plt.ylabel("I(bits)", fontsize=12)
plt.xlabel("S", fontsize=12)
plt.title("Retrocausal bilateral")
plt.legend(fontsize=12)
plt.grid()
plt.savefig("S x I.png")
plt.savefig("S x I.pdf")
plt.show()

plt.figure()
plt.scatter(results[:,0]*4,results[:,1],label='Resultado Numérico')
plt.plot(s,I_teo, label='Hall e Branciard (2020)', color='r')
plt.axvline(x=2*np.sqrt(2), color='gray', linestyle='--', linewidth=1, label=r'$S$ = $2\sqrt{2}$')
plt.ylabel("I(bits)", fontsize=12)
plt.xlabel("S", fontsize=12)
plt.title("Retrocausal bilateral")
plt.legend(fontsize=12)
plt.grid()
plt.savefig("S x I (consolidado).png")
plt.savefig("S x I (consolidado).pdf")
plt.show()


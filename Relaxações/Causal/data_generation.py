import numpy as np
import torch
import torch.nn as nn

bins = np.linspace(0, 1, 3)
bin_centers = (bins[:-1] + bins[1:]) / 2

lossfunc = nn.KLDivLoss(reduction="sum", log_target=False)

def data_chsh_relaxed(batch_size, device, r):
    
    rng = np.random.default_rng()
    
    font = rng.random(batch_size)
    fontx = rng.random(batch_size)
    
    flips = round(batch_size*(1-r))
    flips_indices = np.random.choice(np.arange(0, batch_size), size=flips, replace=False)
    fontx[flips_indices] = np.copy(font[flips_indices])

    # 1. Calcular o histograma 2D real
    hist_2d, _, _ = np.histogram2d(font, fontx, bins=bins)
    p_xy = hist_2d / np.sum(hist_2d) # Probabilidade conjunta real
    
    # 2. Obter as marginais reais (somas das linhas e colunas)
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    
    # 3. Calcular a Informação Mútua (evitando log de zero)
    mi_x = 0
    for i in range(len(bins)-1):
        for j in range(len(bins)-1):
            if p_xy[i, j] > 0:
                mi_x += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
                
    mi_y = mi_x + 100
    while abs(mi_x - mi_y) > 1e-4: 
        fonty = rng.random(batch_size)
        flips_indices = np.random.choice(np.arange(0, batch_size), size=flips, replace=False)
        fonty[flips_indices] = np.copy(font[flips_indices])
        
        # 1. Calcular o histograma 2D real
        hist_2d, _, _ = np.histogram2d(font, fonty, bins=bins)
        p_xy = hist_2d / np.sum(hist_2d) # Probabilidade conjunta real
        
        # 2. Obter as marginais reais (somas das linhas e colunas)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)
        
        # 3. Calcular a Informação Mútua (evitando log de zero)
        mi_y = 0
        for i in range(len(bins)-1):
            for j in range(len(bins)-1):
                if p_xy[i, j] > 0:
                    mi_y += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
                    
        # print(mi_x)
        # print(mi_y)
        # print()

    inputs = np.column_stack((font, fontx, fonty))
    inputs = torch.tensor(inputs, dtype=torch.float32, device=device)

    return inputs, mi_x + mi_y

def p_ab_given_xy_T(visibility, device):
    p_noise = np.ones((16))/4
    p_pr = np.zeros((16))
    i=0

    for x, y in [(0,0),(0,1),(1,0),(1,1)]:
        for a, b in [(0,0),(0,1),(1,0),(1,1)]:
            #xyab#
            #placement = b + a*2 + y*(2*2) + x(2**3)
            p_pr[i] = (np.mod(a+b,2) == x*y)/2
            i+=1
            
    # p_pr = p_pr/4
    
    target = torch.tensor((1 - visibility)*p_noise + visibility*p_pr, 
                          dtype=torch.float32, device=device)

    return target
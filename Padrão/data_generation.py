import numpy as np
import torch

def data_chsh(batch_size, device):
    '''
    Recebe como parâmetro o número de pontos a serem gerados e o dispositivo ("cpu" ou "cuda")
    Gera valores aleatórios para x, y e font
    Retorna um tensor de pontos flutuantes no formato (x,font,y)
    '''
    x = np.random.randint(0,2, size=batch_size)
    y = np.random.randint(0,2, size=batch_size)
    font = np.random.rand(batch_size)

    inputs = np.column_stack((x, font, y))
    inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
    

    return inputs

def data_chsh_relaxed(batch_size, device, r):
    '''
    Recebe como parâmetro o número de pontos a serem gerados, o dispositivo ("cpu" ou "cuda") 
    e a proporção de informação compartilhada por Alice e Bob
    Gera dois arrays aleatórios de zeros e uns x e y
    Gera um array x_relaxed que compartilha informação com x na proporção fornecida como parâmetro
    Retorna um tensor de pontos flutuantes no formato (x,font,y,x_relaxed)
    '''
    x = np.random.randint(0,2, size=batch_size)
    y = np.random.randint(0,2, size=batch_size)
    font = np.random.rand(batch_size)
    
    flips = round(batch_size*(r))
    flips_indices = np.random.choice(np.arange(0, batch_size), size=flips, replace=False)
    
    x_relaxed = np.copy(x)
    
    x_relaxed[flips_indices] = 1 - x_relaxed[flips_indices]

    inputs = np.column_stack((x,font, y, x_relaxed))
    inputs = torch.tensor(inputs, dtype=torch.float32, device=device)

    return inputs

def p_ab_given_xy_T(visibility, device):
    p_noise = np.ones((16,1))/4
    p_pr = np.zeros((16,1))
    i=0

    for x, y in [(0,0),(0,1),(1,0),(1,1)]:
        for a, b in [(0,0),(0,1),(1,0),(1,1)]:
            #xyab#
            #placement = b + a*2 + y*(2*2) + x(2**3)
            p_pr[i] = (np.mod(a+b,2) == x*y)/2
            i+=1
            
    target = torch.tensor((1 - visibility)*p_noise + visibility*p_pr, 
                          dtype=torch.float32, device=device)

    return target
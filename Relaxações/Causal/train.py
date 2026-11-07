import data_generation as data
import sys
import copy
import torch
import time
import architecture
import numpy as np   
import torch.nn as nn
from matplotlib import pyplot as plt
   

def train(device,nu,r,mi,inputs,validation,lossfunc,normal2):
    begin = time.time()
    
    costs = []
    
    model = architecture.Model()
    
    model.apply(architecture.initialize_weights) # Apply Xavier initialization (uniform)
    model = model.to(device)
    pesos = copy.deepcopy(model.state_dict())
    

    for i in range(len(nu)):
        
        begin_i = time.time()
        
        model = architecture.Model()
        
        model.load_state_dict(pesos)
        model = model.to(device)

        losses_i = []

        p_target = data.p_ab_given_xy_T(nu[i],device)
        p_target = torch.log(p_target)
        
        if nu[i] >= 0.97:
            p_target = data.p_ab_given_xy_T(nu[i],device)
            lossfunc = nn.MSELoss(reduction='sum')
        
        # loop de treino
        losses_i, pesos = train_relaxed_loop(device,model,inputs,p_target,
                                             lossfunc,pesos,r,nu[i])

        
        X = validation[:,0].unsqueeze(1).to(device)
        Y = validation[:,2].unsqueeze(1).to(device)
        font = validation[:,1].unsqueeze(1).to(device)
        X_sig = validation[:,3].unsqueeze(1).to(device)
        Y_sig = validation[:,4].unsqueeze(1).to(device)

        
        p_target = data.p_ab_given_xy_T(nu[i],device)
        
        with torch.no_grad():
            model.eval()
            model.load_state_dict(pesos)
            p_machine = model(X,Y,X_sig,Y_sig,font).exp()
            cost_l2 = normal2(p_machine,p_target).item()
        
        cost_l2 = cost_l2**0.5
            
        costs.append(cost_l2)
        pesos = copy.deepcopy(model.state_dict())
        
        end_i = time.time()
        tempo_i = (end_i-begin_i)/60 #time in minutes
        sys.stdout.write('\r' + ' ' * 60 + '\r')  # limpa a linha
        print(f'v = {nu[i]:.2f}. Time: {tempo_i:.2f} min. Epoch {len(losses_i)} Cost: {cost_l2:.3f}')


    print()


    end = time.time()

    training_time = (end-begin)/60 # time in minutes

    print(f"Tempo de treinamento: {training_time: .2f}")
    
    return costs

def train_relaxed_loop(device,model,inputs,p_target,lossfunc,pesos,r,nu):
    
    epoch = 0
    losses = []
    counter = 0
    k = 0
    m = 0
    
    X = inputs[:, 0].unsqueeze(1).to(device)
    Y = inputs[:, 2].unsqueeze(1).to(device)
    font = inputs[:, 1].unsqueeze(1).to(device)
    X_sig = inputs[:, 3].unsqueeze(1).to(device)
    Y_sig = inputs[:, 4].unsqueeze(1).to(device)
    
    optimizer = torch.optim.LBFGS(model.parameters(),
                                  lr=1, max_iter=200, history_size=50, 
                                  line_search_fn='strong_wolfe')
    optimizer1 = torch.optim.Adam(model.parameters(),
                                   lr=1e-4,betas=(0.9, 0.99))
    def training_step():
        optimizer.zero_grad()
        nonlocal nu
        optimizer.zero_grad()
        if nu >= 0.97: p_machine = model(X,Y,X_sig,Y_sig,font).exp()
        else: p_machine = model(X,Y,X_sig,Y_sig,font)
        loss = lossfunc(p_machine,p_target)
        loss.backward()
        
        return loss
    
    while epoch < 10000:
        
        model.train()
                
        
        if k == 0: loss = optimizer.step(training_step)
            
            
        if k == 1:
            if nu >= 0.97: p_machine = model(X,Y,X_sig,Y_sig,font).exp()
            else: p_machine = model(X,Y,X_sig,Y_sig,font)

            loss = lossfunc(p_machine,p_target) 
            optimizer1.zero_grad() # zera grad acumulados
            loss.backward() # computa o gradiente do modelo 

            optimizer1.step() # atualiza os parâmetros do modelo 
                  
            
        losses.append(loss.item())
        

        if losses[-1] <= min(losses):
            counter = 0
            best_model = copy.deepcopy(model.state_dict())
        else:
            counter += 1

        
        sys.stdout.write('\r' + ' ' * 60 + '\r')  # limpa a linha
        sys.stdout.write(f'E{epoch} Contador: {counter}. Custo: {loss}')
        sys.stdout.flush()
        
        if (losses[-1] < 1e-6) or (epoch > 1000 and (epoch - counter) < 100):
            break
        
        if k == 0 and epoch > 3 and abs(losses[-2] - losses[-1]) < 1e-12:
            k = 1

            optimizer = torch.optim.LBFGS(model.parameters(),
                                          lr=1, max_iter=200, history_size=50, 
                                          line_search_fn='strong_wolfe')


        if k == 1:
            m += 1
                   
        if m == 500:
            k = 0
            m = 0

            optimizer = torch.optim.LBFGS(model.parameters(),
                                          lr=1, max_iter=200, history_size=50, 
                                          line_search_fn='strong_wolfe')
            
        epoch += 1
    
    
    return losses, best_model
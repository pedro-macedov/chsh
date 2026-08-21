import data_generation as data
import sys
import copy
import torch
import time
import architecture
import torch.nn as nn

def train(device,nu,r,mi,inputs,validation,lossfunc,normal2,patience):
    
    begin = time.time()
    
    costs = []
    
    model = architecture.Model()
    
    model.apply(architecture.initialize_weights)
    model = model.to(device)
    pesos = copy.deepcopy(model.state_dict())

    for i in range(len(nu)):
        
        model = architecture.Model()
        
        model.load_state_dict(pesos)
        model = model.to(device)

        losses_i = []

        p_target = data.p_ab_given_xy_T(nu[i],device)
        p_target = torch.log(p_target)
        
        if nu[i] == 0.97:
            p_target = data.p_ab_given_xy_T(nu[i],device)
            lossfunc = nn.MSELoss(reduction='sum')
        
        
        # loop de treino
        losses_i, pesos, threshold_x, threshold_y = train_relaxed_loop(
            device,model,inputs,p_target,lossfunc,
            patience,pesos,nu[i])

        
        font = validation[:,0].unsqueeze(1).to(device)
        fontx = validation[:,1].unsqueeze(1).to(device)
        fonty = validation[:,2].unsqueeze(1).to(device)

        
        p_target = data.p_ab_given_xy_T(nu[i],device)
        
        with torch.no_grad():
            model.eval()
            model.load_state_dict(pesos)
            model.threshold_x = threshold_x
            model.threshold_y = threshold_y
            p_machine = model(font,fontx,fonty).exp()
            cost_l2 = normal2(p_machine,p_target).item()
            
            
        cost_l2 = cost_l2**0.5
            
        costs.append(cost_l2)
        # losses.append(losses_i)

        pesos = copy.deepcopy(model.state_dict())
        
        sys.stdout.write('\r' + ' ' * 60 + '\r')  # limpa a linha
        # print(f'v = {nu[i]:.2f}. [{model.p00.item():.2f},{model.p01.item():.2f},{model.p10.item():.2f},{model.p11.item():.2f}]. Epoch {len(losses_i)} Cost: {cost_l2:.3f}')
        print(f'v = {nu[i]:.2f}. [{model.x_binario.mean().item():.2f},{model.y_binario.mean().item():.2f}]. Epoch {len(losses_i)} Cost: {cost_l2:.3f}')

    print()


    end = time.time()

    training_time = (end-begin)/60 # time in minutes

    print(f"Tempo de treinamento: {training_time: .2f}")
    
    return costs

def train_relaxed_loop(device,model,inputs,p_target,lossfunc,patience,pesos,nu):
    
    
    def loss_total(p_target, output, model, alpha=100.0):
        
        # loss principal
        loss_principal = lossfunc(output,p_target)
        
        # restrição marginal
        penal = model.x.mean() ** 2 + model.y.mean() ** 2
        
        
        return loss_principal + alpha * penal
    
    epoch = 0
    losses = []
    counter = 0
    best_model = copy.deepcopy(model.state_dict())
    
    
    font = inputs[:,0].unsqueeze(1).to(device)
    fontx = inputs[:,1].unsqueeze(1).to(device)
    fonty = inputs[:,2].unsqueeze(1).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,betas=(0.9, 0.99))
    if nu == 0.5: optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,betas=(0.9, 0.99))
            

    while epoch < 10000:
        
        model.train()

        if nu == 0.97: p_machine = model(font,fontx,fonty).exp()
        else: p_machine = model(font,fontx,fonty)
        # loss = lossfunc(p_machine,p_target) # calcula a função de perda do modelo             
        loss = loss_total(p_target,p_machine, model)
        optimizer.zero_grad() # zera grad acumulados
        
        if torch.isnan(loss) or torch.isinf(loss):
            model.load_state_dict(best_model)
            print(f"Loss: {loss}")
            if torch.isinf(loss):
                break
            continue
            
        
        loss.backward() # computa o gradiente do modelo 

        optimizer.step() # atualiza os parâmetros do modelo 
                  
            
        losses.append(loss.item())
        
        if not torch.isnan(loss) and losses[-1] <= min(losses):
            counter = 0
            best_model = copy.deepcopy(model.state_dict())
            threshold_x = model.threshold_x
            threshold_y = model.threshold_y
        else:
            counter += 1
        
        sys.stdout.write('\r' + ' ' * 60 + '\r')  # limpa a linha
        sys.stdout.write(f'E{epoch} Contador: {counter}. Custo:{losses[-1]}')
        sys.stdout.flush()
        
        if (losses[-1] < 1e-6) or (counter == 1000):
                break
        

        epoch += 1
    
    
    return losses, best_model, threshold_x,threshold_y
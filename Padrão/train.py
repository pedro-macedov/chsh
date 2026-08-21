import data_generation as data
import sys
import copy
import torch
import time
import architecture

def train(device,nu,inputs,validation,lossfunc,normal2,patience):
    begin = time.time()
    
    costs = []
    
    model = architecture.Model(2)
    
    model.apply(architecture.initialize_weights) # Apply Xavier initialization (uniform)
    model = model.to(device)
    pesos = copy.deepcopy(model.state_dict())

    for i in range(len(nu)):
        
        begin_i = time.time()
        
        model = architecture.Model(2)
        model.load_state_dict(pesos)
        
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-3, 
                                              betas=(0.9, 0.98))

        losses_i = []

        p_target = data.p_ab_given_xy_T(nu[i],device)
        
        if nu[i] == 1.00:
            eps = 1e-3
            mask = p_target != 0  
            p_target = p_target - mask * eps 
            p_target = p_target.clamp(min=eps)
        
        # loop de treino
        losses_i, pesos = train_loop(
            device,model,inputs,p_target,lossfunc,optimizer,
            patience,pesos)
        
        # plt.figure()
        # plt.plot(losses_i)
        # plt.show()
        
        
        X = validation[:,0] # tensor [x,font]
        X = X.unsqueeze(1)
        
        if len(validation[0]) == 3:
           Y = validation[:,2] # tensor [font,y] ou [font,y,x_relaxed]
           Y = Y.unsqueeze(1)
        if len(validation[0]) == 4:
           Y = validation[:,2:] # tensor [font,y] ou [font,y,x_relaxed]
        font = validation[:,1]
        font = font.unsqueeze(1)
        X = X.to(device)
        Y = Y.to(device)
        font = font.to(device)
        
        if nu[i] == 1.00:
            p_target = data.p_ab_given_xy_T(nu[i],device)
        
        with torch.no_grad():
            model.eval()
            model.load_state_dict(pesos)
            p_machine = model(X,Y, font).exp()
            cost_l2 = normal2(p_machine,p_target).item()
            
        cost_l2 = cost_l2**0.5
            
        costs.append(cost_l2)
        # losses.append(losses_i)
        
        pesos = copy.deepcopy(model.state_dict())
        # torch.save(model.state_dict(), f'models/i_{mi:.2f}/p_{i}.pth') # salva o modelo
            
        
        end_i = time.time()
        tempo_i = (end_i-begin_i)/60 #time in minutes
        sys.stdout.write('\r' + ' ' * 60 + '\r')  # limpa a linha
        print(f'v = {nu[i]: .2f}. Time: {tempo_i: .2f} min. Epoch {len(losses_i)} Cost: {cost_l2: .3f}')


    print()


    end = time.time()

    training_time = (end-begin)/60 # time in minutes

    print(f"Tempo de treinamento: {training_time: .2f}")
    
    return costs

def train_loop(device,model,inputs,p_target,lossfunc,optimizer,patience,pesos):
    
    epoch = 0
    losses = []
    counter = 0

    
    while counter < patience:
        
        X = inputs[:,0] # tensor [x,font]
        X = X.unsqueeze(1)
        
        if len(inputs[0]) == 3:
           Y = inputs[:,2] # tensor [font,y] ou [font,y,x_relaxed]
           Y = Y.unsqueeze(1)
        if len(inputs[0]) == 4:
           Y = inputs[:,2:] # tensor [font,y] ou [font,y,x_relaxed]
        font = inputs[:,1]
        font = font.unsqueeze(1)
        X = X.to(device)
        Y = Y.to(device)
        font = font.to(device)

        model.train()
                
                
        p_machine = model(X,Y, font) # tensor no formato [P_a(0|x,font),P_a(1|x,font)]

        loss = lossfunc(p_machine,p_target) # calcula a função de perda do modelo 
        losses.append(loss.item())
        
        optimizer.zero_grad() # zera grad acumulados
        loss.backward() # computa o gradiente do modelo 

        optimizer.step() # atualiza os parâmetros do modelo 
        # scheduler.step()
                
        sys.stdout.write('\r' + ' ' * 60 + '\r')  # limpa a linha
        sys.stdout.write(f'{epoch} Contador: {counter}. Custo: {loss}')
        sys.stdout.flush()
        
        if losses[-1] <= min(losses):
            counter = 0
            pesos = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            
        if (losses[-1] < 1e-6) or (epoch == 5000):
            break
    
        epoch += 1
        
    
    return losses, pesos

def train_old(device,model,batch,p_target,lossfunc,optimizer,patience,pesos):
    '''
    

    Parameters
    ----------
    device : string
        "cpu" or "cuda".
    model : nn.Module()
        Neural Network.
    inputs : torch.tensor
        [x,font,y] or [x,font,y, x_relaxed].
    p_target : torch.tensor [16,1]
        [p(a=0,b=0|x=0,y=0)]
        [p(a=0,b=1|x=0,y=0)]
        [p(a=1,b=0|x=0,y=0)]
        [       ...        ]
        [p(a=1,b=1|x=1,y=1)].
    lossfunc : nn.MSELoss() or nn.KLDivLoss() ... 
    optimizer : torch.optim.SGD() or torch.optim.Adam()
    patience : int
        epochs without cost evolution that the NN waits before finishing the training.

    Returns
    -------
    losses : list
        Loss function x epoch.

    '''

    epoch = 0
    losses = []
    counter = 0
    l = 0
    lim = 20000

    
    while counter < patience:
        
        inputs = data.data_chsh(batch,device) # Inputs generation
        
        X = inputs[:,0] # tensor [x,font]
        X = X.unsqueeze(1)
        
        if len(inputs[0]) == 3:
           Y = inputs[:,2] # tensor [font,y] ou [font,y,x_relaxed]
           Y = Y.unsqueeze(1)
        if len(inputs[0]) == 4:
           Y = inputs[:,2:] # tensor [font,y] ou [font,y,x_relaxed]
        font = inputs[:,1]
        font = font.unsqueeze(1)
        X = X.to("cpu")
        Y = Y.to("cpu")
        font = font.to("cpu")

        model.train()
        
        
        if epoch > lim:
            break
        
        # if counter == 0.5*patience and k == 0:
        #     optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4, 
        #                                   betas=(0.9, 0.9))
        #     batch = round(batch*2)
        #     counter = 0
        #     k = 1
        #     otimizador = 'tuning'

        
        # if epoch == 100:
        #     optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3, 
        #                                   betas=(0.8, 0.9))
        #     otimizador = 'lr 1e-3'
            
        # if k == 1:
        #     # model.load_state_dict(pesos)
        #     optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4, 
        #                                   betas=(0.6, 0.9))
        #     # optimizer = torch.optim.SGD(model.parameters(),lr=1e-9)
        #     otimizador = 'lr 1e-4'
            
        # if k == 2:
        #     # model.load_state_dict(pesos)
        #     optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5, 
        #                                   betas=(0.6, 0.9))
        #     # optimizer = torch.optim.SGD(model.parameters(),lr=1e-9)
        #     otimizador = 'lr 1e-5'
            
        # if k == 3:
        #     # model.load_state_dict(pesos)
        #     optimizer = torch.optim.AdamW(model.parameters(),lr=1e-6, 
        #                                   betas=(0.6, 0.9))
        #     otimizador = 'lr 1e-6'
            
        # if k == 4:
        #     # model.load_state_dict(pesos)
        #     optimizer = torch.optim.AdamW(model.parameters(),lr=1e-7, 
        #                                   betas=(0.6, 0.9))
        #     otimizador = 'lr 1e-7'
        
        # if counter == patience*0.5:
        #     model.load_state_dict(pesos)
        #     # optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
        #     optimizer = torch.optim.AdamW(model.parameters(),lr=1e-6, 
        #                                   betas=(0.9, 0.999))
        #     otimizador = 'SGD'
        
       
        p_machine = model(X,Y, font) # tensor no formato [P_a(0|x,font),P_a(1|x,font)]

        loss = lossfunc(p_machine,p_target) # calcula a função de perda do modelo 
        losses.append(loss.item())
        
        optimizer.zero_grad() # zera grad acumulados
        loss.backward() # computa o gradiente do modelo 

        optimizer.step() # atualiza os parâmetros do modelo 
                
        sys.stdout.write('\r' + ' ' * 60 + '\r')  # limpa a linha
        sys.stdout.write(f'Contador: {counter}. Custo: {loss}')
        sys.stdout.flush()
        
        if losses[-1] <= min(losses):
            counter = 0
            pesos = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            
        if losses[-1] < 5e-6:
            l = 1
            break
    
        epoch += 1
        
    
    return losses, pesos, l
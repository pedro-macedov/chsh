import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

class Model(nn.Module):
    def __init__(self,input_b1):
        super(Model,self).__init__()

        # ansatz A
        self.a1 = nn.Linear(2,5) # 10 +
        self.a2 = nn.Linear(5,5) # 25 +
        self.a3 = nn.Linear(5,5) # 25 +
        self.a4 = nn.Linear(5,2) # 10 = 70

        # ansatz B
        self.b1 = nn.Linear(input_b1,5)
        self.b2 = nn.Linear(5,5)
        self.b3 = nn.Linear(5,5)
        self.b4 = nn.Linear(5,2)

    def forward(self, X, Y, Font):

        x = torch.cat([X, Font],dim=1)
        y = torch.cat([Y, Font],dim=1)

        # forward in ansatz A
        p_a_x = F.tanh(self.a1(x)) # Ativação tanh na 1ª camada oculta
        p_a_x = F.tanh(self.a2(p_a_x)) # Ativação tanh na 2ª camada oculta
        p_a_x = F.tanh(self.a3(p_a_x))
        p_a_x = F.log_softmax(self.a4(p_a_x), dim=1) # Softmax na camada de saída

        # forward in ansatz B
        p_b_y = F.tanh(self.b1(y)) # Ativação ReLU na 1ª camada oculta
        p_b_y = F.tanh(self.b2(p_b_y)) # Ativação ReLU na 2ª camada oculta
        p_b_y = F.tanh(self.b3(p_b_y))
        p_b_y = F.log_softmax(self.b4(p_b_y), dim=1)
        
        results = torch.cat([X,Y[:,0:],p_a_x,p_b_y],dim=1) # [x,font,y,p_a(0|x,y),p_a(1|x,y),p_b(0|x,y),p_b(1|x,y)]

        combinacoes = [(0, 0), (0, 1), (1, 0), (1, 1)]
        p_ab_xy = []
        
        for x_val, y_val in combinacoes:
            mask = (results[:, 0] == x_val) & (results[:, 1] == y_val)
            results_filtered = results[mask]
                        
            p_a0, p_a1 = results_filtered[:, -4], results_filtered[:, -3]
            p_b0, p_b1 = results_filtered[:, -2], results_filtered[:, -1]
        
            total = len(results_filtered)


            p_00 = torch.logsumexp(p_a0 + p_b0, dim=0) - torch.log(torch.tensor(total, device=p_a0.device))
            p_01 = torch.logsumexp(p_a0 + p_b1, dim=0) - torch.log(torch.tensor(total, device=p_a0.device))
            p_10 = torch.logsumexp(p_a1 + p_b0, dim=0) - torch.log(torch.tensor(total, device=p_a0.device))
            p_11 = torch.logsumexp(p_a1 + p_b1, dim=0) - torch.log(torch.tensor(total, device=p_a0.device))
        
            p_ab_xy.append((p_00).unsqueeze(0))  # P(0,0|x,y)
            p_ab_xy.append((p_01).unsqueeze(0))  # P(0,1|x,y)
            p_ab_xy.append((p_10).unsqueeze(0))  # P(1,0|x,y)
            p_ab_xy.append((p_11).unsqueeze(0))
        
        return torch.stack(p_ab_xy)


# Aplicando Xavier Initialization nos pesos
def initialize_weights(m):
    if isinstance(m, nn.Linear):  # Se for uma camada Linear
        # init.xavier_uniform_(m.weight, gain=init.calculate_gain('tanh'))  # Inicializa os pesos com Xavier Uniform
        init.xavier_normal_(m.weight, gain=init.calculate_gain('tanh'))
        # init.xavier_normal_(m.weight)
        # init.orthogonal_(m.weight)
        # init.kaiming_uniform_(m.weight)
        # init.kaiming_normal_(m.weight, nonlinearity='relu')
        # init.kaiming_normal_(m.weight, nonlinearity='linear')
        if m.bias is not None:
            init.zeros_(m.bias)  # Inicializa os bias como 0
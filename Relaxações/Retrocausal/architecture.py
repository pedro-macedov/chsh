import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        n_lambda = 16
        n_ab = 16

        # ansatz A
        self.Lambda1 = nn.Linear(3,n_lambda)
        self.Lambda2 = nn.Linear(n_lambda,n_lambda)
        self.Lambda3 = nn.Linear(n_lambda,n_lambda)
        self.Lambda4 = nn.Linear(n_lambda,n_lambda)
        self.Lambda5 = nn.Linear(n_lambda,1)

        # ansatz A
        self.a1 = nn.Linear(3,n_ab)
        self.a2 = nn.Linear(n_ab,n_ab)
        self.a3 = nn.Linear(n_ab,n_ab)
        self.a4 = nn.Linear(n_ab,n_ab)
        self.a5 = nn.Linear(n_ab,2)

        # ansatz B
        self.b1 = nn.Linear(3,n_ab)
        self.b2 = nn.Linear(n_ab,n_ab)
        self.b3 = nn.Linear(n_ab,n_ab)
        self.b4 = nn.Linear(n_ab,n_ab)
        self.b5 = nn.Linear(n_ab,2)

    def forward(self, X, Y, X_sig, Y_sig, Font):
        
        f = torch.cat([X_sig, Y_sig, Font],dim=1)
        
        font = F.tanh(self.Lambda1(f))
        font = F.tanh(self.Lambda2(font)) 
        font = F.tanh(self.Lambda3(font))
        font = F.tanh(self.Lambda4(font))
        font = F.tanh(self.Lambda5(font))
            
        self.font = font

        x = torch.cat([X, Font, font],dim=1)
        y = torch.cat([Y, Font, font],dim=1)

        # forward in ansatz A
        p_a_x = F.tanh(self.a1(x))
        p_a_x = F.tanh(self.a2(p_a_x)) 
        p_a_x = F.tanh(self.a3(p_a_x)) 
        p_a_x = F.tanh(self.a4(p_a_x))
        p_a_x = F.log_softmax(self.a5(p_a_x), dim=1) 

        # forward in ansatz B
        p_b_y = F.tanh(self.b1(y))
        p_b_y = F.tanh(self.b2(p_b_y))
        p_b_y = F.tanh(self.b3(p_b_y))
        p_b_y = F.tanh(self.b4(p_b_y))
        p_b_y = F.log_softmax(self.b5(p_b_y), dim=1)
        
        results = torch.cat([X,Y,p_a_x,p_b_y],dim=1)

        combinacoes = [(0, 0), (0, 1), (1, 0), (1, 1)]
        p_ab_xy = []
        
        for x_val, y_val in combinacoes:
            mask = (results[:, 0] == x_val) & (results[:, 1] == y_val)
            results_filtered = results[mask]
                        
            p_a0, p_a1 = results_filtered[:, 2], results_filtered[:, 3]
            p_b0, p_b1 = results_filtered[:, 4], results_filtered[:, 5]
            
        
            total = torch.tensor(len(results_filtered), device=p_a_x.device).log()
            

            p_00 = torch.logsumexp((p_a0 + p_b0),dim=0) - total
            p_01 = torch.logsumexp((p_a0 + p_b1),dim=0) - total
            p_10 = torch.logsumexp((p_a1 + p_b0),dim=0) - total
            p_11 = torch.logsumexp((p_a1 + p_b1),dim=0) - total
        
            p_ab_xy.append((p_00).unsqueeze(0))  # P(0,0|x,y)
            p_ab_xy.append((p_01).unsqueeze(0))  # P(0,1|x,y)
            p_ab_xy.append((p_10).unsqueeze(0))  # P(1,0|x,y)
            p_ab_xy.append((p_11).unsqueeze(0))
        
        return torch.cat(p_ab_xy)


def initialize_weights(self):
    for name, m in self.named_modules():
        if isinstance(m, nn.Linear):
            if name in ['a5', 'b5']:
                init.xavier_normal_(m.weight, gain=1.0)
                init.zeros_(m.bias)
            else:
                init.xavier_normal_(m.weight, gain=init.calculate_gain('tanh'))
                init.zeros_(m.bias)

            

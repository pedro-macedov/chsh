import torch.nn as nn
import torch
import torch.nn.init as init

class STEBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        return (z > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors
        
        grad = 1 - torch.tanh(z)**2
        
        return grad_output*grad


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        n_xy = 4
        n_ab = 8
        
        # ansatz X
        self.X1 = nn.Linear(1,n_xy)
        self.X2 = nn.Linear(n_xy,n_xy)
        self.X3 = nn.Linear(n_xy,n_xy)
        self.X4 = nn.Linear(n_xy,1)
        
        # ansatz Y
        self.Y1 = nn.Linear(1,n_xy)
        self.Y2 = nn.Linear(n_xy,n_xy)
        self.Y3 = nn.Linear(n_xy,n_xy)
        self.Y4 = nn.Linear(n_xy,1)

        # ansatz A
        self.a1 = nn.Linear(2,n_ab)
        self.a2 = nn.Linear(n_ab,n_ab)
        self.a3 = nn.Linear(n_ab,n_ab)
        self.a4 = nn.Linear(n_ab,n_ab)
        self.a5 = nn.Linear(n_ab,2)

        # ansatz B
        self.b1 = nn.Linear(2,n_ab)
        self.b2 = nn.Linear(n_ab,n_ab)
        self.b3 = nn.Linear(n_ab,n_ab)
        self.b4 = nn.Linear(n_ab,n_ab)
        self.b5 = nn.Linear(n_ab,2)

    def forward(self, Font, Fontx, Fonty):
        
        x = torch.tanh(self.X1(Fontx))
        x = torch.tanh(self.X2(x))
        x = torch.tanh(self.X3(x))
        x = torch.clamp(self.X4(x),-10,10)
        
        y = torch.tanh(self.Y1(Fonty))
        y = torch.tanh(self.Y2(y))
        y = torch.tanh(self.Y3(y))
        y = torch.clamp(self.Y4(y),-10,10)
        
        self.x = x
        self.y = y
        
        if self.training:
            threshold_x = x.median().detach()
            threshold_y = y.median().detach()
            x_debiased = x - threshold_x
            y_debiased = y - threshold_y
            
            self.threshold_x = threshold_x.detach()
            self.threshold_y = threshold_y.detach()
            
        else:
            x_debiased = x - self.threshold_x
            y_debiased = y - self.threshold_y
            
        x_binario = STEBinary.apply(x_debiased)
        y_binario = STEBinary.apply(y_debiased)
        
        self.x_binario = x_binario.detach()
        self.y_binario = y_binario.detach()
        
        

        a_cat = torch.cat([Font, x_binario],dim=1)
        b_cat = torch.cat([Font, y_binario],dim=1)

        # forward in ansatz A
        p_a_x = torch.tanh(self.a1(a_cat))
        p_a_x = torch.tanh(self.a2(p_a_x)) 
        p_a_x = torch.tanh(self.a3(p_a_x)) 
        p_a_x = torch.tanh(self.a4(p_a_x)) 
        p_a_x = torch.log_softmax(self.a5(p_a_x), dim=1) 

        # forward in ansatz B
        p_b_y = torch.tanh(self.b1(b_cat))
        p_b_y = torch.tanh(self.b2(p_b_y))
        p_b_y = torch.tanh(self.b3(p_b_y))
        p_b_y = torch.tanh(self.b4(p_b_y))
        p_b_y = torch.log_softmax(self.b5(p_b_y), dim=1)
        
        p_ab_xy = []
        
        results = torch.cat([x_binario,y_binario,p_a_x,p_b_y],dim=1)
        
        combinacoes = [(0,0),(0,1),(1,0),(1,1)]
        
        # batch_size = torch.log(torch.tensor(Font.size(0), device=p_b_y.device))
        
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
            
        return torch.cat(p_ab_xy)

# def initialize_weights(self):
#     for name, m in self.named_modules():
#         if isinstance(m, nn.Linear):
#             if name in ['X4', 'Y4',]:
#                 nn.init.xavier_normal_(m.weight,gain=0.5)
#                 nn.init.zeros_(m.bias)
#             else:
#                 init.xavier_normal_(m.weight, gain=init.calculate_gain('tanh'))
#                 nn.init.zeros_(m.bias)

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
            
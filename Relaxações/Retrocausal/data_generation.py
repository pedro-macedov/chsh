import numpy as np
import torch

def data_chsh_relaxed(batch_size, device, r):
    
    rng = np.random.default_rng()
    
    xy_00 = np.concat((np.zeros((batch_size//4,2)), rng.random((batch_size//4,1))),axis=1)
    xy_01 = np.concat((np.zeros((batch_size//4,1)), np.ones((batch_size//4,1)),
                       rng.random((batch_size//4,1))),axis=1)
    xy_10 = np.concat((np.ones((batch_size//4,1)), np.zeros((batch_size//4,1)),
                      rng.random((batch_size//4,1))), axis=1)
    xy_11 = np.concat((np.ones((batch_size//4,2)), rng.random((batch_size//4,1))),axis=1)
    
    inputs = np.concat((xy_00,xy_01,xy_10,xy_11), axis=0)
    rng.shuffle(inputs, axis=0)
    
    x = inputs[:,0]
    y = inputs[:,1]
    font = inputs[:,2]
    
    # x = rng.integers(0,2, size=batch_size)
    # y = rng.integers(0,2, size=batch_size)
    # font = rng.random(batch_size)
    
    # prob_error = 1/(2*batch_size)
    
    # px0 = np.sum((x == 0)*1)/batch_size
    # py0 = np.sum((y == 0)*1)/batch_size
    
    # print("Generating Alice's input...")
    # while np.abs(px0 - 0.5) >= prob_error:
    #     x = rng.integers(0,2, size=batch_size)
    #     px0 = np.sum((x == 0)*1)/batch_size
    
    # print("Generating Bob's input...")
    # while np.abs(py0 - 0.5) >= prob_error:
    #     y = rng.integers(0,2, size=batch_size)
    #     py0 = np.sum((y == 0)*1)/batch_size
    
    flips = round(batch_size*(r))
    flips_indices = rng.choice(np.arange(0, batch_size), size=flips, replace=False)
    
    x_relaxed = np.copy(x)
    
    x_relaxed[flips_indices] = 1 - x_relaxed[flips_indices]
    
    flips = round(batch_size*(r))
    flips_indices = rng.choice(np.arange(0, batch_size), size=flips, replace=False)
    
    y_relaxed = np.copy(y)
    
    y_relaxed[flips_indices] = 1 - y_relaxed[flips_indices]

    inputs = np.column_stack((x,font, y, x_relaxed, y_relaxed))
    inputs = torch.tensor(inputs, dtype=torch.float32, device=device)

    return inputs

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
            
    target = torch.tensor((1 - visibility)*p_noise + visibility*p_pr, 
                          dtype=torch.float32, device=device)

    return target
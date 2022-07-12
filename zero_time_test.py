import time
import numpy as np
import torch

w = np.random.rand(100,1000,1000)
i = np.random.rand(100,1000,1000)
z = np.zeros(100*1000*1000)
z_r = z.reshape(100,1000,1000)

t_z = torch.Tensor(z_r)
t_w = torch.Tensor(w)
t_i = torch.Tensor(i)

print(t_z[0][0])
print(t_w.shape,t_i.shape,t_z.shape)

def time_mm(m1,m2,num):
    sum_t = 0
    for n in range(num):
        s = time.time()
        torch.matmul(m1,m2)
        sum_t += time.time() - s
    return sum_t/num
    
torch.cuda.set_device(3)
t_w = t_w.cuda()
t_i = t_i.cuda()
t_z = t_z.cuda()
print('w*i = '+ str(time_mm(t_w,t_i,50)))
print('z*i = '+ str(time_mm(t_z,t_i,50)))
print('w*i = '+ str(time_mm(t_w,t_i,50)))
print('z*i = '+ str(time_mm(t_z,t_i,50)))


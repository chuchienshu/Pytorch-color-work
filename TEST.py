from skimage.color import rgb2gray, rgb2yuv, yuv2rgb
from skimage.transform import resize
from utils import frame_utils
import numpy as np
import os,sys

'''
import torch
from torch.autograd import Variable, Function
import numpy as np
import torch.nn as nn
from torch.nn import init


class Rebalance_Op(Function):
    @staticmethod
    def forward(ctx, input, factors):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.save_for_backward(input, factors)
        
        # return tensor * constant
        #return 不能仅返回 input 否则 不会执行 backward 操作,
        return input * 1.

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        input, factors = ctx.saved_variables
        grad_input = grad_factors = None
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output.mm(weight.t())
            grad_input = grad_output * factors
            # grad_input = grad_output
        # if ctx.needs_input_grad[1]:
        #     #t() 转置, mm() matrix multiplication
        #     grad_factors = grad_output.t().mm(input)
        return grad_input, None

# mc = Rebalance_Op.apply

# target = np.random.randint()
def hook(module, grad_input, grad_output):
    print('hook1 grad_input ',grad_input)
    print('hook2  grad_output',grad_output)

fact = Variable(torch.from_numpy(np.array([[0,200,3]]).astype('float32')))

class My_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,5, bias=False)
        self.fc2 = nn.Linear(5,3, bias=False)
        self.mc = Rebalance_Op.apply
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.constant(m.weight, 1.)
                
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self,input):
        fc1 = self.fc1(input)
        out = self.fc2(fc1)
        out = self.mc(out, fact)
        # out = out * 6

        return out

#fix input&target for the debug convience
target = Variable(torch.zeros(2, 3))
input = Variable(torch.ones(2, 4))
# f = Variable(torch.from_numpy(np.arange(0,4).astype('float32')))
# print(input * f)

network = My_net()
# network.fc2.register_backward_hook(hook)

optimizer = torch.optim.SGD(network.parameters(), lr=1)
optimizer.zero_grad()
# for p in network.parameters():
#     print(p)
# print('    /////////// inital paras ')
loss = ((network(input)-target)**2).sum()
loss.backward()
print('grad1 ', network.fc1.weight.grad)
print('grad2', network.fc2.weight.grad)

optimizer.step()
print('#########')
# print(network.fc1.weight)
print(network.parameters())
for p in network.parameters():
    print(p)
print(network.fc2.weight.grad)
print('#########')




exit()
'''
from glob import glob
from os.path import join
import natsort

# root = '/home/chuchienshu/Documents/propagation_refine/data/sintel_test_clean/'
root = '/home/chuchienshu/Downloads/dataset/dataset4color_propagation/Zootopia/'
in_image_root = '/home/chuchienshu/Documents/propagation_refine/data/inputs/'
gt_images = sorted(glob(join(root, '*.png')))
images = natsort.natsorted(glob(join(root, '*.png')))
#ordered as os dir order
f1 = images[:-1]
f2 = images[1:]

assert len(f1) == len(f2)
f_ = open('zoopedia.txt', 'a')
for i, j in zip(f1,f2):
    print(i[-10:], '  ', j[-10:])
    f_.write('%s %s\n' % (i, j))
f_.flush()
f_.close()

exit()
def frame(path, list_name, is_dir=True, del_f=0):  
    '''
    this function is used to get the list of the frame 2 in a two order dir.
    '''
    fil_lis = sorted(os.listdir(path))
    if not is_dir:
        del fil_lis[del_f]
    for file in fil_lis:  
        file_path = os.path.join(path, file)  
        if os.path.isdir(file_path):  
                
            frame(file_path, list_name, is_dir=False, del_f=del_f)  
        else:  
            list_name.append(file_path)  
    list_name = sorted(list_name)
    return list_name

f1 = frame(root, [], del_f=-1)
f2 = frame(root, [])
assert len(f1) == len(f2)
for i in range(len(f1)):
    assert f1[i][:-8] == f2[i][:-8]

f_ = open('sintal_testset.txt', 'a')
for i, j in zip(f1,f2):
    print(i)
    f_.write('%s %s\n' % (i, j))
f_.flush()
f_.close()

print(len(f1))

'''
ff = frame_1(root, [])

file_ = open('thrtype.txt', 'a')
for i in range(len(ff)) :
    print(ff[i])
    print(gt_images[i])
    print(images[i])
    file_.write('%s %s %s \n' % (images[i],ff[i], gt_images[i] ))
    print('@@@@@@@@@@@@@@@')
    assert ff[i][:-9] == gt_images[i][:-9]
'''
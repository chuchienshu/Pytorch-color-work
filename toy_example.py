from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def check_value(inds, val):
    ''' Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function '''
    if(np.array(inds).size==1):
        if(inds==val):
            return True
    return False

def na(): # shorthand for new axis
    return np.newaxis

def flatten_nd_array(pts_nd,axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
    pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS,SHP[axis])
    return pts_flt

def unflatten_2d_array(pts_flt,pts_nd,axis=1,squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])

    if(squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out

class NNEncode():
    ''' Encode points using NearestNeighbors search and Gaussian kernel '''
    def __init__(self,NN,sigma,km_filepath='',cc=-1):
        if(check_value(cc,-1)):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(self.cc)

        self.alreadyUsed = False

    def encode_points_mtx_nd(self,pts_nd,axis=1,returnSparse=False,sameBlock=True):
        pts_flt = flatten_nd_array(pts_nd,axis=axis)
        #pts_flt ---> [N*H*W, 2]
        P = pts_flt.shape[0]
        #P ---> N*H*W
        if(sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0 # already pre-allocated
            print('alreadyUsed')
            print(self.p_inds)
        else:
            print('notUsed')
            # print(self.p_inds)
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P,self.K))
            #self.pts_enc_flt.shape ---> [N*H*W, 313]
            self.p_inds = np.arange(0,P,dtype='int')[:,na()]
            #self.p_inds.shape ---> [N*H*W, 1]

        (dists,inds) = self.nbrs.kneighbors(pts_flt)
        #inds.shape ---> [N*H*W, NN]

        wts = np.exp(-dists**2/(2*self.sigma**2))
        wts = wts/np.sum(wts,axis=1)[:,na()]
        #wts.shape ---> [N*H*W, NN]
        
        #将输入的 feature map(ab 值)与调色板 bin 中最近的 NN(此处取 10) 个距离值赋值到 pts_enc_flt 中，然后展开成 4d 形式返回。
        self.pts_enc_flt[self.p_inds,inds] = wts
        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt,pts_nd,axis=axis)
        #pts_enc_nd.shape  -----> [N, 313, H, W]

        return pts_enc_nd

    def decode_points_mtx_nd(self,pts_enc_nd,axis=1):
        pts_enc_flt = flatten_nd_array(pts_enc_nd,axis=axis)
        pts_dec_flt = np.dot(pts_enc_flt,self.cc)
        pts_dec_nd = unflatten_2d_array(pts_dec_flt,pts_enc_nd,axis=axis)
        return pts_dec_nd

    def decode_1hot_mtx_nd(self,pts_enc_nd,axis=1,returnEncode=False):
        pts_1hot_nd = nd_argmax_1hot(pts_enc_nd,axis=axis)
        pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd,axis=axis)
        if(returnEncode):
            return (pts_dec_nd,pts_1hot_nd)
        else:
            return pts_dec_nd

# self.cretion( output, torch.max(target, 1)[1] )
nnenc = NNEncode(10,5,km_filepath='/home/chuchienshu/Documents/propagation_classification/models/custom_layers/pts_in_hull.npy')


bottom = np.random.randint(0,10,(2,3,3,3)).astype('float32')

# print(bottom)
bt = Variable(torch.from_numpy(bottom).cuda())
fac = np.array([[1,2],[3,4],[5,6]])
fac_a = fac[:,0][np.newaxis,:,np.newaxis,np.newaxis]
fac_b = fac[:,1][np.newaxis,:,np.newaxis,np.newaxis]

pred_ab = np.concatenate((np.sum(bottom * fac_a, axis=1, keepdims=True), np.sum(bottom * fac_b, axis=1, keepdims=True)), axis=1)


# print(fac_a,fac_a.shape)
# print(fac_b,fac_b.shape)
# print(bottom * fac_a, '   jfdis')
# print(bottom * fac_b, '   fac_b')
# print(np.sum(bottom * fac_a, axis=1, keepdims=True), '   44')
# print(np.sum(bottom * fac_b, axis=1, keepdims=True), '   66')
print(pred_ab, pred_ab.shape)

for i, im in enumerate(pred_ab):
    print(im)
    print(i)
exit()
# bt = flatten_nd_array(bt.data.numpy())
##bt = bt.permute(0,2,3,1).contiguous().view(50, -1)


#/////////////////////////////////////////////////////////
bottom = np.random.randint(0,10,(8,2,5,5)).astype('float32')
print(bottom)

nnenc.encode_points_mtx_nd(bottom,axis=1)
for _ in range(6):
    print('fjkfd')
    print(nnenc.cc
) 
print('############')


exit()
#/////////////////////////////////////////////////////////
import matplotlib.pyplot as plt

n = 1024
# x = np.random.normal(0, 1, n)  # 平均值为0，方差为1，生成1024个数
# y = np.random.normal(0, 1, n)
x = X[:,0]
y = X[:,1]

t = np.arctan2(x, y)  # for color value，对应cmap

plt.scatter(x, y, s=65, c=t, alpha=0.5)   # s为size，按每个点的坐标绘制，alpha为透明度
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xticks([])
plt.yticks([])
plt.show()
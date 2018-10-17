from glob import glob
import numpy as np
from  os.path import join
import sklearn.neighbors as neighbors
from skimage import color
from skimage.io import imread
import time

root = '/home/chuchienshu/Documents/propagation_refine/data/sintel_test_clean/'
filename_lists = sorted(glob(join(root, '*/*.png')))

# print(filename_lists)
# CREATE BIN GRIDS
hull = []
i = -120
for _ in range(25):
    a = [i]
    j = -120
    for _ in range(25):
        for k in range(25):
            a.append(j)
            
            j += 10
            if k == 24:
                j = -120
            
            hull.append(a)
            a = [i]
            break
    i += 10

# print(len(hull))
hull = np.array(hull)
# print(hull.shape)

# mark_dict is used to record the number that pixels close to each bin. 
mark_dict = dict()
for i in range(len(hull)):
    mark_dict[i] = 0

# print(mark_dict)
nbrs = neighbors.NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(hull)
start = time.time()
print (time.asctime( time.localtime(time.time()) ))
# l_50 here for record the number of whole ab pairs under condition l = 50.
l_50 = []
for img_f in filename_lists:
    img = imread(img_f)
    img = color.rgb2lab(img)
    indexs = np.argwhere( np.around(img[:,:,0]) == 50 )
    # return the channel a collection according to the given index collection.
    a_ = [img[:,:,1][indexs[:,0], indexs[:,1]]]
    b_ = [img[:,:,2][indexs[:,0], indexs[:,1]]]
    ab_ = np.concatenate( (a_, b_)).transpose((1,0))
    l_50 += list(ab_)
    print(len(l_50))

    # for ind in indexs:
    #     l_50.append(img[ind[0], ind[1], 1:])
    # break

(_,inds) = nbrs.kneighbors(l_50)

for i in mark_dict.keys():
    num = np.sum(inds == i)
    mark_dict[i] += num

sorted_mark_dict = sorted(mark_dict.items(), key=lambda a:a[1], reverse=True)
log = open('log.txt', 'a')
for j,k in sorted_mark_dict:
    log.write('%d %d \n' % (j,k))

NUM = 60 # set NUM by your need.
tranced = sorted_mark_dict[:NUM]

for_save = []
for j, _ in tranced:
    for_save.append(hull[j])
# print(for_save)

np.save('my_hull', for_save)
# for k,j in mark_dict.items():
#     print(k, ' ',j)
print (time.asctime( time.localtime(time.time()) ))
print('time went on %s seconds' % (time.time() - start))
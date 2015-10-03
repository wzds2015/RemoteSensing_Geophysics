import numpy as np
import math
import matplotlib.pyplot as plt


def quadtree(data,th,max_l):
    '''
        function used for generating a quadtree
        data (float): 2D image file (numpy array)
        th (float): threshold for standard deviation
        max_l (int): max level of leave (step=2**max_l)
    '''

    line, col = data.shape[0], data.shape[1]
    dim = 1
    if line >= col:
        dim_data = line
    else:
        dim_data = col
    while (dim<dim_data):
        dim *= 2
    print "dim is: ", dim
    level_max = int(math.log(dim,2))
    data_big = np.zeros((dim,dim),dtype=np.float32) * np.nan
    data_big[:line,:col] = data
    check = np.zeros((dim,dim),dtype=np.int16)
    level = np.copy(check)
    level[:,:] = level_max
    center_x = np.copy(check)
    center_y = np.copy(check)
    k = 1
    for temp_level in range(max_l,0,-1):
        step = 2**temp_level
        for ni in range(0,dim,step):
            for nj in range(0,dim,step):
                temp_data = data_big[ni:ni+step,nj:nj+step]
                temp_data = temp_data[~np.isnan(temp_data)]
                if (0 in check[ni:ni+step,nj:nj+step]):
                    if temp_data.size > 30 or temp_data.size == step**2:
                        temp_std = np.std(temp_data)
                        if temp_std < th:
                            check[ni:ni+step,nj:nj+step] = k
                            k += 1
                            level[ni:ni+step,nj:nj+step] = temp_level
                            data_big[ni:ni+step,nj:nj+step] = np.average(temp_data)
                            center_y[ni:ni+step,nj:nj+step] = ni + step/2.
                            center_x[ni:ni+step,nj:nj+step] = nj + step/2.
                    elif temp_data.size == 0:
                        check[ni:ni+step,nj:nj+step] = -10
                        level[ni:ni+step,nj:nj+step] = temp_level
                        center_y[ni:ni+step,nj:nj+step] = ni + step/2.
                        center_x[ni:ni+step,nj:nj+step] = nj + step/2.
                        data_big[ni:ni+step,nj:nj+step] = np.nan

    for ni in range(0,dim):
        for nj in range(0,dim):
            if check[ni,nj] == 0:
                center_y[ni,nj] = ni + 0.5
                center_x[ni,nj] = nj + 0.5
                check[ni,nj] = k
                level[ni,nj] = 0
                k += 1

    center_x = center_x.reshape(dim*dim,1)
    center_y = center_y.reshape(dim*dim,1)
    level = level.reshape(dim*dim,1)
    check1, indice_c = np.unique(check, return_index=True)
    ind = np.where(check1 == -10)
    indice_c = np.delete(indice_c,ind)
    check1 = np.delete(check1,ind)
    if -10 in check1:
        print "Array still contains nan! Check again.\n"
        exit(1)
    center_x1 = center_x[indice_c]
    center_y1 = center_y[indice_c]
    level1 = level[indice_c]

    return level1,center_x1,center_y1,data_big

def plot_QDT(level,center_x,center_y,data):
    '''
        function used for plotting a quadtree
    '''

    v_min = data[~np.isnan(data)].min()
    v_max = data[~np.isnan(data)].max()
    fig, ax = plt.subplots(1)
    im = ax.imshow(data,cmap=plt.cm.jet, vmin=v_min, vmax=v_max)
    for ni in range(level.size):
        step = 2 ** level[ni]
        '''
        if step >= 32:
            print str(ni) + " is wrong\n"
            fig.close()
            exit(1)
        '''
        upleft_x = center_x[ni] - step/2.
        upleft_y = center_y[ni] - step/2. + 0.5
        upright_x = center_x[ni] + step/2.
        upright_y = center_y[ni] - step/2. + 0.5
        lowright_x = center_x[ni] + step/2.
        lowright_y = center_y[ni] + step/2. + 0.5
        lowleft_x = center_x[ni] - step/2.
        lowleft_y = center_y[ni] + step/2. + 0.5
        ax.plot([upleft_x,upright_x],[upleft_y,upright_y],'k-',linewidth=0.4)
        ax.plot([upright_x,lowright_x],[upright_y,lowright_y],'k-',linewidth=0.4)
        ax.plot([lowright_x,lowleft_x],[lowright_y,lowleft_y],'k-',linewidth=0.4)
        ax.plot([lowleft_x,upleft_x],[lowleft_y,upleft_y],'k-',linewidth=0.4)

    plt.savefig('QDT.png')

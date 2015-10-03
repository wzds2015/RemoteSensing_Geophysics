# -*- coding: utf-8 -*-

'''
Created on Mon Oct 27 03:16:06 2014

@author: Wenliang Zhao
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pylab
import math
from scipy.sparse.linalg import lsqr
from scipy.optimize import fmin_slsqp



class ice_cap:
    
    def __init__(self,y,x,step):
        self.dim_y                  = y
        self.dim_x                  = x
        self.step                   = step 
        self.ice_load               = None
        self.load_outline           = None
        self.load_distance          = None
        self.disp                   = None
        self.ang_inc                = None
        self.ang_azi                = None
        self.qt_ice                 = self.qt_ice()
        self.qt_disp                = self.qt_disp()
        self.model_inverse          = None
        self.density                = None
        self.young                  = None
        self.poisson                = None
        self.altimetry_load         = None
        self.design_mat             = None
        
    class qt_ice:
        def __init__(self):
            self.level          = None
            self.center_x       = None
            self.center_y       = None
            self.data_big       = None
            self.distance_small = None
            self.neighbor       = None
            self.altimetry      = None
            
    class qt_disp:
        def __init__(self):
            self.level      = None
            self.center_x   = None
            self.center_y   = None
            self.data_big   = None
            self.data_small = None
        
    def simulate_disk(self, center_y, center_x, radius):
        x1, y1 = np.linspace(0,self.dim_x,self.dim_x), np.linspace(0,self.dim_y,self.dim_y)
        x2, y2 = np.meshgrid(x1,y1)
        distance = np.power(np.power(y2 - center_y,2) + np.power(x2 - center_x, 2),0.5) * self.step
        distance[distance>=radius] = np.nan
        self.ice_load = (7.2 * np.exp(distance / 30000.)) / 50 - 2
        
    def outline(self):
        lo = np.copy(self.ice_load)
        lo[~np.isnan(lo)] = 1
        lo[np.isnan(lo)] = 0
        lo = lo.astype('int16')
        temp_mat = np.zeros((self.dim_y+2,self.dim_x+2),dtype=np.int16)
        temp_mat[1:-1,1:-1] = np.copy(lo)
        for ni in range(1,self.dim_y+1):
            for nj in range(1,self.dim_x+1):
                temp_mat1 = temp_mat[ni-1:ni+2,nj-1:nj+2]
                if np.unique(temp_mat1).size == 1:
                    lo[ni-1,nj-1] = 0
                else:
                    lo[ni-1,nj-1] = 1
        return lo
                
    def dis_to_edge(self):
        lo = np.copy(self.ice_load)
        lo[~np.isnan(lo)] = 1
        x1, y1 = np.linspace(0,self.dim_x,self.dim_x), np.linspace(0,self.dim_y,self.dim_y)
        x2, y2 = np.meshgrid(x1,y1)
        x3,y3 = np.copy(x2), np.copy(y2)
        x2, y2 = x2[~np.isnan(lo)], y2[~np.isnan(lo)]
        x3, y3 = x3[self.load_outline==1], y3[self.load_outline==1]
        for ni in range(x2.size):
            lo[y2[ni],x2[ni]] = np.min(np.power(np.power(y2[ni]-y3,2) + np.power(x2[ni]-x3,2), 0.5)) * self.step
        
        return lo
    
    def simulate_altimetry(self,n_line,noise_std):
        altimetry_array = np.array([])
        fig = plt.figure()
        v_min = self.ice_load[~np.isnan(self.ice_load)].min()
        v_max = self.ice_load[~np.isnan(self.ice_load)].max()
        im1 = plt.imshow(self.ice_load, cmap=plt.cm.jet, vmin=v_min, vmax=v_max)
        plt.ylim((0,self.dim_y))
        plt.xlim((0,self.dim_x))
        for ni in range(n_line):
            pts = fig.ginput(2)
            pts = np.array(pts)
            pts[0][0] = round(pts[0][0])
            pts[0][1] = round(pts[0][1])
            pts[1][0] = round(pts[1][0])
            pts[1][1] = round(pts[1][1])
            x = np.array([pts[0][0],pts[1][0]])
            y = np.array([pts[0][1],pts[1][1]])
            plt.plot(x,y,color='red',linewidth=2.0, linestyle="-")
            
            if np.abs(x[1]-x[0])<=1 and np.abs(y[1]-y[0])<=1:
                print "The space is too small. Please reselect points!\n"
                exit(1)
            else:
                if np.abs(x[1]-x[0]) >= np.abs(y[1]-y[0]):
                    temp_step = int(abs(x[1]-x[0])-1)
                    temp_x = np.linspace(float(x[0]),float(x[1]),temp_step+2)
                    if y[1]-y[0]==0:
                        temp_y = np.ones((1,temp_step)) * y[0]
                    else:
                        temp_y = np.linspace(float(y[0]),float(y[1]),temp_step+2)
                elif np.abs(x[1]-x[0]) < np.abs(y[1]-y[0]):
                    temp_step = int(abs(y[1]-y[0])-1)
                    temp_y = np.linspace(float(y[0]),float(y[1]),temp_step+2)
                    if x[1]-x[0]==0:
                        temp_x = np.ones((1,temp_step)) * x[0]
                    else:
                        temp_x = np.linspace(float(x[0]),float(x[1]),temp_step+2)
                    temp_x = temp_x.round()
                    temp_y = temp_y.round()
            temp_ind = temp_y * self.dim_x + temp_x
            altimetry_array = np.hstack((altimetry_array,temp_ind))
            
        plt.close()
        altimetry_array = np.unique(altimetry_array)
        altimetry_y = altimetry_array.astype('int16') / self.dim_x
        altimetry_x = np.remainder(altimetry_array.astype('int16'),self.dim_x)
        ind = []
        for ni in range(altimetry_array.size):
            if np.isnan(self.ice_load[altimetry_y[ni],altimetry_x[ni]]):
                ind.append(ni)
        ind = np.array(ind)
        np.delete(altimetry_y,ind)
        np.delete(altimetry_x,ind)
        temp_noise = np.random.normal(0.0,0.2,altimetry_array.size)
        self.altimetry_load = np.zeros((self.dim_y,self.dim_x),dtype=np.float32) * np.nan
        self.altimetry_load[altimetry_y,altimetry_x] = self.ice_load[altimetry_y,altimetry_x] + temp_noise
        
        
    def proj2insar(self,fz,fx,fn,lookAng,azimuth):
        '''
        function used to project 3D displacement components onto LOS
        lookAng: InSAR incidence angle in degree, always positive and less than 90 (float)
        azimuth: InSAR azimuth angle, positive clockwise from North (float)
        '''
    
        dz = np.fromfile(fz,dtype=np.float32)
        dx = np.fromfile(fx,dtype=np.float32)
        dn = np.fromfile(fn,dtype=np.float32)
        lookAng = lookAng / 180.0 * np.pi
        azimuth = azimuth / 180.0 * np.pi
        self.disp = (dz * np.cos(lookAng) - dx * np.sin(lookAng) * np.cos(azimuth) + dn * np.sin(lookAng) * np.sin(azimuth)).reshape(self.dim_y,self.dim_x)
                
                
    def quadtree(self,th,max_l,N_nan,target='model',extra_list=[]):
        '''
        function used for generating a quadtree
        data (float): 2D image file (numpy array)
        th (float): threshold for standard deviation
        max_l (int): max level of leave (step=2**max_l)
        N_nan: small value to give tolerance of a few nan in the window; large value for ice model decoposition
        '''
        if target == 'data':       
            data = self.disp
        elif target == 'model':
            data = self.ice_load
            distance = self.load_distance
        elif target == 'altimetry':
            data = self.ice_load
            distance = self.load_distance
            altimetry_data = self.altimetry_load
            
        line, col = data.shape[0], data.shape[1]
        dim = 1
        if line >= col:
            dim_data = line
        else:
            dim_data = col
        level_max = int(math.ceil(math.log(dim_data,2)))
        dim = 2 ** level_max
        
        print "dim is: ", dim
        data_big = np.zeros((dim,dim),dtype=np.float32) * np.nan
        data_big[:line,:col] = data
        if target == 'model':
            distance_small = np.copy(data_big)
            distance_small[:line,:col] = distance
            distance = None
        if target == 'altimetry':
            distance_small = np.copy(data_big)
            distance_small[:line,:col] = distance
            distance = None
            altimetry_small = np.copy(data_big)
            altimetry_small[:line,:col] = altimetry_data
            altimetry_data = None
          
        new_data_list = []
        if len(extra_list):
            new_file_list = []
            for ni in xrange(len(extra_list)):
                temp_data = extra_list[ni]
                temp_data1 = np.zeros((dim,dim),dtype=np.float32) * np.nan
                temp_data1[:line,:col] = temp_data
                new_data_list.append(temp_data1)           
        
        check = np.zeros((dim,dim),dtype=np.int16)
        level = np.copy(check)
        level[:,:] = level_max
        center_x = np.zeros((dim,dim), dtype=np.float32)
        center_y = np.copy(center_x)
        k = 1
        
        if target == 'data':
            for temp_level in range(max_l,-1,-1):
                step = 2**temp_level
                for ni in range(0,dim,step):
                    for nj in range(0,dim,step):
                        temp_data = data_big[ni:ni+step,nj:nj+step]
                        temp_data = temp_data[~np.isnan(temp_data)]
                        if (0 in check[ni:ni+step,nj:nj+step]):
                            if temp_data.size > N_nan and temp_data.size / float(step**2) > 0.3:
                                temp_std = np.std(temp_data)
                                temp_th = th
                                if temp_std <= temp_th:
                                    check[ni:ni+step,nj:nj+step] = k
                                    k += 1
                                    level[ni:ni+step,nj:nj+step] = temp_level
                                    data_big[ni:ni+step,nj:nj+step] = np.average(temp_data)
                                    if len(extra_list):
                                        for nk in xrange(len(extra_list)):
                                            temp_data1 = new_data_list[nk][ni:ni+step,nj:nj+step]
                                            temp_data1 = temp_data1[~np.isnan(temp_data1)]
                                            new_data_list[nk][ni:ni+step,nj:nj+step] = np.average(temp_data1)
                                    center_y[ni:ni+step,nj:nj+step] = ni + step/2.
                                    center_x[ni:ni+step,nj:nj+step] = nj + step/2.
                            
                            elif temp_data.size == 0:
                                check[ni:ni+step,nj:nj+step] = -10             ### level -10 represents NAN
                                level[ni:ni+step,nj:nj+step] = temp_level
                                center_y[ni:ni+step,nj:nj+step] = ni + step/2.
                                center_x[ni:ni+step,nj:nj+step] = nj + step/2.
                                data_big[ni:ni+step,nj:nj+step] = np.nan
                                if len(extra_list):
                                    for nk in xrange(len(extra_list)):
                                        new_data_list[nk][ni:ni+step,nj:nj+step] = np.nan
                                
        elif target == 'model':
            for temp_level in range(max_l,-1,-1):
                step = 2**temp_level
                for ni in range(0,dim,step):
                    for nj in range(0,dim,step):
                        temp_data = data_big[ni:ni+step,nj:nj+step]
                        temp_dis  = distance_small[ni:ni+step,nj:nj+step]
                        temp_data = temp_data[~np.isnan(temp_data)]
                        if (0 in check[ni:ni+step,nj:nj+step]):
                            if temp_data.size == step ** 2 or temp_data.size > N_nan:
                                temp_std = np.std(temp_data)
                                if temp_std <= th:
                                    check[ni:ni+step,nj:nj+step] = k
                                    k += 1
                                    level[ni:ni+step,nj:nj+step] = temp_level
                                    data_big[ni:ni+step,nj:nj+step] = np.average(temp_data)
                                    distance_small[ni:ni+step,nj:nj+step] = np.average(temp_dis)
                                    center_y[ni:ni+step,nj:nj+step] = ni + step/2.
                                    center_x[ni:ni+step,nj:nj+step] = nj + step/2.
                            
                            elif temp_data.size == 0:
                                check[ni:ni+step,nj:nj+step] = -10             ### level -10 represents NAN
                                level[ni:ni+step,nj:nj+step] = temp_level
                                center_y[ni:ni+step,nj:nj+step] = ni + step/2.
                                center_x[ni:ni+step,nj:nj+step] = nj + step/2.
                                data_big[ni:ni+step,nj:nj+step] = np.nan
                                
        elif target == 'altimetry':
            for temp_level in range(max_l,-1,-1):
                step = 2**temp_level
                for ni in range(0,dim,step):
                    for nj in range(0,dim,step):
                        temp_data = data_big[ni:ni+step,nj:nj+step]
                        temp_dis  = distance_small[ni:ni+step,nj:nj+step]
                        temp_altimetry = altimetry_small[ni:ni+step,nj:nj+step]
                        temp_data = temp_data[~np.isnan(temp_data)]
                        if (0 in check[ni:ni+step,nj:nj+step]):
                            if temp_data.size == step ** 2 or temp_data.size > N_nan:
                                temp_std = np.std(temp_data)
                                if temp_std <= th:
                                    check[ni:ni+step,nj:nj+step] = k
                                    k += 1
                                    level[ni:ni+step,nj:nj+step] = temp_level
                                    data_big[ni:ni+step,nj:nj+step] = np.average(temp_data)
                                    distance_small[ni:ni+step,nj:nj+step] = np.average(temp_dis)
                                    altimetry_small[ni:ni+step,nj:nj+step] = np.average(temp_altimetry)
                                    center_y[ni:ni+step,nj:nj+step] = ni + step/2.
                                    center_x[ni:ni+step,nj:nj+step] = nj + step/2.

                            elif temp_data.size == 0:
                                check[ni:ni+step,nj:nj+step] = -10             ### level -10 represents NAN
                                level[ni:ni+step,nj:nj+step] = temp_level
                                center_y[ni:ni+step,nj:nj+step] = ni + step/2.
                                center_x[ni:ni+step,nj:nj+step] = nj + step/2.
                                data_big[ni:ni+step,nj:nj+step] = np.nan

        
        for ni in xrange(0,dim):
            for nj in xrange(0,dim):
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
        if 9 in level1:
            print "Yes this is 512 in levels. Check the code again!\n"
            exit(1)
            
        if target == 'data':
            data_small = np.copy(data_big)
            data_small = data_small.reshape(dim*dim,1)
            data_small = data_small[indice_c]
            self.qt_disp.level = level1
            self.qt_disp.center_x = center_x1
            self.qt_disp.center_y = center_y1
            self.qt_disp.data_big = data_big
            self.qt_disp.data_small = data_small
            if len(extra_list):
                self.qt_disp.data_big = new_data_list[0]
                data_small = np.copy(new_data_list[0])
                data_small = data_small.reshape(dim*dim,1)
                data_small = data_small[indice_c]
                self.qt_disp.data_small = data_small
            
        elif target == 'model':
            distance_small = distance_small.reshape(dim*dim,1)
            distance_small = distance_small[indice_c]
            self.qt_ice.level = level1
            self.qt_ice.center_x = center_x1
            self.qt_ice.center_y = center_y1
            self.qt_ice.data_big = data_big
            self.qt_ice.distance_small = distance_small
            
        elif target == 'altimetry':
            distance_small = distance_small.reshape(dim*dim,1)
            distance_small = distance_small[indice_c]
            altimetry_small = altimetry_small.reshape(dim*dim,1)
            altimetry_small = altimetry_small[indice_c]
            self.qt_ice.level = level1
            self.qt_ice.center_x = center_x1
            self.qt_ice.center_y = center_y1
            self.qt_ice.data_big = data_big
            self.qt_ice.distance_small = distance_small
            self.qt_ice.altimetry = altimetry_small 
        
     
    def plot_QDT(self,target='model'):
        '''
        function used for plotting a quadtree
        '''
        if target == 'data':
            data = self.qt_disp.data_big
            center_x = self.qt_disp.center_x
            center_y = self.qt_disp.center_y
            level = self.qt_disp.level
            
        elif target == 'model':
            data = self.qt_ice.data_big
            center_x = self.qt_ice.center_x
            center_y = self.qt_ice.center_y
            level = self.qt_ice.level
        
        if 8 in level:
            print "This is a bug!\n"
            
        v_min = data[~np.isnan(data)].min()
        v_max = data[~np.isnan(data)].max()
        
        ### temporal setting for test
        if target == 'data':
            v_min, v_max = -40, 40
        
        fig = plt.figure()
        ax = plt.axes()
        im = plt.imshow(data,cmap=plt.cm.jet, vmin=v_min, vmax=v_max)
        
        for ni in range(level.size):
            step = 2 ** level[ni]
        
            upleft_x = center_x[ni] - step/2. - 0.5
            upleft_y = center_y[ni] - step/2. - 0.5
            upright_x = center_x[ni] + step/2. - 0.5
            upright_y = center_y[ni] - step/2. - 0.5
            lowright_x = center_x[ni] + step/2. - 0.5
            lowright_y = center_y[ni] + step/2. - 0.5
            lowleft_x = center_x[ni] - step/2. - 0.5
            lowleft_y = center_y[ni] + step/2. - 0.5
            ax.plot([upleft_x,upright_x],[upleft_y,upright_y],'k-',linewidth=0.4)
            ax.plot([upright_x,lowright_x],[upright_y,lowright_y],'k-',linewidth=0.4)
            ax.plot([lowright_x,lowleft_x],[lowright_y,lowleft_y],'k-',linewidth=0.4)
            ax.plot([lowleft_x,upleft_x],[lowleft_y,upleft_y],'k-',linewidth=0.4)
        
        
        temp_shape = data.shape[0]
        if data.shape[1] > temp_shape:
            temp_shape = data.shape[1]
        
        temp_dim = 2**(math.ceil(math.log(temp_shape,2)))
        ax.set_ylim([temp_dim, 0])
        ax.set_xlim([0, temp_dim])
        plt.colorbar()
        plt.show()
        #plt.savefig('QDT.png')
        
    def inverse_EHS_QT(self):
        
        coord_x_model = (self.qt_ice.center_x).astype('float32')
        coord_y_model = (self.qt_ice.center_y).astype('float32')
        level_model = (self.qt_ice.level).astype('int32')
        temp_array = np.ones((level_model.size,1)) * 2
        size_model = np.power(np.power(temp_array, level_model)*self.step, 2)
        print "size for each block is: ", size_model
        coord_x_data = (self.qt_disp.center_x).astype('float32')
        coord_y_data = (self.qt_disp.center_y).astype('float32')
        value_data = (self.qt_disp.data_small).astype('float64')
        ro, E, mu = self.density, self.young, self.poisson
        g = 9.81
        N_pix = coord_x_data.size
        N_ice = coord_x_model.size
        #ang_azi = ang_azi / 180.0 * np.pi
        #ang_inc - ang_inc / 180.0 * np.pi
        
        const_v = ro * g * (1-mu**2) / E / np.pi
        const_r = ro * g * (1+mu) * (1-2*mu) / E / 2 / np.pi
        self.design_mat = np.zeros((N_pix,N_ice),dtype=np.float64)  
        print "parameters ro,young,mu,N_pix,N_ice,inc,azi,const_v,const_r: ", ro,",",E,",",mu,",",N_pix,",",N_ice,",",self.ang_inc,",",self.ang_azi,",",const_v,",",const_r
        
        for ni in range(N_pix):
            temp_dis = (np.power(np.power(coord_y_data[ni]-coord_y_model,2)+np.power(coord_x_data[ni]-coord_x_model,2),0.5) + 0.5) * self.step 
            temp_sin = (coord_x_data[ni] - coord_x_model) / temp_dis
            temp_cos = -1 * (coord_y_data[ni] - coord_y_model) / temp_dis
            temp_design = (const_v*np.cos(self.ang_inc)/temp_dis*size_model - \
                       const_r*np.sin(self.ang_inc)*np.cos(self.ang_azi)*temp_sin/temp_dis*size_model + \
                       const_r*np.sin(self.ang_inc)*np.sin(self.ang_azi)*temp_cos/temp_dis*size_model).flatten()

            self.design_mat[ni,:] = temp_design
        
        temp_para = lsqr(self.design_mat,value_data.reshape(N_pix,1))[0]
        print "temp_para size is: ", temp_para.shape
        self.model_inverse = np.zeros((self.dim_y,self.dim_x),dtype=np.float32)
        for ni in range(temp_para.size):
            temp_step = 2 ** level_model[ni]
            temp_start_coord_x = round(coord_x_model[ni] - temp_step / 2)
            temp_start_coord_y = round(coord_y_model[ni] - temp_step / 2)
            self.model_inverse[temp_start_coord_y:temp_start_coord_y+temp_step, temp_start_coord_x:temp_start_coord_x+temp_step] = temp_para[ni]
        self.model_inverse[self.model_inverse==0] = np.nan
    
    class neighbor:
        n_list = None
        def __init__(self,n):
            self.n_list = n   ## n: array like
        
    def count_neighbor(self):
        coord_x_model = (self.qt_ice.center_x).astype('float64')
        coord_y_model = (self.qt_ice.center_y).astype('float64')
        level_model = self.qt_ice.level   
        step_mat = np.power(2,level_model)
        if 0 in step_mat:
            print "Yes there is 0!\n"
        print "step_mat is: ", step_mat
        start_mat_x = coord_x_model - (step_mat / 2.)
        start_mat_y = coord_y_model - (step_mat / 2.)
        end_mat_x = coord_x_model + (step_mat / 2.)
        end_mat_y = coord_y_model + (step_mat / 2.)
        print "start_x is: ", start_mat_x, "\n", "end_x is: ", end_mat_x
        
        neighbor_mat = [0] * level_model.size
        
        for ni in range(level_model.size):
            ind1 = np.where( (np.abs(start_mat_x[ni] - end_mat_x)<0.0001) & ( ((start_mat_y <= start_mat_y[ni]) & (end_mat_y >= end_mat_y[ni])) | ((start_mat_y >= start_mat_y[ni]) & (end_mat_y <= end_mat_y[ni])) ) )[0]
            ind2 = np.where( (np.abs(start_mat_y[ni] - end_mat_y)<0.0001) & ( ((start_mat_x <= start_mat_x[ni]) & (end_mat_x >= end_mat_x[ni])) | ((start_mat_x >= start_mat_x[ni]) & (end_mat_x <= end_mat_x[ni])) ) )[0]
            ind3 = np.where( (np.abs(end_mat_x[ni] - start_mat_x)<0.0001) & ( ((start_mat_y <= start_mat_y[ni]) & (end_mat_y >= end_mat_y[ni])) | ((start_mat_y >= start_mat_y[ni]) & (end_mat_y <= end_mat_y[ni])) ) )[0]
            ind4 = np.where( (np.abs(end_mat_y[ni] - start_mat_y)<0.0001) & ( ((start_mat_x <= start_mat_x[ni]) & (end_mat_x >= end_mat_x[ni])) | ((start_mat_x >= start_mat_x[ni]) & (end_mat_x <= end_mat_x[ni])) ) )[0]

            ind = np.concatenate((ind1,ind2,ind3,ind4), axis=0)
            ind = np.unique(ind)
            neighbor_mat[ni] = self.neighbor(ind)
        
        return neighbor_mat
            
        
    def inverse_EHS_QT_laplace(self,w_dis1,w_dis2):
        
        coord_x_model = self.qt_ice.center_x
        coord_y_model = self.qt_ice.center_y
        level_model = self.qt_ice.level
        neighbor_mat = self.count_neighbor()
        self.qt_ice.neighbor = neighbor_mat
        temp_array = np.ones((level_model.size,1)) * 2
        size_model = np.power(np.power(temp_array, level_model)*self.step, 2)
        print "size for each block is: ", size_model
        coord_x_data = self.qt_disp.center_x
        coord_y_data = self.qt_disp.center_y
        value_data1 = self.qt_disp.data_small
        ro, E, mu = self.density, self.young, self.poisson
        g = 9.81
        N_pix = coord_x_data.size
        N_ice = coord_x_model.size
        #ang_azi = ang_azi / 180.0 * np.pi
        #ang_inc - ang_inc / 180.0 * np.pi
        
        const_v = ro * g * (1-mu**2) / E / np.pi
        const_r = ro * g * (1+mu) * (1-2*mu) / E / 2 / np.pi
        D1 = np.zeros((N_pix,N_ice),dtype=np.float64)  
        D2 = np.zeros((N_ice,N_ice),dtype=np.float64)
        print "parameters ro,young,mu,N_pix,N_ice,inc,azi,const_v,const_r: ", ro,",",E,",",mu,",",N_pix,",",N_ice,",",self.ang_inc,",",self.ang_azi,",",const_v,",",const_r
        
        for ni in range(N_pix):
            temp_dis = (np.power(np.power(coord_y_data[ni]-coord_y_model,2)+np.power(coord_x_data[ni]-coord_x_model,2),0.5) + 0.5) * self.step 
            temp_sin = (coord_x_data[ni] - coord_x_model) / temp_dis
            temp_cos = -1 * (coord_y_data[ni] - coord_y_model) / temp_dis
            temp_design = (const_v*np.cos(self.ang_inc)/temp_dis*size_model - \
                       const_r*np.sin(self.ang_inc)*np.cos(self.ang_azi)*temp_sin/temp_dis*size_model + \
                       const_r*np.sin(self.ang_inc)*np.sin(self.ang_azi)*temp_cos/temp_dis*size_model).flatten()
            D1[ni,:] = temp_design
        
        for ni in range(N_ice):
            for nj in range(neighbor_mat[ni].n_list.size):
                temp_dis1 = math.sqrt((coord_x_model[ni] - coord_x_model[neighbor_mat[ni].n_list[nj]]) ** 2 + \
                (coord_y_model[ni] - coord_y_model[neighbor_mat[ni].n_list[nj]]) ** 2) / 1000.    ### on scale of 10 km
                #temp_weight = (w_dis2 * (self.qt_load_distance_small[ni]/1000.)) * (w_dis1 / temp_dis1)
                temp_weight = w_dis1 / np.power(temp_dis1,0.5)
                D2[ni,ni] += temp_weight   ### dis1 for distance between 2 pixel, dis2 for distance from the edge
                D2[ni,neighbor_mat[ni].n_list[nj]] = -1 * temp_weight 
            
        D = np.vstack((D1,D2))
        value_data2 = np.zeros((N_ice,1),dtype=np.float64)
        value_data = np.vstack((value_data1,value_data2))
        #print "lenght is: ", value_data.size, " and length of real number is: ", value_data[~np.isnan(value_data)].size
        #plt.figure()
        #im = plt.plot(value_data)
        #plt.show()
        #exit(0)        
        
        temp_para = lsqr(D,value_data)[0]
        print "temp_para size is: ", temp_para.shape
        self.model_inverse = np.zeros((self.dim_y,self.dim_x),dtype=np.float32)
        for ni in range(temp_para.size):
            temp_step = 2 ** level_model[ni]
            temp_start_coord_x = round(coord_x_model[ni] - temp_step / 2.)
            temp_start_coord_y = round(coord_y_model[ni] - temp_step / 2.)
            self.model_inverse[temp_start_coord_y:temp_start_coord_y+temp_step, temp_start_coord_x:temp_start_coord_x+temp_step] = temp_para[ni]
        self.model_inverse[self.model_inverse==0] = np.nan
        
    
    
    def inverse_EHS_QT_constrain(self,w_dis1,w_dis2):
        def func(x):
            x = x.reshape(x.size,1)
            s = np.sum( np.power((np.dot(D,x) - value_data),2) )
            return s
            
        def func_deriv(x):
            x = x.reshape(1,x.size)
            d = (np.sum(D,axis=0) * 2 * x).flatten()
            return d
            
        
        coord_x_model = self.qt_ice.center_x
        coord_y_model = self.qt_ice.center_y
        level_model = self.qt_ice.level
        neighbor_mat = self.count_neighbor()
        self.qt_ice.neighbor = neighbor_mat
        temp_array = np.ones((level_model.size,1)) * 2
        size_model = np.power(np.power(temp_array, level_model)*self.step, 2)
        print "size for each block is: ", size_model
        coord_x_data = self.qt_disp.center_x
        coord_y_data = self.qt_disp.center_y
        value_data1 = self.qt_disp.data_small
        ro, E, mu = self.density, self.young, self.poisson
        g = 9.81
        N_pix = coord_x_data.size
        N_ice = coord_x_model.size
        #ang_azi = ang_azi / 180.0 * np.pi
        #ang_inc - ang_inc / 180.0 * np.pi
        
        const_v = ro * g * (1-mu**2) / E / np.pi
        const_r = ro * g * (1+mu) * (1-2*mu) / E / 2 / np.pi
        D1 = np.zeros((N_pix,N_ice),dtype=np.float64)  
        D2 = np.zeros((N_ice,N_ice),dtype=np.float64)
        print "parameters ro,young,mu,N_pix,N_ice,inc,azi,const_v,const_r: ", ro,",",E,",",mu,",",N_pix,",",N_ice,",",self.ang_inc,",",self.ang_azi,",",const_v,",",const_r
        
        for ni in range(N_pix):
            temp_dis = (np.power(np.power(coord_y_data[ni]-coord_y_model,2)+np.power(coord_x_data[ni]-coord_x_model,2),0.5) + 0.5) * self.step 
            temp_sin = (coord_x_data[ni] - coord_x_model) / temp_dis
            temp_cos = -1 * (coord_y_data[ni] - coord_y_model) / temp_dis
            temp_design = (const_v*np.cos(self.ang_inc)/temp_dis*size_model - \
                       const_r*np.sin(self.ang_inc)*np.cos(self.ang_azi)*temp_sin/temp_dis*size_model + \
                       const_r*np.sin(self.ang_inc)*np.sin(self.ang_azi)*temp_cos/temp_dis*size_model).flatten()
            D1[ni,:] = temp_design
        
        for ni in range(N_ice):
            for nj in range(neighbor_mat[ni].n_list.size):
                temp_dis1 = math.sqrt((coord_x_model[ni] - coord_x_model[neighbor_mat[ni].n_list[nj]]) ** 2 + \
                (coord_y_model[ni] - coord_y_model[neighbor_mat[ni].n_list[nj]]) ** 2) / 1000.    ### on scale of 10 km
                #temp_weight = (w_dis2 * (self.qt_load_distance_small[ni]/1000.)) * (w_dis1 / temp_dis1)
                temp_weight = w_dis1 / np.power(temp_dis1,0.5)
                D2[ni,ni] += temp_weight   ### dis1 for distance between 2 pixel, dis2 for distance from the edge
                D2[ni,neighbor_mat[ni].n_list[nj]] = -1 * temp_weight 
            
        D = np.vstack((D1,D2))
        value_data2 = np.zeros((N_ice,1),dtype=np.float64)
        value_data = np.vstack((value_data1,value_data2))
        
        x0 = lsqr(D,value_data)[0]
        bounds = []
        for ni in range(self.qt_ice.altimetry.size):
            temp_tuple = (self.qt_ice.altimetry[ni]-0.2,self.qt_ice.altimetry[ni]+0.2)
            bounds.append(temp_tuple)
        
        print "shapes are: x0: ", x0.size, ", bounds: ", len(bounds), ", D: ", D.shape
        #temp_para = fmin_slsqp(func=func,x0=x0,fprime=func_deriv,bounds=bounds,iter=850,acc=1.0e-9)
        temp_para = fmin_slsqp(func=func,x0=x0,bounds=bounds,iter=200,acc=1.0e-4)
        #temp_para = res.x
        
        print "temp_para size is: ", temp_para.shape
        self.model_inverse = np.zeros((self.dim_y,self.dim_x),dtype=np.float32)
        for ni in range(temp_para.size):
            temp_step = 2 ** level_model[ni]
            temp_start_coord_x = round(coord_x_model[ni] - temp_step / 2.)
            temp_start_coord_y = round(coord_y_model[ni] - temp_step / 2.)
            self.model_inverse[temp_start_coord_y:temp_start_coord_y+temp_step, temp_start_coord_x:temp_start_coord_x+temp_step] = temp_para[ni]
        self.model_inverse[self.model_inverse==0] = np.nan
    
    
    
    def inverse_EHS_QT_constrain_self(self,w_dis1,w_dis2,range_list):
        def func(x):
            x = x.reshape(x.size,1)
            s = np.sum( np.power((np.dot(D,x) - value_data),2) )
            return s
            
        def func_deriv(x):
            x = x.reshape(1,x.size)
            d = (np.sum(D,axis=0) * 2 * x).flatten()
            return d
            
        
        coord_x_model = self.qt_ice.center_x
        coord_y_model = self.qt_ice.center_y
        level_model = self.qt_ice.level
        neighbor_mat = self.count_neighbor()
        self.qt_ice.neighbor = neighbor_mat
        temp_array = np.ones((level_model.size,1)) * 2
        size_model = np.power(np.power(temp_array, level_model)*self.step, 2)
        #print "size for each block is: ", size_model
        coord_x_data = self.qt_disp.center_x
        coord_y_data = self.qt_disp.center_y
        value_data1 = self.qt_disp.data_small
        ro, E, mu = self.density, self.young, self.poisson
        g = 9.81
        N_pix = coord_x_data.size
        N_ice = coord_x_model.size
        #ang_azi = ang_azi / 180.0 * np.pi
        #ang_inc - ang_inc / 180.0 * np.pi
        
        const_v = ro * g * (1-mu**2) / E / np.pi
        const_r = ro * g * (1+mu) * (1-2*mu) / E / 2 / np.pi
        D1 = np.zeros((N_pix,N_ice),dtype=np.float64)  
        D2 = np.zeros((N_ice,N_ice),dtype=np.float64)
        print "parameters ro,young,mu,N_pix,N_ice,inc,azi,const_v,const_r: ", ro,",",E,",",mu,",",N_pix,",",N_ice,",",self.ang_inc,",",self.ang_azi,",",const_v,",",const_r
        
        for ni in range(N_pix):
            temp_dis = (np.power(np.power(coord_y_data[ni]-coord_y_model,2)+np.power(coord_x_data[ni]-coord_x_model,2),0.5) + 0.5) * self.step 
            temp_sin = (coord_x_data[ni] - coord_x_model) / temp_dis
            temp_cos = -1 * (coord_y_data[ni] - coord_y_model) / temp_dis
            temp_design = (const_v*np.cos(self.ang_inc)/temp_dis*size_model - \
                       const_r*np.sin(self.ang_inc)*np.cos(self.ang_azi)*temp_sin/temp_dis*size_model + \
                       const_r*np.sin(self.ang_inc)*np.sin(self.ang_azi)*temp_cos/temp_dis*size_model).flatten()
            D1[ni,:] = temp_design
        
        for ni in range(N_ice):
            for nj in range(neighbor_mat[ni].n_list.size):
                temp_dis1 = math.sqrt((coord_x_model[ni] - coord_x_model[neighbor_mat[ni].n_list[nj]]) ** 2 + \
                (coord_y_model[ni] - coord_y_model[neighbor_mat[ni].n_list[nj]]) ** 2) / 1000.    ### on scale of 10 km
                #temp_weight = (w_dis2 * (self.qt_load_distance_small[ni]/1000.)) * (w_dis1 / temp_dis1)
                temp_weight = w_dis1 / np.power(temp_dis1,0.5)
                D2[ni,ni] += temp_weight   ### dis1 for distance between 2 pixel, dis2 for distance from the edge
                D2[ni,neighbor_mat[ni].n_list[nj]] = -1 * temp_weight 
            
        D = np.vstack((D1,D2))
        value_data2 = np.zeros((N_ice,1),dtype=np.float64)
        value_data = np.vstack((value_data1,value_data2))
        
        print "shapes of D, value_data are: ", D.shape, " and ", value_data.shape
        x0 = lsqr(D,value_data)[0]
        bounds = []
        #self.qt_ice.altimetry = np.copy(self.qt_ice.level)
        for ni in range(self.qt_ice.level.size):
            temp_tuple = (range_list[0],range_list[1])
            bounds.append(temp_tuple)
        
        print "shapes are: x0: ", x0.size, ", bounds: ", len(bounds), ", D: ", D.shape
        #temp_para = fmin_slsqp(func=func,x0=x0,fprime=func_deriv,bounds=bounds,iter=850,acc=1.0e-5)
        #x0[x0<range_list[0]] = 0
        #x0[x0>range_list[1]] = 0
        temp_para = fmin_slsqp(func=func,x0=x0,bounds=bounds,iter=40,acc=1.0e-9)
        
        print "temp_para size is: ", temp_para.shape
        self.model_inverse = np.zeros((self.dim_y,self.dim_x),dtype=np.float32)
        for ni in range(temp_para.size):
            temp_step = 2 ** level_model[ni]
            temp_start_coord_x = round(coord_x_model[ni] - temp_step / 2.)
            temp_start_coord_y = round(coord_y_model[ni] - temp_step / 2.)
            self.model_inverse[temp_start_coord_y:temp_start_coord_y+temp_step, temp_start_coord_x:temp_start_coord_x+temp_step] = temp_para[ni]
        self.model_inverse[self.model_inverse==0] = np.nan
    
    '''
    def inverse_EHS_QT_constrain_self(self,w_dis1,w_dis2,range_list):
        def func(x):
            x = x.reshape(x.size,1)
            s = np.sum( np.power((np.dot(D,x) - value_data),2) )
            return s
            
        def func_deriv(x):
            x = x.reshape(1,x.size)
            d = (np.sum(D,axis=0) * 2 * x).flatten()
            return d
                    
        coord_x_model = self.qt_ice.center_x
        coord_y_model = self.qt_ice.center_y
        level_model = self.qt_ice.level
        neighbor_mat = self.count_neighbor()
        self.qt_ice.neighbor = neighbor_mat
        temp_array = np.ones((level_model.size,1)) * 2
        size_model = np.power(np.power(temp_array, level_model)*self.step, 2)
        #print "size for each block is: ", size_model
        coord_x_data = self.qt_disp.center_x
        coord_y_data = self.qt_disp.center_y
        value_data1 = self.qt_disp.data_small
        ro, E, mu = self.density, self.young, self.poisson
        g = 9.81
        N_pix = coord_x_data.size
        N_ice = coord_x_model.size
        #ang_azi = ang_azi / 180.0 * np.pi
        #ang_inc - ang_inc / 180.0 * np.pi
        
        const_v = ro * g * (1-mu**2) / E / np.pi
        const_r = ro * g * (1+mu) * (1-2*mu) / E / 2 / np.pi
        D1 = np.zeros((N_pix,N_ice),dtype=np.float64)  
        D2 = np.zeros((N_ice,N_ice),dtype=np.float64)
        print "parameters ro,young,mu,N_pix,N_ice,inc,azi,const_v,const_r: ", ro,",",E,",",mu,",",N_pix,",",N_ice,",",self.ang_inc,",",self.ang_azi,",",const_v,",",const_r
        
        for ni in range(N_pix):
            temp_dis = (np.power(np.power(coord_y_data[ni]-coord_y_model,2)+np.power(coord_x_data[ni]-coord_x_model,2),0.5) + 0.5) * self.step 
            temp_sin = (coord_x_data[ni] - coord_x_model) / temp_dis
            temp_cos = -1 * (coord_y_data[ni] - coord_y_model) / temp_dis
            temp_design = (const_v*np.cos(self.ang_inc)/temp_dis*size_model - \
                       const_r*np.sin(self.ang_inc)*np.cos(self.ang_azi)*temp_sin/temp_dis*size_model + \
                       const_r*np.sin(self.ang_inc)*np.sin(self.ang_azi)*temp_cos/temp_dis*size_model).flatten()
            D1[ni,:] = temp_design
        
        for ni in range(N_ice):
            for nj in range(neighbor_mat[ni].n_list.size):
                temp_dis1 = math.sqrt((coord_x_model[ni] - coord_x_model[neighbor_mat[ni].n_list[nj]]) ** 2 + \
                (coord_y_model[ni] - coord_y_model[neighbor_mat[ni].n_list[nj]]) ** 2) / 1000.    ### on scale of 10 km
                #temp_weight = (w_dis2 * (self.qt_load_distance_small[ni]/1000.)) * (w_dis1 / temp_dis1)
                temp_weight = w_dis1 / np.power(temp_dis1,0.5)
                D2[ni,ni] += temp_weight   ### dis1 for distance between 2 pixel, dis2 for distance from the edge
                D2[ni,neighbor_mat[ni].n_list[nj]] = -1 * temp_weight 
            
        D = np.vstack((D1,D2))
        value_data2 = np.zeros((N_ice,1),dtype=np.float64)
        value_data = np.vstack((value_data1,value_data2))
        
        print "shapes of D, value_data are: ", D.shape, " and ", value_data.shape
        x0 = lsqr(D,value_data)[0]
        bounds = []
        #self.qt_ice.altimetry = np.copy(self.qt_ice.level)
        for ni in range(self.qt_ice.level.size):
            temp_tuple = (range_list[0],range_list[1])
            bounds.append(temp_tuple)
        
        print "shapes are: x0: ", x0.size, ", bounds: ", len(bounds), ", D: ", D.shape
        temp_para = fmin_slsqp(func=func,x0=x0, ieqcons=[lambda w: np.dot(D2[ni,:],w) - 0.001 if np.dot(D2[ni,:],w) >= 0 else np.dot(D2[ni,:],w) + 0.001 for ni in range(D2.shape[0])], fprime=func_deriv,bounds=bounds,iter=850,acc=1.0e-12)
        #temp_para = fmin_slsqp(func=func,x0=x0, ieqcons=[lambda w: np.dot(D2[ni,:],w) + 0.001 for ni in range(D2.shape[0])], fprime=func_deriv,bounds=bounds,iter=850,acc=1.0e-12)        
        #temp_para = res.x
        
        print "temp_para size is: ", temp_para.shape
        self.model_inverse = np.zeros((self.dim_y,self.dim_x),dtype=np.float32)
        for ni in range(temp_para.size):
            temp_step = 2 ** level_model[ni]
            temp_start_coord_x = round(coord_x_model[ni] - temp_step / 2.)
            temp_start_coord_y = round(coord_y_model[ni] - temp_step / 2.)
            self.model_inverse[temp_start_coord_y:temp_start_coord_y+temp_step, temp_start_coord_x:temp_start_coord_x+temp_step] = temp_para[ni]
        self.model_inverse[self.model_inverse==0] = np.nan
    '''
        
    #### New test
    def inverse_EHS_QT_constrain_self1(self,w_dis1,w_dis2,range_list):
        def func(x):
            x = x.reshape(x.size,1)
            s = np.sum( np.power((np.dot(D,x) - value_data),2) )
            return s
            
        def func_deriv(x):
            x = x.reshape(1,x.size)
            d = (np.sum(D,axis=0) * 2 * x).flatten()
            return d
                    
        coord_x_model = self.qt_ice.center_x
        coord_y_model = self.qt_ice.center_y
        level_model = self.qt_ice.level
        neighbor_mat = self.count_neighbor()
        self.qt_ice.neighbor = neighbor_mat
        temp_array = np.ones((level_model.size,1)) * 2
        size_model = np.power(np.power(temp_array, level_model)*self.step, 2)
        #print "size for each block is: ", size_model
        coord_x_data = self.qt_disp.center_x
        coord_y_data = self.qt_disp.center_y
        value_data1 = self.qt_disp.data_small
        ro, E, mu = self.density, self.young, self.poisson
        g = 9.81
        N_pix = coord_x_data.size
        N_ice = coord_x_model.size
        #ang_azi = ang_azi / 180.0 * np.pi
        #ang_inc - ang_inc / 180.0 * np.pi
        
        const_v = ro * g * (1-mu**2) / E / np.pi
        const_r = ro * g * (1+mu) * (1-2*mu) / E / 2 / np.pi
        D1 = np.zeros((N_pix,N_ice),dtype=np.float64)  
        D2 = np.zeros((N_ice,N_ice),dtype=np.float64)
        print "parameters ro,young,mu,N_pix,N_ice,inc,azi,const_v,const_r: ", ro,",",E,",",mu,",",N_pix,",",N_ice,",",self.ang_inc,",",self.ang_azi,",",const_v,",",const_r
        
        for ni in range(N_pix):
            temp_dis = (np.power(np.power(coord_y_data[ni]-coord_y_model,2)+np.power(coord_x_data[ni]-coord_x_model,2),0.5) + 0.5) * self.step 
            temp_sin = (coord_x_data[ni] - coord_x_model) / temp_dis
            temp_cos = -1 * (coord_y_data[ni] - coord_y_model) / temp_dis
            temp_design = (const_v*np.cos(self.ang_inc)/temp_dis*size_model - \
                       const_r*np.sin(self.ang_inc)*np.cos(self.ang_azi)*temp_sin/temp_dis*size_model + \
                       const_r*np.sin(self.ang_inc)*np.sin(self.ang_azi)*temp_cos/temp_dis*size_model).flatten()
            D1[ni,:] = temp_design
        
        for ni in range(N_ice):
            for nj in range(neighbor_mat[ni].n_list.size):
                temp_dis1 = math.sqrt((coord_x_model[ni] - coord_x_model[neighbor_mat[ni].n_list[nj]]) ** 2 + \
                (coord_y_model[ni] - coord_y_model[neighbor_mat[ni].n_list[nj]]) ** 2) / 1000.    ### on scale of 10 km
                #temp_weight = (w_dis2 * (self.qt_load_distance_small[ni]/1000.)) * (w_dis1 / temp_dis1)
                temp_weight = w_dis1 / np.power(temp_dis1,0.5)
                D2[ni,ni] += temp_weight   ### dis1 for distance between 2 pixel, dis2 for distance from the edge
                D2[ni,neighbor_mat[ni].n_list[nj]] = -1 * temp_weight 
            
        D = np.vstack((D1,D2))
        value_data2 = np.zeros((N_ice,1),dtype=np.float64)
        value_data = np.vstack((value_data1,value_data2))
        
        print "shapes of D, value_data are: ", D.shape, " and ", value_data.shape
        x0 = lsqr(D,value_data)[0]
        bounds = []
        #self.qt_ice.altimetry = np.copy(self.qt_ice.level)
        for ni in range(self.qt_ice.level.size):
            temp_tuple = (range_list[0],range_list[1])
            bounds.append(temp_tuple)
        
        print "shapes are: x0: ", x0.size, ", bounds: ", len(bounds), ", D: ", D.shape
        x0[x0<range_list[0]] = 0
        x0[x0>range_list[1]] = 0
        #temp_para = fmin_slsqp(func=func,x0=x0, ieqcons=[lambda w: np.dot(D2[ni,:],w) - 0.001 if np.dot(D2[ni,:],w) >= 0 else np.dot(D2[ni,:],w) + 0.001 for ni in range(D2.shape[0])], fprime=func_deriv,bounds=bounds,iter=850,acc=1.0e-12)
        #temp_para = fmin_slsqp(func=func,x0=x0, ieqcons=[lambda w: (0.2 * math.sqrt((np.max(self.qt_ice.distance_small)/1000+1) / (self.qt_ice.distance_small[ni]/1000+1)) - np.std(np.hstack((w[ni],w[np.array(neighbor_mat[ni].n_list)])))) for ni in range(D2.shape[0])], fprime=func_deriv,bounds=bounds,iter=1000,acc=1.0e-9)   
        temp_para = fmin_slsqp(func=func,x0=x0, ieqcons=[lambda w: (0.2 * math.sqrt((np.max(self.qt_ice.distance_small)/1000+1) / (self.qt_ice.distance_small[ni]/1000+1)) - np.std(np.hstack((w[ni],w[np.array(neighbor_mat[ni].n_list)])))) for ni in range(D2.shape[0])], bounds=bounds,iter=20,acc=1.0e-4)
        
        #temp_para = (temp_para * (-1) + 1) * 5.
        print "temp_para size is: ", temp_para.shape
        self.model_inverse = np.zeros((self.dim_y,self.dim_x),dtype=np.float32)
        for ni in range(temp_para.size):
            temp_step = 2 ** level_model[ni]
            temp_start_coord_x = round(coord_x_model[ni] - temp_step / 2.)
            temp_start_coord_y = round(coord_y_model[ni] - temp_step / 2.)
            self.model_inverse[temp_start_coord_y:temp_start_coord_y+temp_step, temp_start_coord_x:temp_start_coord_x+temp_step] = temp_para[ni]
        self.model_inverse[self.model_inverse==0] = np.nan
        #self.model_inverse = (self.model_inverse * (-1) + 1.) * 5.
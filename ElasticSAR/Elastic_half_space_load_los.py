#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 19:20:35 2013

@author: Wenliang Zhao
"""

import sys
import Rsmas_Utilib as ut
import numpy as np
from mpi4py import MPI
#from mpi4py.MPI import ANY_SOURCE

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:

        procfile = sys.argv[1].strip()
        para_list = ut.readProc(procfile)
        ice_model_file = para_list.get("ice_model").strip()    ### load model
        ice_model_col = int(para_list.get("ice_model_col").strip())
        grid_size = float(para_list.get("grid_size").strip())  ### grid size in m
        r = float(para_list.get("r").strip())     ### density
        v = float(para_list.get("v").strip())     ### Poisson's ratio
        E = int(para_list.get("E").strip())       ### Young's modulus
        E_unit = para_list.get("E_unit").strip()
        if E_unit == 'MPa':
            E = E * (10 ** 6)
        elif E_unit == 'GPa':
            E = E * (10 ** 9)
        else:
            print "Unit of Young's modulus should be either MPa or GPa!\n"
            exit(1)
        print "Young's modulus is: ", E
        angle_az = float(para_list.get("angle_az").strip())
        angle_inc = float(para_list.get("angle_inc").strip())

        ice_model_data = np.fromfile(ice_model_file,dtype=np.float32)
        ice_model_lin = ice_model_data.size / ice_model_col
        dis_model_v = np.zeros((ice_model_data.size),dtype=np.float32)
        #dis_model_e = np.zeros((ice_model_data.size),dtype=np.float32).reshape(ice_model_lin,ice_model_col)
        #dis_model_n = np.zeros((ice_model_data.size),dtype=np.float32).reshape(ice_model_lin,ice_model_col)
        g = 9.81

        vec_col = np.linspace(0,ice_model_col-1, ice_model_col)
        vec_lin = np.linspace(0,ice_model_lin-1, ice_model_lin)
        x1, y1 = np.meshgrid(vec_col, vec_lin)
        x2, y2 = x1, y1

        x1 = x1.reshape(x1.size)
        y1 = y1.reshape(y1.size)
        x2 = x2.reshape(ice_model_data.size)[np.abs(ice_model_data)>0.00001]
        y2 = y2.reshape(ice_model_data.size)[np.abs(ice_model_data)>0.00001]
        ice_model_data = ice_model_data[np.abs(ice_model_data)>0.00001]

        const_v = grid_size ** 2 * r * g * (1-v**2) / E / np.pi
        const_r = grid_size ** 2 * r * g * (1+v) * (1-2*v) / E / 2 / np.pi
        print "2 constants are: ", const_v, " and ", const_r

        dis_model_v_batch = np.array_split(dis_model_v,comm.size)
        dis_model_v = None
        dis_model_e_batch = dis_model_v_batch
        dis_model_n_batch = dis_model_v_batch
        x1_batch = np.array_split(x1,size)
        y1_batch = np.array_split(y1,size)
    
    else:
        ice_model_data = None
        dis_model_v_batch = None
        dis_model_e_batch = None
        dis_model_n_batch = None
        x2 = None
        y2 = None
        const_v = None
        const_r = None
        x1_batch = None
        y1_batch = None
        grid_size = None
    
    const_v = comm.bcast(const_v, root=0)
    const_r = comm.bcast(const_r, root=0)
    x2 = comm.bcast(x2, root=0)
    y2 = comm.bcast(y2, root=0)
    grid_size = comm.bcast(grid_size, root=0)
    ice_model_data = comm.bcast(ice_model_data, root=0)
    dis_model_v_batch = comm.scatter(dis_model_v_batch, root=0)
    dis_model_e_batch = comm.scatter(dis_model_e_batch, root=0)
    dis_model_n_batch = comm.scatter(dis_model_n_batch, root=0)
    x1_batch = comm.scatter(x1_batch, root=0)
    y1_batch = comm.scatter(y1_batch, root=0)
    
    N_pix = len(x1_batch)
    print N_pix, " data received!\n"
    print "Shape of dis_model_v_batch is: ", dis_model_v_batch.shape
    
    
    
    for ni in range(N_pix):
        line1 = y1_batch[ni]
        col1 = x1_batch[ni]
        dis_y = line1 - y2
        dis_x = col1 - x2
        distance = np.power(np.power(dis_y,2) + np.power(dis_x,2), 0.5)
        distance[distance<1] = 10000000.
        sin_ang = dis_x / distance
        cos_ang = -1 * dis_y / distance
        distance = distance * grid_size
        dis_model_v_batch[ni] = np.sum(const_v * (1 / distance) * ice_model_data)
        dis_model_e_batch[ni] = np.sum(const_r * (1 /distance) * ice_model_data * sin_ang)
        dis_model_e_batch[ni] = np.sum(const_r * (1 /distance) * ice_model_data * cos_ang)
    
    
    dis_model_v_batch = comm.gather(dis_model_v_batch, root=0)
    dis_model_e_batch = comm.gather(dis_model_e_batch, root=0)
    dis_model_n_batch = comm.gather(dis_model_n_batch, root=0)
    
    if rank == 0:
        dis_model_v = dis_model_v_batch[0]
        dis_model_e = dis_model_e_batch[0]
        dis_model_n = dis_model_n_batch[0]
        for ni in range(size-1):
            dis_model_v = np.hstack((dis_model_v, dis_model_v_batch[ni+1]))
            dis_model_e = np.hstack((dis_model_e, dis_model_e_batch[ni+1]))
            dis_model_n = np.hstack((dis_model_n, dis_model_n_batch[ni+1]))
         
        dis_model_los = dis_model_v * np.cos(angle_inc/180*np.pi) - \
        dis_model_e * np.sin(angle_inc/180*np.pi) * np.cos(angle_az/180*np.pi) + \
        dis_model_n * np.sin(angle_inc/180*np.pi) * np.sin(angle_az/180*np.pi)
        
        dis_model_los.astype('float32').tofile('elastic_half_space_load_los_model')

if __name__ == "__main__":
    sys.exit(main())         

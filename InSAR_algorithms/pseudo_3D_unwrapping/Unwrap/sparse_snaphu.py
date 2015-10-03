#! /usr/bin/env python

''' File used for sparse network unwrapping based on snaphu
    WZ, Mar. 2014
'''
import os
import re
import sys
import Rsmas_Utilib as ut
import numpy as np
from scipy import interpolate
import matplotlib.cm
import matplotlib.pyplot as plt
import subprocess
import glob


if len(sys.argv) != 2:
    print '''
    Usage: sparse_snaphu.py int_folder  
    template must include path to interferogram
    
'''
    exit(0)

int_folder = sys.argv[1].strip()
os.system("cd " + int_folder)
procfile = int_folder + "/sparse_para"
procfile1 = int_folder + "/snaphu_para"
para_list = ut.readProc(procfile)
para_list1 = ut.readProc(procfile1)

if para_list.get("interferogram"):
    intfile = para_list.get("interferogram").strip() 
else:
    print "Must have path to interferogram!\n"
    exit(0)   

if para_list.get("coherence"):
    corfile = para_list.get("coherence").strip()
else:
    print "Must have path to coherence!\n"
    exit(0)


N_line = int(para_list.get("FILE_LENGTH").strip())
N_col = int(para_list.get("WIDTH").strip())
N_looks = para_list.get("rlks").strip()
N_looks1 = int(N_looks[:-5])
ratio_p = int(para_list.get("pixel_ratio").strip())    
    
### read interferogram ###
[amplitude_int, phase_int, rscContent_int,l_int,w_int] = ut.readInt(intfile)
 
### read coherence ###
[amplitude_cor, phase_cor, rscContent_cor,l,w] = ut.read_bifloat(corfile)
length_cor = phase_cor.shape[0]
print "length_cor is: ", length_cor, " and N_line is: ", N_line,"\n"
if para_list.get("mask"):
    maskfile = para_list.get("mask")
    [phase_M, rscContent_M,l_M,w_M] = ut.read_float(maskfile)
    if l_M>length_cor:
        temp_array = np.zeros((l_M-length_cor,w_M),dtype=np.float)
        phase_cor = np.vstack((phase_cor,temp_array),dtype=np.float)
        amplitude_cor = np.vstack((amplitude_cor,temp_array),dtype=np.float)
        temp_array1 = np.zeros((N_line*2,N_col),dtype=np.float32)
        temp_array1[::2,:] = amplitude_cor
        temp_array1[1::2,:]   = phase_cor 
        new_length = l_M
        rscContent_cor["FILE_LENGTH"] = new_length
        os.system("rm -f "+corfile)
        temp_array1.astype('float32').tofile(corfile)
        cor_rsc = corfile + ".rsc"
        os.system("rm -f "+cor_rsc)
        IN = open(cor_rsc, 'w')
        for k,v in rscContent_cor.iteritems():
            IN.write(k.ljust(41)+v+"\n")
        IN.close()

    elif l_M < length_cor
        print "Yes we are here\n"
        temp_N = length_cor - l_M
        phase_cor = phase_cor[:l_M,:]
        amplitude_cor = amplitude_cor[:l_M,:] 
        temp_array1 = np.zeros((N_line*2,N_col),dtype=np.float32)
        temp_array1[::2,:] = amplitude_cor
        temp_array1[1::2,:]   = phase_cor
        new_length = l_M
        rscContent_cor["FILE_LENGTH"] = new_length
        os.system("rm -f "+corfile)
        temp_array1.astype('float32').tofile(corfile)
        cor_rsc = corfile + ".rsc"
        os.system("rm -f "+cor_rsc)
        IN = open(cor_rsc, 'w')
        for k,v in rscContent_cor.iteritems():
            IN.write(k.ljust(41)+v+"\n")
        IN.close()
    

    
    ### replace amplitude of interferogram by corhence's
    amplitude_int = np.copy(amplitude_cor) 
   

    if para_list.get("mask"):
        maskfile = para_list.get("mask")
        [phase_M, rscContent_M,l_M,w_M] = ut.read_float(maskfile)
        if l_M>l_int:
            temp_array = np.zeros((l_M-l_int,w_int),dtype=np.float)
            phase_int = np.vstack((phase_int,temp_array),dtype=np.float)
            amplitude_int = np.vstack((amplitude_int,temp_array),dtype=np.float)
            temp_array1 = np.zeros((l_M*2,N_col),dtype=np.float32)
            temp_array1[::2,:] = amplitude_int
            temp_array1[1::2,:]   = phase_int
            new_length = l_M
            rscContent_int["FILE_LENGTH"] = new_length
        phase_int[np.isnan(phase_M)] = np.nan
    if para_list.get("cor_th"):
        th = float(para_list.get("cor_th"))
        phase_int[phase_cor<th] = np.nan

 
    ### read SLC1 ###
    SLC1file = para_list.get("SLC1").strip()
    date1 = para_list.get("date1").strip()
    date2 = para_list.get("date2").strip()
    temp_str = os.getenv("INT_SCR").strip() + "/look.pl " + SLC1file + " " + N_looks1 + " " + N_looks1*ratio_p 
    sys.stderr.write(temp_str+"\n")
    subprocess.Popen(temp_str).wait()
    SLC1file_multi = glob.glob(int_folder+"/"+date1+"*"+N_looks+"*.slc")[0]
    amplitude_slc1_name = int_folder + "/amplitude1"
    [amplitude_slc1, phase_slc1, rscContent_slc1,l,w] = ut.readInt(SLC1file_multi)
    length_slc_1 = phase_slc1.shape[0]
    
    if l_M>l:
        temp_array = np.zeros((l_M-l,w),dtype=np.float)
        amplitude_slc1 = np.vstack((amplitude_slc1,temp_array),dtype=np.float32)
        new_length = l_M
        rscContent_slc1["FILE_LENGTH"] = new_length    
    amplitude_slc1.astype('float32').tofile(amplitude_slc1_name)
    rscContent_slc1["FILE_LENGTH"] = N_line
    SLC1rsc_multi = SLC1file_multi + ".rsc"
    os.system("rm -f "+SLC1rsc_multi)
    IN = open(SLC1rsc_multi, 'w')   
    for k,v in rscContent_slc1.iteritems():
        IN.write(k.ljust(41)+v+"\n")
    IN.close()
    
    
    ### read SLC2 ###
    SLC2file = para_list.get("SLC2").strip()
    date1 = para_list.get("date1").strip()
    temp_str = os.getenv("INT_SCR").strip() + "/look.pl " + SLC2file + " " + N_looks1 + " " + N_looks1*ratio_p 
    sys.stderr.write(temp_str+"\n")
    subprocess.Popen(temp_str).wait()
    SLC2file_multi = glob.glob(int_folder+"/"+date1+"*"+N_looks+"*.slc")[0]
    amplitude_slc2_name = int_folder + "/amplitude2"
    [amplitude_slc2, phase_slc2, rscContent_slc2,l,w] = ut.readInt(SLC2file_multi)
    length_slc_2 = phase_slc2.shape[0]
    if l_M>l:
        temp_array = np.zeros((l_M-l,w),dtype=np.float32)
        amplitude_slc2 = np.vstack((amplitude_slc2,temp_array),dtype=np.float32)
        new_length = l_M
        rscContent_slc2["FILE_LENGTH"] = new_length
    amplitude_slc2.astype('float32').tofile(amplitude_slc2_name)
    rscContent_slc2["FILE_LENGTH"] = N_line
    SLC2rsc_multi = SLC2file_multi + ".rsc"
    os.system("rm -f "+SLC2rsc_multi)
    IN = open(SLC2rsc_multi, 'w')
    for k,v in rscContent_slc2.iteritems():
        IN.write(k.ljust(41)+v+"\n")
    IN.close()
    
else:
    print "No SLC file defined!\n"

#if para_list.get("mask"):
#    maskfile = para_list.get("mask")
#    [phase_M, rscContent_M,l_M,w_M] = ut.read_float(maskfile)
#    if l_M>l_int:
#        temp_array = np.zeros((l_M-l_int,w_int),dtype=np.float)
#        phase_int = np.vstack((phase_int,temp_array),dtype=np.float)
#        amplitude_int = np.vstack((amplitude_int,temp_array),dtype=np.float)
#        temp_array1 = np.zeros((l_M*2,N_col),dtype=np.float32)
#        temp_array1[::2,:] = amplitude_int
#        temp_array1[1::2,:]   = phase_int 
#        new_length = l_M
#        rscContent_int["FILE_LENGTH"] = new_length
#    phase_int[np.isnan(phase_M)] = np.nan
#if para_list.get("cor_th"):
#    th = float(para_list.get("cor_th"))
#    phase_int[phase_cor<th] = np.nan

#temp = phase_int[np.invert(np.isnan(phase_int))]
#print "min phase is: ", temp.min(),"\n"
#plt.figure()
#im = plt.imshow(phase_int, cmap=matplotlib.cm.jet, vmin=-1*np.pi, vmax=np.pi)
#plt.show()

phase_int1 = phase_int.reshape((N_col*N_line,1))
temp_x = np.linspace(0,1,N_col)
temp_y = np.linspace(0,1,N_line)
grid_x,grid_y = np.meshgrid(temp_x,temp_y)
print "size of grid_x is: ", grid_x.shape, "\n"
print "size of int is: ", N_line, N_col, "\n"
grid_x1 = grid_x.reshape((N_col*N_line,1))
grid_y1 = grid_y.reshape((N_col*N_line,1))
grid_x2 = grid_x1[np.invert(np.isnan(phase_int1))]
grid_y2 = grid_y1[np.invert(np.isnan(phase_int1))]
values = phase_int1[np.invert(np.isnan(phase_int1))]
phase_int_new = interpolate.griddata((grid_x2,grid_y2), values, (grid_x, grid_y), method='nearest')
new_data = amplitude_int * np.exp(j*phase_int_new)
new_data_name = intfile[:-4] + "_NN.int" 
new_data.astype('complex64').tofile(new_data_name)
rsc_int = intfile + ".rsc"
rsc_int_new = new_data_name + ".rsc"
IN = open(rsc_int_new, 'w') 
if "DATE12" not in rscContent_int:       #### only consider a bug in NSBAS, no 'DATE12' in rsc after unwrapping
    rscContent_int["DATE12"] = date1[2:] + "-" + date2[2:] 
#if "DATE12" not in rscContent_int:
#    print "RSC file does have attribute DATE12!\n"
    exit(1)
for k,v in rscContent_int.iteritems():
    IN.write(k.ljust(41)+v+"\n")
IN.close()

temp_str = "cp " + rsc_int + " " + rsc_int_new
os.system(temp_str)
#plt.figure()
#im = plt.imshow(phase_int_new, cmap=matplotlib.cm.jet, vmin=-1*np.pi, vmax=np.pi)
#plt.show()

temp = amplitude_slc1[np.invert(np.isnan(amplitude_slc1))]
temp_max = temp.max()
temp_min = temp.min()
print "Max and min are: ", temp_max, temp_min,"\n"
#plt.figure()
#im = plt.imshow(amplitude_slc1, cmap=matplotlib.cm.gray, vmin=0, vmax=1000)
#plt.show()

##### snaphu unwrapping ##### 
default_para = os.getenv("INT_SCR") + "/snaphu.default"
snaphu_dic = ut.readProc(default_para)
for k, v in para_list1.items():
    ut.change_key(snaphu_dic,k,v)
IN = open("snaphu.conf", 'w')
for k, v in snaphu_dic.items():
    IN.write(k+"      "+v+"\n")
IN.close()
temp_str = "snaphu -v -f snaphu.conf"
sys.stderr.write(temp_str+"\n")
subprocess.Popen(temp_str).wait()

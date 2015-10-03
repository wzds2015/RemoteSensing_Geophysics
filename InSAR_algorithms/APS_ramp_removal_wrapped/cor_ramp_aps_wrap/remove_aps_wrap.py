#! /usr/bin/env python

import matplotlib.pyplot as plt
import os
import re
import sys
import Rsmas_Utilib as ut
import numpy as np
import glob
import subprocess

def main():
    
    global th_d, l_M, w_M, method, line1, col1, step
            
    
    intfile = sys.argv[1].strip()
    corfile = sys.argv[2].strip()
    maskfile = sys.argv[3].strip()
    demfile = sys.argv[4].strip()
    plane_type = int(sys.argv[5].strip())
    APS_type = int(sys.argv[6].strip())
    th_d = float(sys.argv[7].strip())
    th_c = float(sys.argv[8].strip())
    method = int(sys.argv[9].strip())
    solution = int(sys.argv[10].strip())
    iteration = int(sys.argv[11].strip())
    step = int(sys.argv[12].strip())
    print "All the parameters obtained!\n"
    print "Iteration is: ", str(iteration),"\n"
    ### read in mask file
    print "Reading maskfile\n"
    [phase_M, rscContent_M,l_M,w_M] = ut.read_float(maskfile)

    ### read in intfile corfile and demfile
    print "Reading interferogram and coherence!\n"
    print "intfile is: ", intfile
    N_looks = re.findall('\d+rlks',intfile)[0].strip()
    if len(N_looks) == 5:
        newName = intfile[:-9] + "ramp_aps_" + intfile[-9:]
    elif len(N_looks) == 6:
        newName = intfile[:-10] + "ramp_aps_" + intfile[-10:]
    if iteration != 1:
        intfile = newName
    
 
    [amplitude_int, phase_int, rscContent_int,l_int_temp,w_int] = ut.readInt(intfile)
    if l_M>l_int_temp:
        temp_array = np.zeros((l_M-l_int_temp,w_M),dtype=np.float)
    	phase_int = np.vstack((phase_int,temp_array))
    print "Coherence file is: ", corfile, "\n"
    [amplitude_cor, phase_cor, rscContent_cor,l_cor,w_cor] = ut.read_bifloat(corfile)
    print "Coherence size is: ", phase_cor.shape, "\n"
    if l_M>l_cor:
        temp_array = np.zeros((l_M-l_cor,w_M),dtype=np.float)
        phase_cor = np.vstack((phase_cor,temp_array))
    if l_M<l_cor:
        phase_cor = phase_cor[:l_M,:]
    #[phase_dem, rscContent_dem,l_dem,w_dem] = ut.read_float(demfile)
    [phase_dem, rscContent_dem,l_dem,w_dem] = ut.readDEM(demfile)
    if l_M>l_dem:
        temp_array = np.zeros((l_M-l_dem,w_M),dtype=np.float)
        phase_dem = np.vstack((phase_dem,temp_array))
      
    phase_temp = np.copy(phase_int)
    folder = os.path.dirname(intfile)
    os.chdir(folder)
    temp_dir = os.getcwd()
    print "Current dir is: ", temp_dir,"\n"
    temp_rsc = "temp.int.rsc"
    rscContent_int["FILE_LENGTH"] = str(l_M)
    IN = open(temp_rsc, 'w')
    for k,v in rscContent_int.iteritems():
        IN.write(k.ljust(41)+v+"\n")
    IN.close()

    temp_list = ["cp","-f",intfile+".rsc", "temp.int.rsc"]
    print temp_list, "\n"
    output = subprocess.Popen(temp_list).wait()
    temp_list = ["SWfilter_casc",intfile,corfile,"temp.int",str(w_M),str(l_int_temp),"0.01"]
    print temp_list, "\n"
    output = subprocess.Popen(temp_list).wait()
    if output == 0:
        print "success!\n"
    else:
        print "failed!\n"
        exit(0)
    intfile = "temp.int"
    [amplitude_int, phase_int, rscContent_int,l_int,w_int] = ut.readInt(intfile)
    if l_M>l_int:
        temp_array = np.zeros((l_M-l_int,w_M),dtype=np.float)
        phase_int = np.vstack((phase_int,temp_array))    
    if l_M<l_int:
        phase_int = phase_int[:l_M,:]

    ### mask phase and reshape files
    inData = np.copy(phase_int)
    phase_int = None
    inData[np.isnan(phase_M)] = np.nan
    inData[phase_cor<th_c] = np.nan
    inData[inData==0] = np.nan
    inData = inData.reshape(l_M,w_M)
    #phase_cor = None
    phase_dem = phase_dem.reshape(l_M,w_M)



    ### differentiation in range and azimuth
    line1 = l_M - step 
    col1 = w_M - step
    temp_x = np.linspace(0, w_M, w_M)
    temp_y = np.linspace(0, l_M, l_M)
    ran, azi = np.meshgrid(temp_x,temp_y)

    data1 = inData[:,step:] - inData[:,:(-1*step)]
    data11 = data1[:,step:] - data1[:,:(-1*step)]
    data2 = inData[step:,:] - inData[:(-1*step),:]
    data_r = np.vstack((data1.reshape(l_M*col1,1),data2.reshape(line1*w_M,1)))
    data_r[np.abs(data_r)>th_d] = np.nan
    length_r = data_r.size

    ran1 = ran[:,step:] - ran[:,:(-1*step)]
    ran2 = ran[step:,:] - ran[:(-1*step),:]
    range_t = ran.reshape(l_M*w_M,1)

    azi1 = azi[:,step:] - azi[:,:(-1*step)]
    azi2 = azi[step:,:] - azi[:(-1*step),:]
    azimuth_t = azi.reshape(l_M*w_M,1)
    azi2_sum = azi[step:,:] + azi[:(-1*step),:]

    phase_dem = phase_dem.astype('float')
    phase_dem[np.abs(phase_dem)>10000] = np.nan
    phase_dem[np.abs(phase_dem)<0.001] = np.nan
    dem1 = phase_dem[:,step:] - phase_dem[:,:(-1*step)]
    dem1[np.abs(dem1)>800] = np.nan
    #dem11 = dem1[:,step1:] - dem1[:,:(-1*step1)]
    #dem11[np.abs(dem11)>800] = np.nan
    #data11[np.isnan(dem11)] = np.nan
    #data11[np.abs(data11)>th_d] = np.nan
    #temp = data11 / dem11
    #temp[np.isinf(temp)] = np.nan
    #temp[np.abs(temp)>0.1] = np.nan
    #rate2 = np.mean(temp[~np.isnan(temp)])
    #data11 = None
    #dem11 = None
    #temp = None
    dem2 = phase_dem[step:,:] - phase_dem[:(-1*step),:]
    dem2[np.abs(dem2)>800] = np.nan    

    height = phase_dem.reshape(l_M*w_M,1)
    dem1_sum = phase_dem[:,step:] + phase_dem[:,:(-1*step)]
    dem2_sum = phase_dem[step:,:] + phase_dem[:(-1*step),:];
    dem_r = np.vstack((dem1.reshape(l_M*col1,1),dem2.reshape(line1*w_M,1)))
    mask1 = np.ones((length_r,1))
    mask1[np.isnan(data_r)] = np.nan
    mask1[np.isnan(dem_r)] = np.nan
    data_r[np.isnan(mask1)] = np.nan

    
    #### initial original design matrix
    if plane_type == 1:
        POINTS_ori = np.copy(range_t)
        if APS_type == 1:
            POINTS_ori = np.hstack((POINTS_ori,height))
        elif APS_type == 2:
            POINTS_ori = np.hstack((POINTS_ori,height))
            POINTS_ori = np.hstack((POINTS_ori,np.power(height,2)))
    elif plane_type == 2:
        POINTS_ori = np.copy(range_t)
        POINTS_ori = np.hstack((POINTS_ori,azimuth_t))
        if APS_type == 1:
            POINTS_ori = np.hstack((POINTS_ori,height))
        elif APS_type == 2:
            POINTS_ori = np.hstack((POINTS_ori,height))
            POINTS_ori = np.hstack((POINTS_ori,np.power(height,2)))
    elif plane_type == 3:
        POINTS_ori = np.copy(range_t)
        POINTS_ori = np.hstack((POINTS_ori,azimuth_t))
        POINTS_ori = np.hstack((POINTS_ori,np.power(azimuth_t,2)))
        if APS_type == 1:
            POINTS_ori = np.hstack((POINTS_ori,height))
        elif APS_type == 2:
            POINTS_ori = np.hstack((POINTS_ori,height))
            POINTS_ori = np.hstack((POINTS_ori,np.power(height,2)))
    else:
        if APS_type == 1:
            POINTS_ori = np.copy(height)
        elif APS_type == 2:
            POINTS_ori = np.copy(height)
            POINTS_ori = np.hstack((POINTS_ori,np.power(height,2)))

    azimuth_t = None
    range_t = None
    ran = None
    azi = None
    height = None

    ### initial design matrix
    if solution == 1:
        if plane_type == 1:
            POINTS = ran1.reshape((l_M*col1,1))
            data_r_temp = np.copy(data_r_temp[:ran1.size])
            mask1_temp = np.copy(mask1[:ran1.size])
            if APS_type == 1:
                POINTS = np.hstack((POINTS,dem1.reshape(l_M*col1,1)))
            elif APS_type == 2:
                POINTS = np.hstack((POINTS,dem1.reshape(l_M*col1,1)))
                POINTS = np.stack((POINTS,(dem1*dem1_sum).reshape(l_M*col1,1)))
            plane,APS,par = inverse_mat(POINTS,data_r_temp,mask1_temp,POINTS_ori,plane_type,solution)
        elif plane_type == 2:
            POINTS = np.copy(ran1.reshape((l_M*col1,1)))
            data_r_temp = np.copy(data_r[:ran1.size])
            mask1_temp = np.copy(mask1[:ran1.size])
            if APS_type == 1:
                POINTS = np.hstack((POINTS,dem1.reshape(l_M*col1,1)))
            elif APS_type == 2:
                POINTS = np.hstack((POINTS,dem1.reshape(l_M*col1,1)))
                POINTS = np.hstack((POINTS,(dem1*dem1_sum).reshape(l_M*col1,1)))
            if POINTS[0,:].size > 2:
                plane1,APS,par1 = inverse_mat(POINTS,data_r_temp,mask1_temp,POINTS_ori,plane_type,APS_type,solution)
            else:
                if method == 0:
                    par1 = np.mean(data_r_temp[np.invert(np.isnan(data_r_temp))]/step)
                else:
                    par1 = np.median(data_r_temp[np.invert(np.isnan(data_r_temp))]/step)
                plane1 = POINTS_ori[:,0] * par1
                APS = np.array([])
            if POINTS[0,:].size > 2:
                APS = APS.reshape(l_M,w_l)
                TT2 = APS[step:,:] - APS[:(-1*step),:]
                data_r_temp = data_r[ran1.size:] - TT2.reshape(line1*w_M,1)
                data_r_temp[np.abs(data_r_temp)>th_d] = np.nan
                if method == 0:
                    par2 = np.mean(data_r_temp[np.invert(np.isnan(data_r_temp))]/step)
                else:
                    par2 = np.median(data_r_temp[np.invert(np.isnan(data_r_temp))]/step)
                plane2 = POINTS_ori[:,1] * par2
                par = np.array([[par1[0]],[par2],[par1[1:]]])
            else:
                data_r_temp = np.copy(data_r[ran1.size:])
                if method == 0:
                    par2 = np.mean(data_r_temp[np.invert(np.isnan(data_r_temp))]/step)
                else:
                    par2 = np.median(data_r_temp[np.invert(np.isnan(data_r_temp))]/step)
            plane2 = POINTS_ori[:,1] * par2
            par = np.array([[par1],[par2]])
            plane = (plane1 + plane2).reshape(l_M,w_M)

        elif plane_type == 3:
            POINTS = np.copy(ran1.reshape(l_M*col1,1))
            data_r_temp = np.copy(data_r)
            data_r_temp = np.copy(data_r_temp[:ran1.size])
            mask1_temp = np.copy(mask1[:ran1.size])
            if APS_type == 1:
                POINTS = np.hstack((POINTS,dem1.reshape(l_M,col1)))
            elif APS_type == 2:
                POINTS = np.hstack((POINTS,dem1.reshape(l_M,col1)))
                POINTS = np.hstack((POINTS,(dem1*dem1_sum).reshape(l_M,col1)))
            plane_type = 4
            plane1,APS1,par1 = inverse_mat(POINTS,data_r_temp,mask1_temp,POINTS_ori,plane_type,APS_type,solution)
        
            POINTS = np.copy(azi2.reshape(line1*w_M,1))
            POINTS = np.hstack((POINTS,(azi2*azi2_sum).reshape(line1*w_M,1)))
            data_r_temp = np.copy(data_r[ran1.size:])
            mask1_temp = np.copy(mask1[ran1.size:])
            if APS_type == 1:
                POINTS = np.hstack((POINTS,dem2.reshape(line1*w_M,1)))
            elif APS_type == 2:
                POINTS = np.hstack((POINTS,dem2.reshape(line1*w_M,1)))
                POINTS = np.hstack((POINTS,(dem2*dem2_sum).reshape(line1*w_M,1)))

            plane2,APS2,par2 = inverse_mat(POINTS,data_r_temp,mask1_temp,POINTS_ori,plane_type,APS_type,solution)
    
            if APS_type == 1:
                par = np.array([[par1[0]],[par2[0]],[par2[1]],[(par1[1]+par2[2])/2]])
                plane = (plane1 + plane2).reshape(l_M,w_M)
                APS = ((APS1 + APS2) / 2).reshape(l_M,w_M)
            elif APS_type == 2:
                par = np.array([[par1[0]],[par2[0] ],[par2[1]],[(par1[1]+par2[2])/2],[(par1[2]+par2[3])/2]])
                plane = (plane1 + plane2).reshape(l_M,w_M)
                APS = ((APS1 + APS2) / 2).reshape(l_M,w_M)
            else:
                par = np.array([[par1],[par2]])
                plane = (plane1 + plane2).reshape(l_M,w_M)
                APS = np.array([])
         
    elif solution == 2:
        if plane_type == 1:
            data_r_temp = np.copy(data_r)
            mask1_temp = np.copy(mask1)
            if APS_type == 1:
                POINTS = np.vstack((dem1.reshape(l_M*col1,1),dem2.reshape(line1*w_M,1)))
            elif APS_type == 2:
                POINTS = np.vstack((dem1.reshape(l_M*col1,1),dem2.reshape(line1*w_M,1)))
                POINTS = np.hstack((POINTS,np.vstack(((dem1*dem1_sum).reshape(l_M*col1,1),(dem2*dem2_sum).reshape(line1*w_M,1)))))

            if APS_type > 0:
                plane,APS,par = inverse_mat(POINTS,data_r_temp,mask1_temp,POINTS_ori,plane_type,APS_type,solution)
                plane = plane.reshape(l_M,w_M)
            else:
                if method == 0:
                    par = np.mean(data_r[:ran1.size]/step)
                    plane = POINTS_ori[:,0] * par
                    APS = np.array([])
                else:
                    par = np.median(data_r[:ran1.size]/step)
                    plane = (POINTS_ori[:,0] * par).reshape(l_M,w_M)
                    APS = np.array([])

        elif plane_type == 2:
            data_r_temp = np.copy(data_r)
            mask1_temp = np.copy(mask1)
            if APS_type == 1:
                POINTS = np.vstack((dem1.reshape(l_M*col1,1),dem2.reshape(line1*w_M,1)))
            elif APS_type == 2:
                POINTS = np.vstack((dem1.reshape(l_M*col1,1),dem2.reshape(line1*w_M,1)))
                POINTS = np.hstack((POINTS,np.vstack(((dem1*dem1_sum).reshape(l_M*col1,1),(dem2*dem2_sum).reshape(line1*w_M,1)))))
            if APS_type > 0:
                print "size of POINTS: ", POINTS.size, ",data", data_r_temp.size,",mask", mask1_temp.size, ",plane_type",plane_type,",APS_type",APS_type,",solution",solution,"\n" 
                plane1,APS,par1 = inverse_mat(POINTS,data_r_temp,mask1_temp,POINTS_ori,plane_type,APS_type,solution)
            else:
                data_r_temp = data_r_temp[:ran1.size]
                if method == 0:
                    par1 = np.mean(data_r_temp[~np.isnan(data_r_temp)]/step)
                else:
                    par1 = np.median(data_r_temp[~np.isnan(data_r_temp)]/step)
                print "par1 is: ", par1, "\n"
                plane1 = POINTS_ori[:,0] * par1
                APS = np.array([])

            if APS_type > 0:
                APS = APS.reshape(l_M,w_M)
                TT2 = APS[step:,:] - APS[:(-1*step),:]
                data_r_temp = data_r[ran1.size:] - TT2.reshape(line1*w_M,1)
                data_r_temp[np.abs(data_r_temp)>th_d] = np.nan    
                if method == 0:
                    par2 = np.mean(data_r_temp[~np.isnan(data_r_temp)]/step)
                else:
                    par2 = np.median(data_r_temp[~np.isnan(data_r_temp)]/step)
                plane2 = POINTS_ori[:,1] * par2
                print "term1 is: ", par1[0],", term2 is: ", par2, ", term3 is: ", par1[1:],"\n"
                par = np.array([[par1[0]],[par2],[par1[1:]]])
            else:
                data_r_temp = np.copy(data_r[ran1.size:]) 
                if method == 0:
                    par2 = np.mean(data_r_temp[~np.isnan(data_r_temp)]/step)
                else:
                    par2 = np.median(data_r_temp[~np.isnan(data_r_temp)]/step)
                print "par2 is: ", par2,"\n"
                plane2 = POINTS_ori[:,1] * par2
                par = np.array([[par1],[par2]])
            plane = plane1 + plane2

        elif plane_type == 3:
            data_r_temp = np.copy(data_r)
            mask1_temp = np.copy(mask1)
            if APS_type == 1:
                POINTS = np.copy(dem1.reshape(l_M,col1))
            elif APS_type == 2:
                POINTS = np.copy(dem1.reshape(l_M,col1))
                POINTS = np.hstack((POINTS,(dem1*dem1_sum).reshape(l_M,col1)))
        
            if APS_type > 0:
                plane1,APS,par1 = inverse_mat(POINTS,data_r_temp,mask1_temp,POINTS_ori,plane_type,APS_type,solution)
                APS = APS.reshape(l_M,w_M)
                TT2 = APS[1:,:] - APS[0:-1,:]
                data_r_temp = np.copy(data_r[ran1.size:])
                POINTS = (azi2*azi2_sum).reshape(line1*w_M,1)
                APS_type = 0
                plane2,APS2,par2 = inverse_mat(POINTS,data_r_temp,mask1_temp,POINTS_ori,plane_type,APS_type,solution)
                par = np.array[[par1[0]],[par2],[par1[1]]]
            else:
                data_r_temp = np.copy(data_r[:ran1.size])
                if method == 0:
                    par1 = np.mean(data_r_temp[np.invert(np.isnan(data_r_temp))]/step)
                else:
                    par1 = np.median(data_r_temp[np.invert(np.isnan(data_r_temp))]/step)
                plane1 = POINTS_ori[:,0] * par1            
                APS = np.array([])
                data_r_temp = np.copy(data_r[ran1.size:])
                if method == 0:
                    par2 = np.mean(data_r_temp[np.invert(np.isnan(data_r_temp))]/step)
                else:
                    par2 = np.median(data_r_temp[np.invert(np.isnan(data_r_temp))]/step)
                plane2 = POINTS_ori[:,0] * par2
                par = np.array[[par1],[par2]]
  
            plane = (plane1 + plane2).reshape(l_M,w_M)
 
        elif plane_type == 0:
            data_r_temp = np.copy(data_r)
            mask1_temp = np.copy(mask1)
            if APS_type == 1:
                POINTS = np.vstack((dem1.reshape(l_M*col1,1),dem2.reshape(line1*w_M,1)))
            elif APS_type == 2:
                POINTS = np.vstack((dem1.reshape(l_M*col1,1),dem2.reshape(line1*w_M,1)))
                POINTS = np.hstack((POINTS,np.vstack(((dem1*dem1_sum).reshape(l_M*col1,1),(dem2*dem2_sum).reshape(line1*w_M,1)))))
            plane,APS,par = inverse_mat(POINTS,data_r_temp,mask1_temp,POINTS_ori,plane_type,APS_type,solution)


    ##### finalize the results
    TT = np.zeros((l_M,w_M))
    if plane.size:
        plane = plane.reshape(l_M,w_M)
        TT = TT + plane
    if APS.size:
        APS = APS.reshape(l_M,w_M)
        TT = TT + APS
    ramp_file = "temp_ramp"+N_looks+".unw"
    if os.path.isfile(ramp_file):
        [amplitude_ramp, phase_ramp, rscContent_ramp,l_ramp,w_ramp] = ut.read_bifloat(ramp_file)
        phase_ramp = phase_ramp + TT[:l_int_temp,:]
        temp_array = np.zeros((l_ramp*2,w_ramp),dtype=np.float32)
        temp_array[::2] = amplitude_ramp
        temp_array[1::2] = phase_ramp
        temp_array.astype('float32').tofile(ramp_file)
        temp_array = None 
    else:
        amplitude_ramp = amplitude_int
        phase_ramp = np.copy(TT[:l_int_temp,:])
        temp_array = np.zeros((l_int_temp*2,w_int),dtype=np.float32)
        temp_array[::2] = amplitude_ramp
        temp_array[1::2] = phase_ramp
        temp_array.astype('float32').tofile(ramp_file)
        temp_array = None
 
    TT = np.fmod(TT,2*np.pi)
    TT[TT<(-1*np.pi)] = TT[TT<(-1*np.pi)] + np.pi*2
    TT[TT>np.pi] = TT[TT>np.pi] - np.pi*2
    phase_temp[np.where(phase_temp==0)] = np.nan
    phase = np.fmod(phase_temp-TT,2*np.pi)
    phase[phase<(-1*np.pi)] = phase[phase<(-1*np.pi)] + np.pi*2
    phase[phase>np.pi] = phase[phase>np.pi] - np.pi*2
    phase = phase[:l_int_temp,:]
    phase[np.isnan(phase)] = 0
    #im = plt.imshow(phase,cmap=plt.cm.jet, vmin=-3.14, vmax=3.14)
    #plt.show()
    if l_M>l_int_temp:
        phase_cor = phase_cor[:l_int_temp,:]
        temp_array = np.zeros((l_int_temp*2,w_int))
        temp_array[::2,:] = amplitude_int
        print "Size of phase_cor is: ", phase_cor.shape, ", Size of temp_array is: ", temp_array.shape, "\n"
        temp_array[1::2,:]   = np.copy(phase_cor[:l_int_temp,:]) 
        rscContent_cor["FILE_LENGTH"] = str(l_int_temp)
        corfile_rsc = corfile + ".rsc"
        os.system("rm -f "+corfile_rsc)
        IN = open(corfile_rsc, 'w')   
        for k,v in rscContent_cor.iteritems():
            IN.write(k.ljust(41)+v+"\n")
        IN.close()
        os.system("rm -f "+corfile)
        temp_array.astype('float32').tofile(corfile)

    elif l_M<l_int_temp:
        temp = np.zeros((l_int_temp-l_M,w_int),np.float32)
        phase_cor = np.vstack((phase_cor,temp))
        temp_array = np.zeros((l_int_temp*2,w_int))
        temp_array[::2,:] = amplitude_int
        print "Size of phase_cor is: ", phase_cor.shape, ", Size of temp_array is: ", temp_array.shape, "\n"
        temp_array[1::2,:]   = np.copy(phase_cor[:l_int_temp,:])
        rscContent_cor["FILE_LENGTH"] = str(l_int_temp)
        corfile_rsc = corfile + ".rsc"
        os.system("rm -f "+corfile_rsc)
        IN = open(corfile_rsc, 'w')
        for k,v in rscContent_cor.iteritems():
            IN.write(k.ljust(41)+v+"\n")
        IN.close()
        os.system("rm -f "+corfile)
        temp_array.astype('float32').tofile(corfile)


    temp_array = amplitude_int * np.exp(1j*phase)
    print "newName is: ", newName, "\n" 
    temp_array.astype('complex64').tofile(newName)
    newName_rsc = newName + ".rsc"
    intfile_rsc = intfile + ".rsc"
    ramp_file_rsc = ramp_file + ".rsc"
    os.system("cp -f "+intfile_rsc+" "+newName_rsc)
    os.system("cp -f "+intfile_rsc+" "+ramp_file_rsc)
    temp_dir = os.getcwd()
    print "Current dir is: ", temp_dir,"\n"
    temp_str = "rm -f temp.int*"
    print temp_str,"\n"
    os.system(temp_str)

def inverse_mat(POINTS_local,data_r_local,mask1_local,P_ori,p_type,a_type,solu):
    
    data_r1_local = np.copy(data_r_local)
    POINTS1_local = np.copy(POINTS_local)
    print "Design matrix original is: ", POINTS1_local.shape,"\n"
    POINTS1_local = np.delete(POINTS1_local,np.where(np.isnan(mask1_local)),0) #POINTS1_local[np.invert(np.isnan(mask1_local)),:]
    data_r1_local = np.delete(data_r1_local,np.where(np.isnan(mask1_local)),0) #data_r1_local[np.invert(np.isnan(mask1_local)),:]

    if solu == 1:
        parLocal = np.dot(np.linalg.pinv(POINTS1_local),data_r1_local)
    else:
        mean_p = np.mean(POINTS1_local,axis=0)
        std_p = np.std(POINTS1_local,axis=0)
        data_r2_local = data_r1_local - np.mean(data_r1_local)
        print "Design matrix shape before standardization is: ", POINTS1_local.shape,"\n"
        POINTS1_s_local = np.divide((POINTS1_local - np.array([mean_p])),np.array([std_p]))
        print "size of std_p is: ", np.array([std_p]),"\n"
        parLocal = np.dot(np.linalg.pinv(POINTS1_s_local),data_r2_local)
        parLocal = np.array(parLocal)
        std_p = np.array([std_p])
        std_p = np.transpose(std_p)
        print "size of parameter is: ", parLocal.shape, " and size of std_p is: ", std_p.shape,"\n"
        parLocal = np.divide(parLocal,std_p) 
        
        print "parameters are: ", parLocal,"\n"

    #print "POINTS shape is: ", POINTS1_s_local.shape,"\n"
    #print "data shape is: ", data_r2_local.shape,"\n"
    if solu == 1:
        if p_type == 1:
            planeLocal = P_ori[:,0] * parLocal[0]
            if a_type == 1:
                APSLocal = P_ori[:,1] * parLocal[1]
            elif a_type == 2:
                APSLocal = np.dot(P_ori[:,1:3],parLocal[1:3])
            else:
                APSLocal = np.array([])
        elif p_type == 2:
            planeLocal = np.dot(P_ori[:,0],parLocal[0])
            if a_type == 1:
                APSLocal = P_ori[:,2] * parLocal[1]
            elif a_type == 2:
                APSLocal = np.dot(P_ori[:,2:4],parLocal[1:3])
            else:
                APSLocal = np.array([])
        elif p_type == 3:
            planeLocal = P_ori[:,0] * parLocal[0]
            if a_type == 1:
                APSLocal = P_ori[:,3] * parLocal[1]
            elif a_type == 2:
                APSLocal = np.dot(P_ori[:,3:5],parLocal[1:3])
            else:
                APSLocal = np.array([])
        elif p_type == 4:
            plane2Local = P_ori[:,2] * parLocal
            temp = (plane2Local.reshape(l_M,w_M)[step:,:] - plane2Local.reshape(l_M,w_M)[:(-1*step),:]).reshape(line1*w_M,1)
            data_r_local = np.fmod(data_r_local-temp,2*np.pi)
            data_r_local[data_r_local<(-1*np.pi)] = data_r_local[data_r_local<(-1*np.pi)] + np.pi*2
            data_r_local[data_r_local>np.pi] = data_r_local[data_r_local>np.pi] - np.pi*2
            data_r_local = data_r_local[l_M*col1:]
            data_r_local[np.abs(data_r_local)>th_d] = np.nan
            if method == 0:
                par_p = np.mean(data_r_local[np.invert(np.isnan(data_r_local))])
            else:
                par_p = np.median(data_r_local[np.invert(np.isnan(data_r_local))])            
            plane1Local = P_ori[:,1] * par_p
            planeLocal = plane1Local + plane2Local
            parLocal = np.array([[par_p],[parLocal]])
            APSLocal = np.array([])
        else:
            planeLocal = np.array([])
            if a_type == 1:
                APSLocal = P_ori[:,0] * parLocal[0]
            elif a_type == 2:
                APSLocal = np.dot(P_ori[:,0:2],parLocal[0:2])


    elif solu == 2:
        print "parLocal is: ", parLocal,"\n"
        if p_type<=2 and p_type>0:
            if a_type == 1 and p_type == 1:
                APSLocal = P_ori[:,1] * parLocal
            elif a_type == 2 and p_type == 1:
                APSLocal = np.dot(P_ori[:,1:3],parLocal)
            elif a_type == 1 and p_type == 2:
                APSLocal = P_ori[:,2] * parLocal
            elif a_type == 2 and p_type == 2:
                print "parLocal size is", P_ori[:,2:4].shape,"\n"
                APSLocal = np.dot(P_ori[:,2:4],parLocal)
           
            print "shape of APSLocal is: ", APSLocal.shape,"\n" 
            #APSLocal = np.transpose(APSLocal)
            print "data size is: ", data_r_local.size,"\n" 
            APSLocal1 = (APSLocal.reshape(l_M,w_M)[:,step:] - APSLocal.reshape(l_M,w_M)[:,:(-1*step)]).reshape(l_M*col1,1)
            print "Line 445 Now the useful number is: ", np.count_nonzero(~np.isnan(data_r_local)),"\n"
            print "Line 446 Now the useful number is: ", np.count_nonzero(~np.isnan(APSLocal1)),"\n"
            data_r_local = np.fmod(data_r_local[:l_M*col1]-APSLocal1,2*np.pi)
            print "Line 448 Now the useful number is: ", np.count_nonzero(~np.isnan(data_r_local)),"\n"
            data_r_local[data_r_local<(-1*np.pi)] = data_r_local[data_r_local<(-1*np.pi)] + np.pi*2
            print "Line 450 Now the useful number is: ", np.count_nonzero(~np.isnan(data_r_local)),"\n"
            data_r_local[data_r_local>np.pi] = data_r_local[data_r_local>np.pi] - np.pi*2    
            print "Line 452 Now the useful number is: ", np.count_nonzero(~np.isnan(data_r_local)),"\n"
            data_r_local[np.abs(data_r_local)>th_d] = np.nan
            temp = np.copy(data_r_local[~np.isnan(data_r_local)])
            #im = plt.imshow(APSLocal.reshape(l_M,w_M),cmap=plt.cm.jet, vmin=-3.14, vmax=3.14)
            #plt.show()
            #plt.plot(temp[0:1000])
            #plt.show()
            print "Line Now the useful number is: ", np.count_nonzero(~np.isnan(data_r_local)),"\n"
            if method == 0:
                par_p = data_r_local[np.invert(np.isnan(data_r_local))].mean()
            else:
                print "Now the useful number is: ", np.count_nonzero(~np.isnan(data_r_local)),"\n"
                par_p = np.median(data_r_local[~np.isnan(data_r_local)])

            print "par_p is: ",par_p,"\n"
            planeLocal = np.array([P_ori[:,0] * par_p])
            parLocal = np.array([[par_p],parLocal])
            print "result are: ", planeLocal.shape,",",APSLocal.shape,",",parLocal,"\n"

        elif p_type == 3:
            if a_type == 1:
                APSLocal = (P_ori[:,3] * parLocal).reshape(l_M,w_M)
                APSLocal1 = (APSLocal[:,step:] - APSLocal[:,:(-1*step)]).reshape(l_M*col1,1)
                data_r_local = np.fmod(data_r_local-APSLocal1,2*np.pi)
                data_r_local[data_r_local<(-1*np.pi)] = data_r_local[data_r_local<(-1*np.pi)] + np.pi*2
                data_r_local[data_r_local>np.pi] = data_r_local[data_r_local>np.pi] - np.pi*2
                data_r_local = data_r_local[l_M*col1:]
                data_r_local[np.abs(data_r_local)>th_d] = np.nan
                if method == 0:
                    par_p = np.mean(data_r_local[~np.isnan(data_r_local)]/step)
                else:
                    par_p = np.median(data_r_local[~np.isnan(data_r_local)]/step)
                planeLocal = P_ori[:,0] * par_p
                parLocal = np.array([[par_p],[parLocal]])
            elif a_type == 2:
                APSLocal = (np.dot(P_ori[:,3:5],parLocal)).reshape(l_M,w_M)
                APSLocal1 = (APSLocal[:,step:] - APSLocal[:,0:(-1*step)]).reshape(l_M*col1,1)
                data_r_local = np.fmod(data_r_local-APSLocal1,2*np.pi)
                data_r_local[data_r_local<(-1*np.pi)] = data_r_local[data_r_local<(-1*np.pi)] + np.pi*2
                data_r_local[data_r_local>np.pi] = data_r_local[data_r_local>np.pi] - np.pi*2
                data_r_local = data_r_local[l_M*col1:]
                data_r_local[np.abs(data_r_local)>th_d] = np.nan
                if method == 0:
                    par_p = np.mean(data_r_local[~np.isnan(data_r_local)]/step)
                else:
                    par_p = np.median(data_r_local[~np.isnan(data_r_local)]/step)
                planeLocal = P_ori[:,0] * par_p
                parLocal = np.array([[par_p],[parLocal]])
            elif a_type == 0:
                plane1Local = P_ori[:,2] * parLocal
                temp = plane1Local.reshape(l_M,w_M)
                planeLocal_temp = (temp[step:,:] - temp[:(-1*step),:]).reshape(line1,w_M)
                data_r_local = np.fmod(data_r_local-planeLocal_temp,2*np.pi)
                data_r_local[data_r_local<(-1*np.pi)] = data_r_local[data_r_local<(-1*np.pi)] + np.pi*2
                data_r_local[data_r_local>np.pi] = data_r_local[data_r_local>np.pi] - np.pi*2
                data_r_local = data_r_local[l_M*col1:]
                data_r_local[np.abs(data_r_local)>th_d] = np.nan
                if method == 0:
                    par_p = np.mean(data_r_local[~np.isnan(data_r_local)]/step)
                else: 
                    par_p = np.median(data_r.local[~np.isnan(data_r_local)]/step)
                plane2Local = P_ori[:,1] * par_p
                planeLocal = plane1Local + plane2Local
                APSLocal = np.array([])
                parLocal = np.array([[par_p],[parLocal]])
                 
        elif p_type == 0:
            if a_type == 1:
                POINTS_local[np.abs(POINTS_local)<0.0000000001] = np.nan
                temp = data_r_local / POINTS_local
                #temp[np.abs(temp)>0.01] = np.nan
                parLocal = np.mean(temp[~np.isnan(temp)])
                print "parLocal is: ", parLocal,"\n"
                parLocal = np.array(parLocal)
                APSLocal = P_ori[:,0] * parLocal
            elif a_type == 2:
                #APS_temp = (rate2 * P_ori[:,0]).reshape(l_M,w_M)
                #APS1 = (APS_temp[:,step:] - APS_temp[:,:(-1*step)]).reshape(l_M*col1,1)
                #print "size of APS1, data_r_local, POINTS_local: ", APS1.size," ",data_r_local.size," ",POINTS_local.shape, "\n"
                #APS2 = (APS_temp[step:,:] - APS_temp[:(-1*step),:]).reshape(line1*w_M,1)
                #APS1 = np.vstack((APS1,APS2))
                #APS2 = None
                #APS_temp = None
                #print "shape of APS1, data_r_local: ", APS1.shape," ",data_r_local.shape,"\n"
                #temp = data_r_local - APS1
                #data_r_local = None
                #APS1 = None
                #temp1 = np.copy(POINTS_local[:,1]).reshape(temp.size,1)
                #POINTS_local = None
                #print "size of temp and temp1: ", temp.shape," ", temp1.shape,"\n"
                #temp[np.isnan(temp1)] = np.nan
                #temp1[np.isnan(temp)] = np.nan
                #temp = np.divide(temp[~np.isnan(temp)],temp1[~np.isnan(temp1)])
                #temp[np.isinf(temp)] = np.nan
                #temp[abs(temp)>0.1] = np.nan               
                #parLocal = np.array([[np.mean(temp[~np.isnan(temp)])],[rate2]])
                APSLocal = np.dot(P_ori[:,0:2],parLocal)
            planeLocal = np.array([]) 
        return planeLocal,APSLocal,parLocal



if __name__ == "__main__":
    sys.exit(main())

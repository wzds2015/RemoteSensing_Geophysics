#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Sun Aug 17 18:14:13 2014

code used for layered elastic model of surface loading

Wenliang Zhao
'''

import os,sys,re,glob,subprocess,math,time
import Rsmas_Utilib as ut
import ElasticSAR as es
import numpy as np
from osgeo import gdal
import time

para_list = ut.readProc(sys.argv[1])
processdir = para_list.get("processdir").strip()
disp_dir = processdir + "/InSAR"
line_insar = int(para_list.get("line_insar").strip())
col_insar = int(para_list.get("col_insar").strip())
left_lon_insar = float(para_list.get("left_lon_insar").strip())
top_lat_insar = float(para_list.get("top_lat_insar").strip())
res_lon_insar = float(para_list.get("res_lon_insar").strip())
res_lat_insar = float(para_list.get("res_lat_insar").strip())
lookAng_insar = float(para_list.get("lookAng").strip())
azimuth_insar = float(para_list.get("azimuth").strip())
utmZone = para_list.get("utmZone").strip()
res_x_utm = int(para_list.get("spacing").strip())
res_y_utm = res_x_utm
box = para_list.get("box")
method = para_list.get("method").strip()
density = para_list.get("density").strip()
example_model = para_list.get("example_model")
do_velo = int(para_list.get("velocity").strip())
velo_file = disp_dir + "/velocity.unw"

try:
    velo_sample = para_list.get("velo_sample").strip()
    velo_sample_max = int(para_list.get("velo_sample_max").strip())
    velo_sample_th = float(para_list.get("velo_sample_th").strip())
except:
    print "Sampling method not given!\n"
    print "Using default full resolution grid!\n"

### other parameters
load_line = int(para_list.get("load_line").strip())
load_col = int(para_list.get("load_col").strip())
center_x_utm = int(para_list.get("center_x_utm").strip())
center_y_utm = int(para_list.get("center_y_utm").strip())
area_dim = int(para_list.get("area_dim").strip())
area_meter = float(para_list.get("area_meter").strip())
center_x_utm = float(para_list.get("center_x_utm").strip())
center_y_utm = float(para_list.get("center_y_utm").strip())
xbase = int(para_list.get("xbase").strip())
layer2_depth = int(para_list.get("layer2_depth").strip())
layer3_depth = int(para_list.get("layer3_depth").strip())
layer4_depth = int(para_list.get("layer4_depth").strip())
layer2_vp = int(para_list.get("layer2_vp").strip())
layer3_vp = int(para_list.get("layer3_vp").strip())
layer4_vp = int(para_list.get("layer4_vp").strip())
layer2_vs = int(para_list.get("layer2_vs").strip())
layer3_vs = int(para_list.get("layer3_vs").strip())
layer4_vs = int(para_list.get("layer4_vs").strip())
layer2_density = int(para_list.get("layer2_density").strip())
layer3_density = int(para_list.get("layer3_density").strip())
layer4_density = int(para_list.get("layer4_density").strip())


### read insar deformation file
disp_list = glob.glob(disp_dir+"/geo_def_*.unw")
disp_list.sort()
disp_list.sort()
date_list = es.cal_dateList(disp_list)
print date_list
if do_velo==1:
    velocity_file = disp_dir+"/velocity.unw"
    disp_list.append(velocity_file)
#disp_list_short = disp_list
#for ni in range(len(disp_list_short)):
#    disp_list_short[ni] = os.path.basename(disp_list_short[ni]).strip()
    
os.chdir(disp_dir)    ### until line 92
for ni in range(len(disp_list)):
    f = disp_list[ni]
    f1 = f[:-4]
    temp_str = "cut_insar " + f + " " + str(line_insar) + " " + str(col_insar) + " " + box[0] + "/" + box[1] + "/" + box[2] + "/" + box[3] + "\n"
    es.cut_insar(f,line_insar,col_insar,box)
    if box: 
        print "Yes there is a box!\n"
        #exit(0)
        line_insar1 = int(box[1]) - int(box[0]) + 1
        col_insar1 = int(box[3]) - int(box[2]) + 1
        #temp1 = 90.180580555555555555
        #temp2 = 29.6320666666666667
        left_lon_insar1 = left_lon_insar + res_lon_insar * (int(box[2]) - 1)
        #left_lon_insar1 = left_lon_insar1 - (temp1 - left_lon_insar1)
        top_lat_insar1 = top_lat_insar - res_lat_insar * (int(box[0]) - 1)    
        print "corner is: ", top_lat_insar1,", ", left_lon_insar1,"\n"
    else:
        line_insar1 = line_insar
        col_insar1 = col_insar
        left_lon_insar1 = left_lon_insar
        top_lat_insar1 = top_lat_insar
        
    temp_str = "load_elastic_layer.py insar2geotiff_utm " + str(line_insar1) + " " + str(col_insar1) + " " \
               + str(left_lon_insar1) + " " + str(top_lat_insar1) + " " + str(res_lon_insar) + " " \
               + str(res_lat_insar) + " " + f1 + " " + utmZone + " " \
               + str(res_x_utm) + " " + str(res_y_utm) 
        
    '''
    temp_list = ["insar2geotiff_utm.py",str(line_insar),str(col_insar),str(left_lon_insar), \
                     str(top_lat_insar),str(res_lon_insar),str(res_lat_insar),f,str(utmZone), \
                     str(res_x_utm),str(res_y_utm),box[0],box[1],box[2],box[3]]
    '''
        
    print temp_str, "\n"
    es.insar2geotiff_utm(line_insar1,col_insar1,left_lon_insar1,top_lat_insar1,res_lon_insar,res_lat_insar,f1,utmZone,res_x_utm,res_y_utm)
        
    #temp_str = "rm -f " + f
    #temp_list = ["rm","-f",f]
    #print temp_str, "\n"
    new_file = f1 + "_UTM.tif"
    disp_list[ni] = new_file
        
    ### Read geotiff on UTM to get upper left coordinates    
    if ni == 0:
        datafile = gdal.Open(new_file)
        info = datafile.GetGeoTransform() 
        left_x_utm_insar = info[0]
        top_y_utm_insar = info[3]
        col_insar_utm = datafile.RasterXSize
        line_insar_utm = datafile.RasterYSize

line_insar = line_insar1
col_insar = col_insar1
left_lon_insar = left_lon_insar1
top_lat_insar = top_lat_insar1   

if do_velo:
    velosity_file_UTM = disp_list[-1]
    del disp_list[-1]
velo_file = velo_file[:-4] + "_UTM.tif"

            
model_dir = processdir + "/Model"
source_code = processdir + "/load_elastic_layer.f"
load_file = processdir + "/charge.bin"
load_history = processdir + "/load_history"
young_min = int(para_list.get("young_min").strip())
young_max = int(para_list.get("young_max").strip())
young_unit = para_list.get("young_unit").strip()
young_step = int(para_list.get("young_step").strip())
if young_unit == 'MPa':
    young_min = young_min * math.pow(10,6)
    young_max = young_max * math.pow(10,6)
    young_step = young_step * math.pow(10,6)
else:
    young_min = young_min * math.pow(10,9)
    young_max = young_max * math.pow(10,9)
    young_step = young_step * math.pow(10,9)
    
poisson_min = float(para_list.get("possion_min").strip())
poisson_max = float(para_list.get("possion_max").strip())
poisson_step = float(para_list.get("possion_step").strip())
layer1_density = int(para_list.get("layer1_density").strip())
layer1_depth = int(para_list.get("layer1_depth").strip())

'''
### other parameters
load_line = int(para_list.get("load_line").strip())
load_col = int(para_list.get("load_col").strip())
center_x_utm = int(para_list.get("center_x_utm").strip())
center_y_utm = int(para_list.get("center_y_utm").strip())
area_dim = int(para_list.get("area_dim").strip())
area_meter = float(para_list.get("area_meter").strip())
center_x_utm = float(para_list.get("center_x_utm").strip())
center_y_utm = float(para_list.get("center_y_utm").strip())
xbase = int(para_list.get("xbase").strip())
layer2_depth = int(para_list.get("layer2_depth").strip())
layer3_depth = int(para_list.get("layer3_depth").strip())
layer4_depth = int(para_list.get("layer4_depth").strip())
layer2_vp = int(para_list.get("layer2_vp").strip())
layer3_vp = int(para_list.get("layer3_vp").strip())
layer4_vp = int(para_list.get("layer4_vp").strip())
layer2_vs = int(para_list.get("layer2_vs").strip())
layer3_vs = int(para_list.get("layer3_vs").strip())
layer4_vs = int(para_list.get("layer4_vs").strip())  
layner2_density = int(para_list.get("layer2_density").strip())
layer3_density = int(para_list.get("layer3_density").strip())
layer4_density = int(para_list.get("layer4_density").strip())
'''  ### moved to the begining for the option of half space

#### prepare loading hostory array
F = open(load_history,'r')
load_array = np.zeros((1,len(disp_list)),dtype=np.float32)
k = 0
for line in F.readlines():
    if line.strip():
        load_array[0,k] = float(line.strip())
        k += 1
F.close()
 
#print "young_step is: ", young_step,"\n"
if float(young_step) - 0 > 0.01:
    young_list = range(int(young_min),int(young_max)+int(young_step),int(young_step))
else:
    young_list = []
    young_list.append(int(young_min))

if float(poisson_step) - 0 > 0.0001:
    poisson_list = np.arange(poisson_min,poisson_max+poisson_step,poisson_step)
else:
    poisson_list = []
    poisson_list.append(float(poisson_min))   

#print "young_list is: ", young_list, "\n"
length_para = len(young_list)*len(poisson_list)    
para_list_temp = np.zeros((length_para,2),dtype=np.float32)
for ni in range(len(young_list)):
    for nj in range(len(poisson_list)):
        para_list_temp[len(poisson_list)*ni+nj][0] = young_list[ni]
        para_list_temp[len(poisson_list)*ni+nj][1] = poisson_list[nj]
        
os.chdir(processdir)

### temporal comment
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
    #temp_str = "rm -rf Model"
    #temp_list = ["rm","-rf","Model"]
    #print temp_str, "\n"
    #output = subprocess.Popen(temp_list).wait()
    #if output != 0:
    #    print "Error occurred during removing model folder!\n"
    #    exit(1)
#os.mkdir("Model")

os.chdir(model_dir)
#joblist = open("run_model",'w')
k = 0
b = 0 ### batch of run_file, used for mpi, if size for 1 batch is too big, will get congestion
#t = 0 ### count for total number
if method == 'L':    ### for layer model
    joblist = open("run_model",'w')
    for ni in range(length_para):
        ID = ni + 1
        temp_model_dir = model_dir + "/" + "model_%04d" % ID    #### now up to 9999 models
        if not os.path.isdir(temp_model_dir):
            os.mkdir(temp_model_dir)
        os.chdir(temp_model_dir)
        temp_str = "ln -s " + load_file + " " + temp_model_dir + "/charge.bin"
        os.system(temp_str)
        temp_str = "cp -f " + source_code + " " + temp_model_dir + "/"
        os.system(temp_str)
    
        temp_E = para_list_temp[ni][0]
        temp_sigma = float(para_list_temp[ni][1])
        F = open("temp_para",'w')
        temp_E_g = temp_E/np.power(10,9)
        F.write("E = %3f" % temp_E_g + "\n")
        F.write("NU=%1.2f" % temp_sigma + "\n")
        F.close()
    
        vp_modulus =  temp_E * (1.0 - temp_sigma) / (1.0 + temp_sigma) / (1 - 2.0 * temp_sigma)
        layer1_vp = math.sqrt(temp_E * (1.0 - temp_sigma) / (1.0 + temp_sigma) / (1.0 - 2.0 * temp_sigma) / layer1_density)  
        layer1_vs = layer1_vp / (math.sqrt((2.0 - 2.0 * temp_sigma) / (1.0 - 2.0 * temp_sigma)))
    
        F = open("elasticity",'w')
        F.write("Layer1_thickness = " + str(layer1_depth) + "\n")
        F.write("Layer1 VP = " + str(layer1_vp) + "\n")
        F.write("Layer1 VS = " + str(layer1_vs) + "\n")
        F.write("Layer1 density = " + str(layer1_density) + "\n")
        F.close()
    
        IN = open(source_code, 'r')
        code_list = IN.readlines()
        IN.close()
    
        OUT = open("model.f", 'w')
        code_list[36] = "      parameter          (nz=1000,nfx=" + str(load_col) + ",nfy=" + str(load_line) + ",ndiff=8)\n"
        code_list[42] = "      PARAMETER          (nx=" + str(area_dim) + ",nvarc=2)\n"
        code_list[44] = "      PARAMETER          (clarg=" + str(area_meter) + ")\n"
        code_list[90] = "      x0c=" + str(center_x_utm) + "\n"
        code_list[92] = "      x0c=" + str(center_y_utm) + "\n"
        code_list[112] = "      xbase=" + str(xbase) + ".\n"
        code_list[567] = "      epcsed=" + str(layer1_depth) + ".\n"
        code_list[569] = "      epcsup=" + str(layer2_depth) + ".\n"
        code_list[571] = "      epcint=" + str(layer3_depth) + ".\n"
        code_list[573] = "      epcinf=" + str(layer4_depth) + ".\n"
        code_list[575] = "      vpsed=" + str(layer1_vp) + "\n"
        code_list[577] = "      vssed=" + str(layer1_vs) + "\n"
        code_list[579] = "      vpsup=" + str(layer2_vp) + ".\n"
        code_list[581] = "      vssup=" + str(layer2_vs) + ".\n"
        code_list[583] = "      vpint=" + str(layer3_vp) + ".\n"
        code_list[585] = "      vsint=" + str(layer3_vs) + ".\n"
        code_list[587] = "      vpinf=" + str(layer4_vp) + ".\n"
        code_list[589] = "      vsinf=" + str(layer4_vs) + ".\n"
        code_list[591] = "      rhosed=" + str(layer1_density) + ".\n"
        code_list[593] = "      rhosup=" + str(layer2_density) + ".\n"
        code_list[595] = "      rhoint=" + str(layer3_density) + ".\n"
        code_list[597] = "      rhoinf=" + str(layer4_density) + ".\n"
    
        for line in code_list:
            OUT.write(line)
        OUT.close()
    
        for f in disp_list:
            bName_disp_temp = os.path.basename(f).strip()
            temp_str = "ln -s " + f + " " + temp_model_dir + "/" + bName_disp_temp
            os.system(temp_str)
    
        temp_str = "ifort -assume byterecl -O3 -parallel -axP -ip -132 model.f -o elastic_model"
        temp_list = ["ifort","-assume","byterecl","-O3","-parallel","-axP","-ip","-132","model.f","-o","elastic_model"]
        print temp_str,"\n"
        #output = subprocess.Popen(temp_list).wait()
        #if output != 0:
        #    print "Error occurs during compilation!\n"
        #    exit(1)
    
        if k < length_para:
            k += 1
            joblist.write("run_load_model.py "+temp_model_dir+"\n")
        else:
            joblist.write("run_load_model.py "+temp_model_dir)

    joblist.close()
    print "Model_dir is: ", model_dir, "\n"
    os.chdir(model_dir)        
    temp_str = "createBatch_8.pl --workdir " + model_dir + " --infile " + model_dir+"/run_model walltime=15:00"
    temp_list = ["createBatch_8.pl","--workdir",model_dir,"--infile",model_dir+"/run_model","walltime=15:00"]
    print temp_str,"\n"
    #output = subprocess.Popen(temp_list).wait()
    #if output != 0:
    #    print "Error occurs during modeling!\n"
    #    exit(1)

elif method == "H":   ### for half space model
    temp_str = "rm -f job*.sh"
    output = os.system(temp_str)
    #if output != 0:
    #    print "Removing files failed!\n"
    #    exit(1)
    es.prepCharge(load_file,load_col,area_dim)

    for ni in range(length_para):
        ID = ni + 1
        temp_model_dir = model_dir + "/" + "model_%04d" % ID    #### now up to 9999 models
        if not os.path.isdir(temp_model_dir):
            os.mkdir(temp_model_dir)
        os.chdir(temp_model_dir)
    
        for f in disp_list:
            bName_disp_temp = os.path.basename(f).strip()
            temp_str = "ln -s " + f + " " + temp_model_dir + "/" + bName_disp_temp
            os.system(temp_str)

        temp_E = para_list_temp[ni][0]
        temp_sigma = float(para_list_temp[ni][1])
        F = open("temp_para",'w')
        temp_E_g = round(temp_E/np.power(10,9))
        F.write("ice_model="+processdir+"/charge_big.bin\n")
        F.write("ice_model_col="+str(area_dim)+"\n")
        F.write("grid_size="+str(res_x_utm)+"\n")
        F.write("r="+str(density)+"\n")
        F.write("v="+str(temp_sigma)+"\n")
        F.write("E="+str(temp_E_g)+"\n")
        F.write("E_unit="+young_unit)
        F.close()
    
        WTime = "50:00"
        jobfile = open(model_dir+"/job_%04d.sh"%ID,'w')
        jobfile.write("#! /bin/env csh\n")
        jobfile.write("#BSUB -J model_%04d"%ID+"\n")
        jobfile.write("#BSUB -o z_output_model_%04d.o"%ID+"\n") 
        jobfile.write("#BSUB -e z_output_model_%04d.e"%ID+"\n")
        jobfile.write("#BSUB -n 18\n")
        jobfile.write("#BSUB -q general\n") 
        jobfile.write("#BSUB -W "+WTime+"\n")
        jobfile.write('#BSUB -R "span[ptile=1]"\n')
        jobfile.write("#BSUB -B\n")
        jobfile.write("#BSUB -N\n")
        jobfile.write("cd "+temp_model_dir+"\n")
        jobfile.write("mpiexec -n 18 Elastic_half_space_load.py temp_para")
        jobfile.close()
        
        '''
        if ID % max_N != 0 and ID < length_para:
            joblist.write("run_elastic_mpi.py "+temp_model_dir+"\n")
        else:
            joblist.write("run_elastic_mpi.py "+temp_model_dir)
            joblist.close()
            if ID < length_para:
                b += 1
                joblist = open(model_dir+"/run_model"+str(b),'w')
        '''

    #max_N = 30    ### adjustable depending on the limit of number of processor
    h,m = WTime.split(":")
    WTime_m = int(h.strip()) * 60 + int(m.strip())
    print "Model_dir is: ", model_dir, "\n"
    os.chdir(model_dir)
    count = 1
    check = 1
    temp_list = ["rm","-f","z_output*.*"]
    output = subprocess.Popen(temp_list).wait()
    if output != 0:
        print "Error occurs during removing z_output files!\n"
        exit(1)

'''
    run_list = glob.glob(model_dir+"/job*.sh")
    for ni in range(len(run_list)):
        temp_str = "bsub < " + run_list[ni]
        os.system(temp_str)

    time.sleep(60)
    while (check==1):
        if count > WTime_m:
            print "Time exeeds maximum!\n"
            exit(1)
        else:
            finish_list = glob.glob(model_dir+"/z_output*.o")
            print "Current number of jobs finished in <" + model_dir + ">: <" + str(len(finish_list)) + "> out of <" + str(len(run_list)) + "> after <" + str(count) + "> minutes\n"
            if len(finish_list) == len(run_list):
                time.sleep(60)
                check == -1
                print "All jobs finished!\n"
            else:
                time.sleep(60)
                count += 1
'''

'''
    t = 0
    run_list = glob.glob(model_dir+"/job*.sh")
    temp_list = ["rm","-f","z_output*.*"]
    while (check):
        output = subprocess.Popen(temp_list).wait()
        if output != 0:
            print "Removing file failed!\n"
        if len(run_list) - count > max_N:
            q_end = count+max_N
            q_N = max_N
        else:
            q_end = len(run_list)
            q_N = len(run_list) - count
            check = 0

        for ni in range(count,q_end):
            temp_str = "bsub < " + run_list[ni]
            os.system(temp_str)
        time.sleep(60)
        finish_list = []
        while (len(finish_list) < q_N):
            if t > WTime_m:
                print "Time exeeds maximum!\n"
                exit(1)
            print "Current number of jobs finished in <" + model_dir + ">: <" + str(len(finish_list)) + "> out of <" + str(q_N) + "> after <" + str(t) + "> minutes\n"     
            time.sleep(60)
            finish_list = glob.glob(model_dir+"/z_output*.o")
            t += 1
        count = q_N
'''
    
os.chdir(model_dir)
dir_list,temp_N = ut.getSubdir(model_dir)
dir_list = sorted(dir_list)
print "subDirs are: ", dir_list, "\n"
model_residual = np.zeros((len(dir_list),1))

if do_velo:
    velo_residual = np.zeros((len(dir_list),1))
    if velo_sample == 'QT':
        dataset = gdal.Open(velo_file)
        temp_data = dataset.GetRasterBand(1).ReadAsArray().flatten()
        temp_data[temp_data==0] = np.nan
        temp_data[temp_data>10000] = np.nan
        w_data = np.fromfile('/scratch/wlzhao/NSBAS/YAMZHO/Model/example_model_half',dtype=np.float32)
        temp_data[np.isnan(w_data)] = np.nan
        temp_data = temp_data.reshape(line_insar_utm,col_insar_utm)
        w_data = w_data.reshape(line_insar_utm,col_insar_utm)
        #level_temp,center_x_temp,center_y_temp,data_big = es.quadtree(temp_data,velo_sample_th,velo_sample_max)
        level_temp,center_x_temp,center_y_temp,data_big = es.quadtree(w_data,0.1,velo_sample_max)

for ni in range(len(dir_list)):
    d = dir_list[ni]
    os.chdir(d)
    def_x = d + "/carto_ux.r4"
    def_y = d + "/carto_uy.r4"
    def_z = d + "/carto_uz.r4"
    def_LOS_file = "carto_LOS.r4"
    if os.path.isfile(def_x):
        es.corOut(def_x,area_dim)
        es.corOut(def_y,area_dim)
        es.corOut(def_z,area_dim)
        es.proj2insar(def_z,def_x,def_y,float(lookAng_insar),float(azimuth_insar))
        es.clip_model(def_LOS_file,area_dim,center_x_utm,center_y_utm,res_x_utm,left_x_utm_insar,top_y_utm_insar,line_insar_utm,col_insar_utm)
        if example_model:
            R = es.cal_resi(disp_list,load_array,def_LOS_file,example_model)
        else:
            R = es.cal_resi(disp_list,load_array,def_LOS_file)
        model_residual[ni][0] = R

        if do_velo and velo_sample == 'QT':
            R_v = es.cal_resi_velo_QT1(def_LOS_file,load_array,velo_file,date_list,col_insar_utm,center_y_temp,center_x_temp)
            velo_residual[ni,0] = R_v
        else:
            R_v = es.cal_resi_velo(def_LOS_file,load_array,velo_file,date_list)
            velo_residual[ni,0] = R_v
        
        if method == "L":
            F = open("temp_para",'r')
            temp_list = F.readlines()
            F.close()
            temp_E = temp_list[0].strip()
            temp_sigma = temp_list[1].strip()
        elif method == "H":
            temp_hs_para_list = ut.readProc("temp_para")
            temp_E = temp_hs_para_list.get("E").strip()
            temp_sigma = temp_hs_para_list.get("v").strip()
        else:
            print "Now only support layer (L) and half-space (H) models!\n"
            exit(1)
      
        F = open("report",'w')
        F.write("Elastic layer surface loading model summary:\n")
        F.write("Data_NUMBER                " + str(len(disp_list)-1) + "\n")
        F.write("DATA_LENGTH                " + str(line_insar) + "\n")
        F.write("DATA_WIDTH                 " + str(col_insar) + "\n")
        F.write("LEFT_X                     " + str(left_x_utm_insar) + "\n")
        F.write("TOP_Y                      " + str(top_y_utm_insar) + "\n")
        F.write("LEFT_LON                   " + str(left_lon_insar) + "\n")
        F.write("TOP_LAT                    " + str(top_lat_insar) + "\n")
        F.write("LAYER1_THICKNESS           " + str(layer1_depth) + "\n")
        F.write("LAYER1_DENSITY             " + str(layer1_density) + "\n")
        F.write("LAYER1_YOUNG_MODULUS       " + temp_E + "\n")
        F.write("LAYER1_POISSON_RATIO       " + temp_sigma + "\n")
        if do_velo:
            F.write("RMSE                       " + str(R) + "\n")
            F.write("RMSE                       " + str(R_v))
        else:
            F.write("RMSE                       " + str(R))
        F.close()

    else:
        print "Modeling error occurs in ", d, "\n"
        model_residual[ni][0] = np.nan
        if do_velo:
            velo_residual[ni,0] = np.nan

### find the smallest residual model and making plots
para_list = np.hstack((para_list_temp,model_residual))

#min_resi = np.min(model_residual)
#min_ind = np.where(model_residual==min_resi)
#min_ind1 = min_ind[0]
min_ind = np.nanargmin(model_residual)

if do_velo:
    para_list_velo = np.hstack((para_list_temp,velo_residual))
    min_ind_velo = np.nanargmin(velo_residual)

os.chdir(dir_list[min_ind])
resi_file = processdir + "/para_residual"
resi_file_1 = dir_list[min_ind] + "/para_residual" 
F = open(resi_file_1,'w')
F.write("The best model is: " + dir_list[min_ind] + "\n")
k = 1
for ni in range(len(dir_list)):
    bName = os.path.basename(dir_list[ni])
    if k != len(dir_list):
        F.write("%03d    %1.2f    %2.5f    %s\n" % (round(para_list[ni][0]/np.power(10,9)),para_list[ni][1],para_list[ni][2],dir_list[ni]))
    else:
        F.write("%03d    %1.2f    %2.5f    %s" % (round(para_list[ni][0]/np.power(10,9)),para_list[ni][1],para_list[ni][2],dir_list[ni]))
F.close()
temp_str = "ln -s " + resi_file_1 + " " + resi_file
os.system(temp_str)

model_file = "carto_LOS.r4"
temp_model = np.fromfile(model_file,dtype=np.float32)
for ni in range(len(disp_list)):
    dataset = gdal.Open(disp_list[ni])
    temp_data = dataset.GetRasterBand(1).ReadAsArray().flatten()
    temp_data[temp_data==0] = np.nan    ### 0 is created from transform, used to fill the gap between coordinates
    temp_data[temp_data>10000] = np.nan      ### this is the meaningful 0
    #temp_data = np.fromfile(disp_list[ni],dtype=np.float32)
    mask = np.fromfile('/scratch/wlzhao/NSBAS/YAMZHO/Model/example_model1',dtype=np.float32)
    temp_data[np.isnan(mask)] = np.nan
    bName = os.path.basename(disp_list[ni])
    temp_date = re.findall('\d\d\d\d\d\d\d\d',bName)[0].strip()
    
    print "load array is: ", load_array, ", size is: ", load_array.shape,"\n"
    temp_model_date = temp_model * load_array[0,ni]
    temp_model_date[np.isnan(temp_data)] = np.nan
    temp_model_date_name = "model_" + temp_date
    temp_model_date.astype('float32').tofile(temp_model_date_name)
    
    temp_model_date_name_rsc = "model_" + temp_date + ".rsc"
    F = open(temp_model_date_name_rsc,'w')
    F.write("FILE_LENGTH        " + str(line_insar_utm) + "\n")
    F.write("WIDTH              " + str(col_insar_utm) + "\n")
    F.close()
    
    print "final dimensions are: data - ", temp_data.shape, ", model - ", temp_model_date.shape, "\n"
    temp_resi = temp_data - temp_model_date
    temp_resi_name = "residual_" + temp_date
    temp_resi.astype('float32').tofile(temp_resi_name)
    
    temp_resi_rsc = "residual_" + temp_date + ".rsc"
    F = open(temp_resi_rsc,'w')
    F.write("FILE_LENGTH        " + str(line_insar_utm) + "\n")
    F.write("WIDTH              " + str(col_insar_utm) + "\n")
    F.close()
    


if do_velo:
    os.chdir(dir_list[min_ind_velo])
    temp_dir = os.getcwd()
    print "Now processing in " + temp_dir + "\n"
    resi_file = processdir + "/para_residual_velo"
    resi_file_1 = dir_list[min_ind_velo] + "/para_residual_velo"
    F = open(resi_file_1,'w')
    F.write("The best model is: " + dir_list[min_ind_velo] + "\n")
    k = 1
    for ni in range(len(dir_list)):
        bName = os.path.basename(dir_list[ni])
        if k != len(dir_list):
            F.write("%03d    %1.2f    %2.5f    %s\n" % (round(para_list_velo[ni][0]/np.power(10,9)),para_list_velo[ni][1],para_list_velo[ni][2],dir_list[ni]))
        else:
            F.write("%03d    %1.2f    %2.5f    %s" % (round(para_list_velo[ni][0]/np.power(10,9)),para_list_velo[ni][1],para_list_velo[ni][2],dir_list[ni]))
    F.close()
    temp_str = "ln -s " + resi_file_1 + " " + resi_file
    os.system(temp_str)

    model_velo_file = dir_list[min_ind_velo] + "/model_velo"
    temp_model = np.fromfile(model_velo_file,dtype=np.float32)
    dataset = gdal.Open(velo_file)
    temp_data = dataset.GetRasterBand(1).ReadAsArray().flatten()
    temp_data[temp_data==0] = np.nan    ### 0 is created from transform, used to fill the gap between coordinates
    temp_data[temp_data>10000] = np.nan      ### this is the meaningful 0
    w_data = np.fromfile('/scratch/wlzhao/NSBAS/YAMZHO/Model/example_model1',dtype=np.float32)
    temp_data[np.isnan(w_data)] = np.nan
    #temp_data = np.fromfile(disp_list[ni],dtype=np.float32)

    temp_model[np.isnan(temp_data)] = np.nan

    model_velo_rsc = model_velo_file + ".rsc"
    F = open(model_velo_rsc,'w')
    F.write("FILE_LENGTH        " + str(line_insar_utm) + "\n")
    F.write("WIDTH              " + str(col_insar_utm) + "\n")
    F.close()

    temp_resi = temp_data - temp_model
    temp_resi_name = dir_list[min_ind_velo] + "/residual_velo"
    temp_resi.astype('float32').tofile(temp_resi_name)

    temp_resi_rsc = dir_list[min_ind_velo] + "/residual_velo.rsc"
    F = open(temp_resi_rsc,'w')
    F.write("FILE_LENGTH        " + str(line_insar_utm) + "\n")
    F.write("WIDTH              " + str(col_insar_utm) + "\n")
    F.close()
    
    
    
    

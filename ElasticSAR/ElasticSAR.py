# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 21:40:36 2014

Utilities for loading modeling

Wenliang Zhao
"""

import os,re,subprocess,math
import numpy as np
import Rsmas_Utilib as ut
from osgeo import gdal
from datetime import date
import matplotlib.pyplot as plt


def cut_insar(def_file,line,col,box):

    '''
        function used for cutting insar file based on a small box
    '''

    print "Now processing on ", def_file,"\n"
    raster = np.fromfile(def_file,np.float32).reshape(line*2,col)[1::2,:]
    if box:
        raster = raster[int(box[0])-1:int(box[1]),int(box[2])-1:int(box[3])]
    outFile = def_file[:-4]
    raster.astype('float32').tofile(outFile)
    

def insar2geotiff_utm(line,col,left_lon,top_lat,res_lon,res_lat,def_file,utmZone,res_x_utm,res_y_utm):
    
    '''
       File used to convert InSAR time-series
       binary result (float32) to geotiff file 
       used by gdal
    '''

    from osgeo import osr, gdal    
    
    print "Line is: ", line, "col is: ", col, "\n"
    tifFile = def_file + ".tif"
    raster = np.fromfile(def_file,np.float32)
    print "file size is: ", raster.size, "\n"      
    raster = raster.reshape(line,col)
    raster[raster==0] = 32768  ### for those meaningful data=0, change to large number first, after transform, change back
    #raster = raster[1::2,:]  ### used for *.unw, last line needs to set line to line*2
    

    # Create gtif
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(tifFile, col, line, 1,gdal.GDT_Float32)  ### outputName,col,line,bandNumbers,format
    #raster = np.zeros((col,line),dtype=np.float32)

    # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    print "real corner is: ", top_lat, ", ", left_lon, "\n"
    output = dst_ds.SetGeoTransform([left_lon, res_lon, 0, top_lat, 0, res_lat*(-1)])
    if output != 0:
        print "Error occurs when setting coordinates!\n"
        exit(1)

    # set the reference info 
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    dst_ds.SetProjection( srs.ExportToWkt() )

    # write the band
    dst_ds.GetRasterBand(1).WriteArray(raster)
    
    # close the data set
    dst_ds = None
    # convert to UTM
    outFile = def_file + "_UTM.tif"
    proj_str = "+proj=utm +zone=" + utmZone + " +datum=WGS84"
    res_str = str(res_x_utm) + " " + str(res_y_utm)
    temp_str = "gdalwarp -t_srs '" + proj_str + "' -tr " + res_str + " " + tifFile + " " + outFile 
    temp_list = ["gdalwarp","-t_srs",proj_str,"-tr",str(res_x_utm),str(res_y_utm),tifFile,outFile]    
    print temp_str,"\n"
    output = subprocess.Popen(temp_list).wait()
    if output != 0:
        print "Error occurs durings deg2utm conversion!\n"
        exit(1)

'''
def align_utm(def_file,center_x_utm,)

'''

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


def clip_model(model_file,dimension,center_x_utm,center_y_utm,spacing,left_x_utm_insar,top_y_utm_insar,line_insar,col_insar):
    
    '''
       function used to clip the product of elastic modeling to be the size of insar
    '''
   
    print "inputs are: model_file -- ", model_file, ", dimension -- ", dimension, ", center_x_utm -- ", center_x_utm, ", center_y_utm -- ", center_y_utm, ", spacing -- ", spacing, ", left_x_utm_insar -- ", left_x_utm_insar, ", top_y_utm_insar -- ", top_y_utm_insar, ", line_insar -- ", line_insar, ", col_insar -- ", col_insar, "\n"
    model = np.fromfile(model_file,dtype=np.float32).reshape(dimension,dimension)
    if np.remainder(dimension,2) == 0:
        index_x = np.arange(center_x_utm - (dimension / 2 - 1) * spacing, center_x_utm + (dimension /2 + 1) * spacing, spacing)
        index_y = np.arange(center_y_utm + (dimension / 2 - 1) * spacing, center_y_utm - (dimension /2 + 1) * spacing, spacing*(-1))
        ind_x = find_nearest(index_x,left_x_utm_insar)
        ind_y = find_nearest(index_y,top_y_utm_insar)
        print "index_x and index_y are: ", ind_x, ", ", ind_y, "\n"

    else:
        index_x = np.arange(center_x_utm - (dimension / 2) * spacing, center_x_utm + (dimension /2 + 1) * spacing,spacing)
        index_y = np.arange(center_y_utm + (dimension / 2) * spacing, center_y_utm - (dimension /2 + 1) * spacing,spacing*(-1))
        ind_x = find_nearest(index_x,left_x_utm_insar)
        ind_y = find_nearest(index_y,top_y_utm_insar)
        print "index_x and index_y are: ", ind_x, ", ", ind_y, "\n"

    model = model[ind_y:ind_y+line_insar,ind_x:ind_x+col_insar]
    print "Line 105, model size is: ", model.size, "\n"
    outFile = "carto_LOS.r4"
    model.astype('float32').tofile(outFile)
        
def corOut(f,dim):
    ''' 
       function used to correct the size of model output (usually several pixel more) and dimension (transpose)
       f: full path of the output file (string); dim = correct dimension (int)
    '''
    
    temp_def = np.fromfile(f,dtype=np.float32)
    if temp_def.size == dim*dim:
        print "File ",f, " is correct!\n"
        #temp_def1 = temp_def.reshape(dim,dim).T.reshape(dim*dim)
        #temp_def1.astype('float32').tofile(f)
    else:
        N = temp_def.size - dim*dim
        print "File ",f, " is " + str(N) + " pixels more!\n"
        temp_def = temp_def[:dim*dim]
        temp_def = temp_def  #convert to mm
        temp_def.astype('float32').tofile(f)
        
def proj2insar(fz,fx,fn,lookAng,azimuth):
    '''
        function used to project 3D displacement components onto LOS
        lookAng: InSAR incidence angle in degree, always positive and less than 90 (float)
        azimuth: InSAR azimuth angle, positive clockwise from North (float)
    '''
    print "input files are: ", fz, ", ", fx, ", ", fn, ", ", lookAng, ", ", azimuth, "\n" 
    dz = np.fromfile(fz.strip(),dtype=np.float32)
    dx = np.fromfile(fx.strip(),dtype=np.float32)
    dn = np.fromfile(fn.strip(),dtype=np.float32)
    lookAng = lookAng / 180.0 * np.pi
    azimuth = azimuth / 180.0 * np.pi
    def_LOS = dz * np.cos(lookAng) - dx * np.sin(lookAng) * np.cos(azimuth) + dn * np.sin(lookAng) * np.sin(azimuth) 
    def_LOS = def_LOS * 1000
    outFile = "carto_LOS.r4"
    if def_LOS[~np.isnan(def_LOS)].size == 0 or def_LOS.size == 0:
        print "size of def_LOS is incorrect!\n"
        exit(1)
    ### test for location
    #def_LOS = dz
    ###
    ### remove model file z,x,y
    #temp_str = "rm -f " + fz
    #os.system(temp_str)
    #temp_str = "rm -f " + fx
    #os.system(temp_str)
    #temp_str = "rm -f " + fn
    #os.system(temp_str)
    #temp_str = "rm -f ux_x"
    #os.system(temp_str)
    #temp_str = "rm -f ux_y"
    #os.system(temp_str)
    #temp_str = "rm -f uy_x"
    #os.system(temp_str)
    #temp_str = "rm -f uy_y"
    #os.system(temp_str)
    #temp_str = "rm -f uz_x"
    #os.system(temp_str)
    #temp_str = "rm -f uz_y"
    #os.system(temp_str)
    #temp_str = "rm -f elastic_model"
    #os.system(temp_str)

    def_LOS.astype('float32').tofile(outFile)
    

def cal_resi(d_list,loading,model_file,w=''):
    '''
        function used to calculate RMSE between model and InSAR time-series
        width: width of files; d_list: list for InSAR files (UTM, geotiff format) 
        loading: list containing load as a function of time
        model: model output on LOS
    '''
    
    resi = 0
    N_pix = 0
    temp_model = np.fromfile(model_file,dtype=np.float32)
    print "size of model is: ", temp_model.shape, "\n"
    for ni in range(len(d_list)):
        if ni == 0:
            continue
        else:
            dataset = gdal.Open(d_list[ni]) 
            temp_data = dataset.GetRasterBand(1).ReadAsArray().flatten()
            temp_data[temp_data==0] = np.nan   ### 0 is created from transform, used to fill the gap between coordinates
            temp_data[temp_data>10000] = np.nan     ### this is the original meaningful 0, consider potential spatial average, here the threshold is 10000 instead of 32768
            #temp_data = np.fromfile(d_list[ni],dtype=np.float32)
            print "size of data is: ", temp_data.shape, "\n"
            if w:
                temp_diff_sq = np.square(temp_data - temp_model * loading[0][ni])
                w_data = np.fromfile(w,dtype=np.float32)
                temp_data[np.isnan(w_data)] = np.nan
                #w_data[np.isnan(temp_data)] = 0
                N_pix = np.sum(np.abs(w_data[~np.isnan(w_data)]))
                w_temp_diff_sq = temp_diff_sq * w_data
                resi = resi + np.sum(np.abs(w_temp_diff_sq[~np.isnan(w_temp_diff_sq)])) / N_pix
            else:
                w_data = np.fromfile('/scratch/wlzhao/NSBAS/YAMZHO/Model/example_model1',dtype=np.float32)
                print "Reading mask file!\n"
                temp_data[np.isnan(w_data)] = np.nan
                temp_diff_sq = np.square(temp_data - temp_model * loading[0][ni])
                N_pix = temp_diff_sq[~np.isnan(temp_diff_sq)].size
                resi = resi + np.sum(temp_diff_sq[~np.isnan(temp_diff_sq)]) / N_pix
    
    R = np.sqrt(resi / (len(d_list) - 1))
    return R


def cal_resi_velo(model_file,load_list,velo_file,date_list,w=''):
    
    '''
        function used for residual calculation between 
        InSAR velocity and model. velocity of model is
        by multiplying velocity of load history and the 
        base model.
        model: base model file; col:number of column;
        load_list: loading history (negative means unloading), float python list
        velo_file: InSAR velocity file
    '''
    
    ### caluclate velocity of loading
    if load_list.size != len(date_list):
        print "Dimension inconsistent!\n"
        exit(1)
    load_list = load_list.flatten().astype('float32')
    date_list1 = []
    print "length of date_list: ", len(date_list)
    for ni in range(len(date_list)):
        temp = date_list[ni]
        date_list1.append(date(int(temp[0:4]),int(temp[4:6]),int(temp[6:8])))
        if ni > 0:
            date_list1[ni] = float((date_list1[ni] - date_list1[0]).days) / 365.25
    date_list1[0] = 0.
    date_list1 = np.asarray(date_list1)
    const_array = np.ones(load_list.size,dtype=np.float32)
    X = np.vstack((date_list1,const_array)).T.astype('float32')
    print "date_list1: ", date_list1.shape, ", const_array: ", const_array.shape, ", X: ", X.shape
    coef = np.linalg.lstsq(X,load_list)[0]
    velo = coef[0]
    print "Velocity of loading is: ", velo, "\n"
    
    resi = 0
    N_pix = 0
    temp_model = np.fromfile(model_file,dtype=np.float32)
    temp_model = temp_model * velo
    temp_model.astype('float32').tofile("model_velo")
    dataset = gdal.Open(velo_file)
    temp_data = dataset.GetRasterBand(1).ReadAsArray().flatten()
    temp_data[temp_data==0] = np.nan
    temp_data[temp_data>10000] = np.nan
    
    if w:
        temp_diff_sq = np.square(temp_data - temp_model)
        w_data = np.fromfile(w,dtype=np.float32)
        temp_data[np.isnan(w_data)] = np.nan
        #w_data[np.isnan(temp_data)] = 0
        N_pix = np.sum(np.abs(w_data[~np.isnan(w_data)]))
        w_temp_diff_sq = temp_diff_sq * w_data
        resi = np.sum(w_temp_diff_sq[~np.isnan(w_temp_diff_sq)]) / N_pix
    else:
        w_data = np.fromfile('/scratch/wlzhao/NSBAS/YAMZHO/Model/example_model1',dtype=np.float32)
        temp_data[np.isnan(w_data)] = np.nan
        temp_diff_sq = np.square(temp_data - temp_model)
        N_pix = temp_diff_sq[~np.isnan(temp_diff_sq)].size
        resi = np.sum(temp_diff_sq[~np.isnan(temp_diff_sq)]) / N_pix

    return np.sqrt(resi)


def cal_resi_velo_QT1(model_file,load_list,velo_file,date_list,col,cen_y,cen_x,w=''):

    '''
        function used for residual calculation between 
        InSAR velocity and model. velocity of model is
        by multiplying velocity of load history and the 
        base model. data vector is generated from a quadtree decoposition
        model: base model file; col:number of column;
        load_list: loading history (negative means unloading), float python list
        velo_file: InSAR velocity file
        date_list: python list including all dates
        col: number of column of data file
        max_l: maximum level pf quadtree (2^n)
        th: standard deviation threshold for quadtree
        w: file for weighting
    '''

    ### caluclate velocity of loading
    if load_list.size != len(date_list):
        print "Dimension inconsistent!\n"
        exit(1)
    load_list = load_list.flatten().astype('float32')
    date_list1 = []
    print "length of date_list: ", len(date_list)
    for ni in range(len(date_list)):
        temp = date_list[ni]
        date_list1.append(date(int(temp[0:4]),int(temp[4:6]),int(temp[6:8])))
        if ni > 0:
            date_list1[ni] = float((date_list1[ni] - date_list1[0]).days) / 365.25
    date_list1[0] = 0.
    date_list1 = np.asarray(date_list1)
    const_array = np.ones(load_list.size,dtype=np.float32)
    X = np.vstack((date_list1,const_array)).T.astype('float32')
    print "date_list1: ", date_list1.shape, ", const_array: ", const_array.shape, ", X: ", X.shape
    coef = np.linalg.lstsq(X,load_list)[0]
    velo = coef[0]
    print "Velocity of loading is: ", velo, "\n"

    N_pix = 0
    temp_model = np.fromfile(model_file,dtype=np.float32)
    temp_model = temp_model * velo
    temp_model.astype('float32').tofile("model_velo")
    dataset = gdal.Open(velo_file)
    temp_data = dataset.GetRasterBand(1).ReadAsArray().flatten()
    temp_data[temp_data==0] = np.nan
    temp_data[temp_data>10000] = np.nan

    w_data = np.fromfile('/scratch/wlzhao/NSBAS/YAMZHO/Model/example_model1',dtype=np.float32)
    temp_data[np.isnan(w_data)] = np.nan
    line = temp_data.size / col
    temp_data = temp_data.reshape(line,col)
    temp_model = temp_model.reshape(line,col)

    cen_y = cen_y.astype('int16')
    cen_x = cen_x.astype('int16')
    cen_y[cen_y>line-1] = line - 1
    cen_x[cen_x>col-1] = col - 1

    temp_diff_sq = np.square(temp_data[cen_y,cen_x] - temp_model[cen_y,cen_x])
    N_pixel = temp_diff_sq[~np.isnan(temp_diff_sq)].size
    resi = np.sum(temp_diff_sq[~np.isnan(temp_diff_sq)]) / N_pixel

    if np.isnan(resi):
        print "Residual cannot be nan!\n"
        exit(1)

    return np.sqrt(resi)


def prepCharge(f,col_f,dim_big):
    
    '''
        function used for putting a load file onto a big study area matrix.
        f: the input load file
        col_f: column of f
        dim_big: dimension (square) of the study area
    '''
    
    outDir = os.path.dirname(f)
    outfile = outDir + "/charge_big.bin"
    if os.path.isfile(outfile):
        print "Load file exits!\n"
        return 0
    else:
        data = np.fromfile(f,dtype=np.float32)
        lin_f = data.size / col_f
        data = data.reshape(lin_f,col_f)
        if lin_f % 2 == 0:
            cen_lin = lin_f / 2
        else:
            cen_lin = lin_f / 2 + 1
    
        if col_f % 2 == 0:
            cen_col = col_f / 2
        else:
            cen_col = col_f / 2 + 1

        if dim_big % 2 == 0:
            cen_big = dim_big / 2
        else:
            cen_big = dem_big / 2 + 1

        load_up_big_ind = cen_big - cen_lin
        load_left_big_ind = cen_big - cen_col
        data_big = np.zeros((dim_big,dim_big),np.float32)
        data_big[load_up_big_ind:load_up_big_ind+lin_f, load_left_big_ind:load_left_big_ind+col_f] = data
        data_big.astype("float32").tofile(outfile)
        #exit(0)
    
def cal_dateList(file_list):
   
    '''
        This function calculates a list of all date epochs from a file list (InSAR LOS displacement)
    '''

    t_list = []
    for ni in range(len(file_list)):
        temp_date = re.findall('\d\d\d\d\d\d\d\d',file_list[ni])[0].strip()   
        t_list.append(temp_date)

    return t_list           


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

'''
Created on Oct 11, 2010

@author: wzhao
'''
import os
import re
import numpy as np
import h5py
from pylab import *


def readProc(procfile):
    if re.search("^\$", procfile):
        t1 = procfile.search("/")
        t2 = os.getenv(procfile[1:t1])
        t2.strip()
        procfile = t2 + procfile[t1:]
    dictKey = {}
    InFile = file(procfile, 'r')
    FileList = InFile.readlines()
    for FileLine in FileList:
        if not re.search("=", FileLine):
            continue
        else:
            FileLine = FileLine.strip('\n')
            FileLine = FileLine.strip()
            [KeyName, Value] = [FileLine.split("=")[0],FileLine.split("=")[1]]
            if re.search('^#|^%', KeyName):
                continue
            else:
                if re.search('#',Value):
                    Value = Value.split('#')[0]
                KeyName = KeyName.strip()
                Value = Value.strip()
                if re.search("^\$", Value):
                    [temp1, temp2] = Value.split("/", 1)
                    temp3 = temp1[1:]
                    temp4 = os.getenv(temp3)
                    Value = temp4 + "/" + temp2
                Value = re.sub('\[|\]','',Value)
                if   re.search(':', Value): ValueList = Value.split(':')
                elif re.search(',', Value): ValueList = Value.split(',')
                elif re.search(' ', Value): ValueList = Value.split(' ')
                else: ValueList = Value
                dictKey[KeyName] = ValueList
    InFile.close()
    return dictKey    
         
        
def readCommandLine(commandline):
    dictKey = {}
    for Argument in commandline:
        if not re.search('=', Argument): continue
        else:
            Argument = Argument.strip()
            [KeyName, Value] = [Argument.split("=")[0],Argument.split("=")[1]] 
            KeyNmae = KeyName.strip()
            Value = Value.strip() 
            dictKey[KeyNmae] = Value
    return dictKey

def readRsc(inputFile):
    if re.search("^\$", inputFile):
        t1 = inputFile.search("/")
        t2 = os.getenv(inputFile[1:t1])
        t2.strip()
        inputFile = t2 + inputFile[t1:]
    dictKey = {}
    InFile = file(inputFile, 'r')
    FileList = InFile.readlines()
    for FileLine in FileList:
        if not re.search(" ", FileLine):
            continue
        else:
            #FileLine = FileLine.strip('\n')
            FileLine = FileLine.strip()
            [KeyName, Value] = [FileLine.split(" ", 1)[0].strip(),FileLine.split(" ", 1)[1].strip()]
            #if not re.search(KeyName, ":"):
            #    continue
            #else:
            #    Value = Value.strip()
            #    KeyName = KeyName[:-1]
            dictKey[KeyName] = Value
    InFile.close()
    #print "dict is: ", dictKey, "\n"
    return dictKey    

def getSensorInfo(Sat, StrDir):
    if Sat.upper() == "ENV":
        StrDirtemp = StrDir[(len(StrDir)-23):]
        [BeamMode, Orbit, yymmdd, hhmmss] = StrDirtemp.split('_')
        return BeamMode, Orbit, yymmdd, hhmmss
    elif Sat.upper() == "ALOS":
        StrDirtemp = StrDir[(len(StrDir)-18):]
        [Frame, yymmdd, hhmmss] = StrDirtemp.split('_')
        return Frame, yymmdd, hhmmss
    elif Sat.upper() == "ERS":
        StrDirtemp = StrDir[(len(StrDir)-25):]
        [dummy, Orbit, Frame, yymmdd, hhmmss] = StrDirtemp.split('_')
        return Orbit, Frame, yymmdd, hhmmss

def getSubdir(ParrentDir):
    SubList = os.listdir(ParrentDir)
    SubList1 = []
    i = 0
    for Dir in SubList:
        if os.path.isdir(ParrentDir + "/" + Dir): 
            SubList1.append(ParrentDir + "/" + Dir)
            i = i + 1
        else: continue
    return SubList1, i

def grep(String1,List):
    expr = re.compile(String1)
    for text in List:
        match = expr.search(text)
        if match != None:
            print match.String1

"""def getFrameNumber(rawfile, Prefix):
    if Prefix == "VDF":
        answer = os.popen("$INT_BIN/CEOS " rawfile " | grep 'FRAME' | awk '{print $6}'",'r')
        answer1 = answer.read()
        answer1 = answer1.strip()
        ff = answer1.split(" ")
        frame = ff[5]
        frame = frame.strip()
    elif Prefix == "LED":
        answer = os.popen("$INT_BIN/CEOS " rawfile " | grep 'FRAME' | awk '{print $6}'",'r')
        answer1 = answer.read()
        answer1 = answer1.strip() 
        ff = answer1.split(" ")
        frame = ff[5]
        frame = frame.strip()
        toks = frame.split("=")
        frame = toks[2]
    return frame"""

def getFile(inputdir, Suffix, Blink = 0):
    inputdir = inputdir.strip()
    if os.path.isdir(inputdir):
        if Blink == 0:
            if Suffix == "*.template":
                files = os.popen("find " + inputdir + " -maxdepth 1 -type f -iname '" + Suffix + "'")
                files1 = files.readlines()
            elif Suffix == "*.slc":
                files = os.popen("find " + inputdir + " -type f -name '" + Suffix + "'")
                files1 = files.readlines()
            elif Suffix == "*.slc.par":
                files = os.popen("find " + inputdir + " -type f -name '" + Suffix + "'")
                files1 = files.readlines()
            elif Suffix == "ASA_INS*":
                files = os.popen("find " + inputdir + " -maxdepth 1 -type f -name '" + Suffix + "'")
                files1 = files.readlines()
            elif Suffix == "ASA_XCA":
                files = os.popen("find " + inputdir + " -maxdepth 1 -type f -name '" + Suffix + "'")
                files1 = files.readlines()
            else:
                files = os.popen("find " + inputdir + " -type f -iname '" + Suffix + "'")
                files1 = files.readlines()
        elif Blink == 1:
            files = os.popen("find " + inputdir + " -type l -iname '" + Suffix + "'")
            files1 = files.readlines()
        for i in range(0, len(files1)):
                files1[i] = files1[i].strip()
    else: files1 = []
    return files1

def getPRF(rawfile, StrSatellite, StrPrefix, Seek_opt = 0):
    if (StrPrefix == "LED" and StrSatellite == "ALOS"):
        print "parameters are OK\n"
        Infile = file(rawfile, 'rb')
        print "open file OK\n"
        Infile.seek(1654, Seek_opt)
        print "find the postion\n"
        Value = Infile.read(16)
        print "get value is ", Value
        Value = Value.strip()       
    return Value

def readRscFile(file):
    '''
    read the .rsc file into a python dictionary structure
    '''
    rsc_dict = dict(np.loadtxt(file,dtype=str))
    return rsc_dict

def read_bifloat(file):
  '''
  Reads roi_pac unw, cor, or hgt data
  Requires the file path and returns amplitude and phase
  Usage:
  amp, phase, rsc = readUnw('/Users/sbaker/Desktop/geo_070603-070721_0048_00018.unw')
  '''
  rscBifloatContents = readRscFile(file + '.rsc')
  width = int(rscBifloatContents['WIDTH'])
  length = int(rscBifloatContents['FILE_LENGTH'])
  oddindices = np.where(np.arange(length*2)&1)[0]
  data = np.fromfile(file,np.float32)
  data = data.reshape(np.size(data)/width,width)
  length = np.size(data)/width/2
  print "widht is: ", width, ", and length is: ", length, "\n"
  #a = np.array([data.take(oddindices-1,axis=0)]).reshape(length,width)
  #p = np.array([data.take(oddindices,axis=0)]).reshape(length,width)
  a = data[::2,:]
  p = data[1::2,:]
  return a, p, rscBifloatContents, length, width

def readInt(file):
    '''
    Reads roi_pac int or slc data
    Requires the file path and returns amplitude and phase
    Usage:
    amp, phase, rsc = readInt('/Users/sbaker/Desktop/geo_070603-070721_0048_00018.int')
    '''
    rscIntContents = readRscFile(file + '.rsc')
    width = int(rscIntContents['WIDTH'])
    length = int(rscIntContents['FILE_LENGTH'])
    data = np.fromfile(file,np.complex64,length*2*width)
    length = np.size(data)/width
    data = data.reshape(length,width)
    a = np.array([np.hypot(data.real,data.imag)]).reshape(length,width)
    p = np.array([np.arctan2(data.imag,data.real)]).reshape(length,width)
    return a, p, rscIntContents,length,width

'''
def readCor(file):

    read a roipac geocoded coherence file

    rscCorContents = readRscFile(file + '.rsc')
    width = int(rscCorContents['WIDTH'])
    length = int(rscCorContents['FILE_LENGTH'])
    d=np.fromfile(file,dtype=np.float).reshape(length,width)
    return d, rscCorContents
'''

def readDEM(file):
    '''
    read a roipac dem file
    '''
    rscDEMContents = readRscFile(file + '.rsc')
    width = int(rscDEMContents['WIDTH'])
    length = int(rscDEMContents['FILE_LENGTH'])
    d=np.fromfile(file,dtype=np.int16).reshape(length,width)
    return d, rscDEMContents,length,width

def read_float(file):
    '''
    read a roipac float file
    '''
    rscHGTContents = readRscFile(file + '.rsc')
    width = int(rscHGTContents['WIDTH'])
    length = int(rscHGTContents['FILE_LENGTH'])
    p=np.fromfile(file,dtype=np.float32).reshape(length,width)
    return p, rscHGTContents,length,width

def change_key(dic1,key1,value1):
    for key, value in dic1.items():
        if key == key1:
            dic1[key] = value1

'''
def set_shade(a,intensity=None,cmap=cm.jet,scale=10.0,azdeg=165.0,altdeg=45.0,range=None):
    ### sets shading for data array based on intensity layer
    #or the data's value itself.
    inputs:
    a - a 2-d array or masked array
    intensity - a 2-d array of same size as a (no chack on that)
                    representing the intensity layer. if none is given
                    the data itself is used after getting the hillshade values
                    see hillshade for more details.
    cmap - a colormap (e.g matplotlib.colors.LinearSegmentedColormap
              instance)
    scale,azdeg,altdeg - parameters for hilshade function see there for
              more details
    output:
    rgb - an rgb set of the Pegtop soft light composition of the data and 
           intensity can be used as input for imshow()
    based on ImageMagick's Pegtop_light:
    http://www.imagemagick.org/Usage/compose/#pegtoplight
   ### 
    if intensity is None:
# hilshading the data
        intensity = hillshade(a,scale=10.0,azdeg=165.0,altdeg=45.0)
    else:
# or normalize the intensity
        intensity = (intensity - intensity.min())/(intensity.max() - intensity.min())
# get rgb of normalized data based on cmap
    if range:
        temp_list = range.split("_")
        v_min = float(temp_list[0].strip())
        v_max = float(temp_list[1].strip())
        rgb = cmap((a-v_min)/float(v_max-v_min))[:,:,:3]
    else:
        a[a==0] = np.nan
        v_min = a[np.invert(np.isnan(a))].min()
        v_max = a[np.invert(np.isnan(a))].max()
    print "color scale is: ",v_min,",",v_max,'\n'
    #print "old cmap is: ", cmap,"\n"
    cmap.set_bad('w',1.0)
    #print "new cmap is: ", cmap,"\n"
    temp_rgb = cmap((1.5-v_min)/float(v_max-v_min))
    print "temp_rgb is: ", temp_rgb[:3],"\n"
    rgb = cmap((a-v_min)/float(v_max-v_min))[:,:,:3]
# form an rgb eqvivalent of intensity
    print "rgb size is: ", rgb.shape,"\n"
    temp = rgb[:,:,0]
    temp[isnan(a)]=160/255.0
    rgb[:,:,0]=temp
    temp = rgb[:,:,1]
    temp[isnan(a)]=160/255.0
    rgb[:,:,1]=temp
    temp = rgb[:,:,2]
    temp[isnan(a)]=160/255.0
    rgb[:,:,2]=temp
    d = intensity.repeat(3).reshape(rgb.shape)
# simulate illumination based on pegtop algorithm.
    rgb = 2*d*rgb+(rgb**2)*(1-2*d)
    return rgb

def hillshade(data,scale=10,azdeg=165.0,altdeg=45.0):
    # convert data to hillshade based on matplotlib.colors.LightSource class.
    #input:
    #     data - a 2-d array of data
    #     scale - scaling value of the data. higher number = lower gradient
    #     azdeg - where the light comes from: 0 south ; 90 east ; 180 north ;
    #                  270 west
    #     altdeg - where the light comes from: 0 horison ; 90 zenith
    #output: a 2-d array of normalized hilshade
    #
  # convert alt, az to radians
    az = azdeg*pi/180.0
    alt = altdeg*pi/180.0
  # gradient in x and y directions
    dx, dy = gradient(data/float(scale))
    slope = 0.5*pi - arctan(hypot(dx, dy))
    aspect = arctan2(dx, dy)
    intensity = sin(alt)*sin(slope) + cos(alt)*cos(slope)*cos(-az - aspect - 0.5*pi)
    intensity = (intensity - intensity.min())/(intensity.max() - intensity.min())
    return intensity
'''  # no plot lib allowed 


def getFile_h5(keys_list,pattern):
    new_list = []
    for k in keys_list:
        if re.search(pattern,k):
            new_list.append(k)
    return new_list
            

def histeq(im,nbr_bins=256):
   im1 = im[~np.isnan(im)]
   im1 = (im1 - im1.min()) / (im1.max() - im1.min())
   #get image histogram
   imhist,bins = histogram(im1,nbr_bins,normed=True)
   print "bins is: ", bins
   cdf = imhist.cumsum() #cumulative distribution function
   #cdf = 255 * cdf / cdf[-1] #normalize
   cdf = 255 * (cdf - cdf.min()) / (im1.size - cdf.min())
   #use linear interpolation of cdf to find new pixel values
   im2 = interp(im1,bins[:-1],cdf)
   im[~np.isnan(im)] = im2
   im = np.power(im,3)
   return im, cdf

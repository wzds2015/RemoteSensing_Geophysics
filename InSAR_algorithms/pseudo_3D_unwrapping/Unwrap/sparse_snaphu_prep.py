#! /usr/bin/env python

''' sparse_snaphu_prep.py
    Prepring files used for sparse_snaphu.py  
    Transfer template parameters to snaphu_input
    Mainly for interferogram, coherence, amplitude
    W.Zhao, Mar.2014
'''

import os
import re
import sys
import Rsmas_Utilib as ut
import glob
import subprocess

if len(sys.argv) != 2:
    print '''
    Usage: sparse_snaphu.py *.template    
    template must include path to interferogram
    
'''
    exit(0)
    
para_list = ut.readProc(sys.argv[1])
int_str = para_list.get("unwrap_str")
#snaphu_def = os.getenv("INT_SCR").strip() + "/snaphu.default"
#para_snaphu1 = ut.readProc(snaphu_def)
if not para_list.get("method"):
    method = "RSMAS"
else:
    method = para_list.get("method").strip()
if method == "RSMAS":
    processdir = os.getenv("PROCESSDIR")
    #WDIR = processdir
    processdir = processdir + "/" + sys.argv[1].partition('.')[0].strip()
    folder_list = glob.glob(processdir+"/PO*")
elif method == "NSBAS":
    processdir = para_list.get("processdir").strip()
    #WDIR = processdir
    processdir = processdir + "/INTERFERO"
    folder_list = glob.glob(processdir+"/int_*")
Joblist1 = processdir + "/run_snaphu"
OP = open(Joblist1, 'w')
q = 1
print "length of int is: ", len(folder_list),"\n"
for fd in folder_list:
    Joblist = fd + "/sparse_para"
    IN = open(Joblist, 'w')
    if para_list.get("MASK"):
        maskfile = para_list.get("MASK").strip()
    cor_th = para_list.get("cor_th").strip()
    intfile = glob.glob(fd+"/*"+int_str)[0]
    corfile = intfile[:-4] + ".cor"
    N_looks = re.findall('\d+rlks',intfile)[0].strip()
    print "fd is: ", fd,"\n"
    print "N_looks is: ", N_looks,"\n"
    if not os.path.isfile(corfile):
        temp_corfile = glob.glob(fd+"/*"+N_looks+"*.cor")[0]
        temp_str = "cp -f " + temp_corfile + " " + corfile
        print "str is: ", temp_str, "\n"
        subprocess.Popen(temp_str,shell=True).wait()
        temp_str = "cp -f " + temp_corfile + ".rsc " + corfile + ".rsc"
        print "Copying rsc: ", temp_str, "\n"
        subprocess.Popen(temp_str,shell=True).wait()
    print "interfero is: ", intfile, "\n"
    [amplitude_int, phase_int, rscContent_int,N_line,N_col] = ut.readInt(intfile)
    if method == "RSMAS":
        temp_str = re.findall('\d\d\d\d\d\d-\d\d\d\d\d\d',intfile)[0].strip()
        date1 = re.findall('\d\d\d\d\d\d',temp_str)[0].strip()
        date2 = re.findall('\d\d\d\d\d\d',temp_str)[1].strip()
        SLC1file = fd + "/" + date1 + ".slc"
        SLC2file = fd + "/" + date2 + ".slc"
    elif method == "NSBAS":
        date1 = re.findall('\d\d\d\d\d\d\d\d',intfile)[0].strip()
        date2 = re.findall('\d\d\d\d\d\d\d\d',intfile)[1].strip()
        SLC1file = processdir + "/" + date1 + "/" + date1 + "_coreg.slc"
        SLC2file = processdir + "/" + date2 + "/" + date2 + "_coreg.slc"
    ratio_p = para_list.get("pixel_ratio").strip()    
    #temp_str = os.getenv("INT_SCR").strip() + "/look.pl " + date1 + ".slc " + N_looks + " " + str(int(N_looks)*int(ratio_p))
    #os.system(temp_str)
    if para_list.get("MASK"):
        [phase_M, rscContent_M,l_M,w_M] = ut.read_float(maskfile)
        N_line = l_M
    IN.write("interferogram="+intfile+"\n")
    IN.write("coherence="+corfile+"\n")
    IN.write("SLC1="+SLC1file+"\n")
    IN.write("SLC2="+SLC2file+"\n")
    IN.write("date1="+date1+"\n")
    IN.write("date2="+date2+"\n")
    IN.write("FILE_LENGTH="+str(N_line)+"\n")
    IN.write("WIDTH="+str(N_col)+"\n")
    IN.write("rlks="+N_looks+"\n")
    IN.write("pixel_ratio="+ratio_p+"\n")
    if para_list.get("MASK"):
        IN.write("MASK="+maskfile+"\n")
    if para_list.get("AMPFILE1"):
        amp1 = fd + "/amplitude1"
        IN.write("AMPFILE1="+amp1+"\n")
    if para_list.get("AMPFILE2"):
        amp2 = fd + "/amplitude2"
        IN.write("AMPFILE2="+amp2+"\n")
    IN.write("cor_th="+cor_th)
    IN.close()
    Joblist = fd + "/snaphu_para"
    IN = open(Joblist, 'w')
    for k,v in para_list.iteritems():
        if k.startswith("snaphu."):
             IN.write(k.partition(".")[2].strip()+"="+str(v)+"\n")
    IN.close()
    temp_name = os.path.basename(fd)
    temp_str = os.getenv("INT_SCR").strip() + "/sparse_snaphu.py " + temp_name
    print "Now is: ", temp_str, "\n"
    OP.write(temp_str)
    if q < len(folder_list):
        q = q + 1
        OP.write("\n")
OP.close()

WDIR = processdir
in_f = processdir + "/run_snaphu"
WT = "2:00"
INT_SCR = os.getenv("INT_SCR").strip()
temp_str = INT_SCR + "/createBatch.pl --workdir " + WDIR + " --infile " + in_f + " walltime=" + WT
temp_list = [INT_SCR + "/createBatch.pl","--workdir",WDIR,"--infile",in_f,"walltime="+WT]
sys.stderr.write(temp_str+"\n")
subprocess.Popen(temp_list).wait()

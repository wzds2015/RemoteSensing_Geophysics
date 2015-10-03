#! /usr/bin/env python

import os
import re
import sys
import Rsmas_Utilib as ut
import glob
import subprocess

def main():
    para_list = ut.readProc(sys.argv[1])
    interf_folder = para_list.get("interf").strip()
    pattern = para_list.get("pattern").strip()
    maskfile = para_list.get("mask").strip()
    th_d = para_list.get("th_d")
    if isinstance(th_d, str):
        th_d = [th_d]
    for ni in range(len(th_d)):
        th_d[ni] = th_d[ni].strip()
    th_c = para_list.get("th_c").strip()
    iteration = len(th_d)
    method = para_list.get("method").strip()
    solution = para_list.get("solution").strip()
    plane_type = para_list.get("plane_type").strip()
    APS_type = para_list.get("APS_type").strip()
    demfile = para_list.get("dem").strip()
    step = para_list.get("step")
    if isinstance(step, str):
        step = [step]
    for ni in range(len(step)):
        step[ni] = step[ni].strip()
    if para_list.get("software")=="RSMAS":
        int_list = glob.glob(interf_folder+"/PO*")
    else:
        int_list = glob.glob(interf_folder+"/int_*")
    
    N_line = 0
    for f in int_list:
        print "f is: ", f, "\n" 
        intfile = glob.glob(f+"/*"+pattern)[0]
        intfile_rsc = intfile + ".rsc"
        para_list1 = ut.readRsc(intfile_rsc)
        temp_line = int(para_list1.get("FILE_LENGTH").strip())
        if N_line < temp_line:
            N_line = temp_line
    print "maskfile is: ",bool(maskfile=='1'), "\n"
    if maskfile == '1':
        print "The maximum line is: ", N_line, "\n"
    else:
        Joblist = interf_folder + "/run_remove_ramp_aps"
        for ni in range(iteration):
            if ni < 2:
                method = "0"
            else:
                method = "1"
            k = 1
            IN = open(Joblist, 'w')
            for f in int_list:
                intfile = glob.glob(f+"/*"+pattern)[0]
                corfile = intfile[:-3]+"cor"
                if ni>5:
                    APS_type == "2"
                if ni % 2 == 0:
                    temp_str = "/nethome/wlzhao/python_code/remove_aps_wrap.py " + intfile + " " + corfile + " " + maskfile + " " + demfile + \
                               " " + plane_type + " " + "0" + " " + th_d[ni] + " " + th_c + " " + method + " " + \
                               solution + " " + str(ni+1) + " " + step[ni]
                else:
                    temp_str = "/nethome/wlzhao/python_code/remove_aps_wrap.py " + intfile + " " + corfile + " " + maskfile + " " + demfile + \
                               " " + "0" + " " + APS_type + " " + th_d[ni] + " " + th_c + " " + method + " " + \
                               solution + " " + str(2) + " " + step[ni]
                IN.write(temp_str)
                if k<len(int_list):
                    IN.write("\n")
                k = k+1
            IN.close() 

            WT = "0:30"
            temp_str = os.getenv("INT_SCR").strip() + "/createBatch_8.pl --workdir " + interf_folder + " --infile " + Joblist + " walltime=" + WT
            temp_list = [os.getenv("INT_SCR").strip()+"/createBatch_8.pl","--workdir",interf_folder,"--infile",Joblist,"walltime="+WT]
            sys.stderr.write(temp_str+"\n")
            output = subprocess.Popen(temp_list).wait()
            if output == 0:
                print "Iteration ", str(ni+1), " has been done!\n"
            else:
                print "There is error happened. Check code again!\n"
                exit(1)
            
        print "All the jobs are done! Good luck!\n"
        exit(0)

if __name__ == "__main__":
    sys.exit(main())


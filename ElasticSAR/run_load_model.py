#! /usr/bin/env python

import sys,os,subprocess

model_dir = sys.argv[1].strip()
os.chdir(model_dir) 
temp_str = model_dir + "/elastic_model"
temp_list = [model_dir+"/elastic_model"]
print temp_str, "\n"
output = subprocess.Popen(temp_list).wait()
if output != 0:
    print "Error occurred during modeling!\n"
    exit(1)

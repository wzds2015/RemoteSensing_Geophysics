#! /usr/bin/env python

import numpy as np
import quadtree

data = np.fromfile('test_model',dtype=np.float32).reshape(1024,1024)
temp_level,temp_center_x,temp_center_y,data_big = quadtree.quadtree(data,0.2,8)
quadtree.plot_QDT(temp_level,temp_center_x,temp_center_y,data_big)


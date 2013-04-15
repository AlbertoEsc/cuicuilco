#! /usr/bin/env python

#Functions for testing the SFAPCA node 

#By Alberto Escalante. Alberto.Escalante@ini.ruhr-uni-bochum.de First Version 1 August 2011
#Ruhr-University-Bochum, Institute of Neurocomputation, Theory of Neural Systems

import numpy
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import Image
import mdp
import sys
sys.path.append("/home/escalafl/work3/cuicuilco_MDP3.2/src/")
import more_nodes
import patch_mdp

import os, sys
import random
import sfa_libs
from sfa_libs import (scale_to, distance_squared_Euclidean, str3, wider_1Darray, ndarray_to_string, cutoff)
 
import time
from matplotlib.ticker import MultipleLocator
import copy
import string
import getopt

num_samples = 20
dim = 3
x = numpy.random.normal(size = (num_samples, dim))
t = numpy.linspace(0,numpy.pi, num_samples)
x[:,2] = x[:,2]*0.5 + (2.0**0.5) * numpy.cos(t)
x[:,1] *= 2

output_dim = 3
max_delta = 0.0

sfa_args = {"block_size":1, "train_mode":"regular"}
sfapca_node = more_nodes.SFAPCANode(max_delta=max_delta, output_dim=output_dim, sfa_args=sfa_args)
sfapca_node.train(x)
sfapca_node.stop_training()

y = sfapca_node.execute(x)
xp = sfapca_node.inverse(y)
print "x =",x
print "xp=",xp
print "y=",y
error = x - xp
print "error=",error
RMSE = ((error**2).sum(axis=1)**0.5).mean()
print "Reconstruction RMSE: ", RMSE

f0 = plt.figure()
ax = plt.subplot(3,1,1)
plt.plot(range(num_samples), x[:,0:3], "-")
plt.xlabel("First three input signals: max_delta=%04f"%(max_delta))
ax = plt.subplot(3,1,2)
plt.plot(range(num_samples), xp[:,0:3], "-")
plt.xlabel("First three recovered signals: max_delta=%04f"%(max_delta))
ax = plt.subplot(3,1,3)
plt.plot(range(num_samples), y[:,0:3], "-")
plt.xlabel("First three extracted signals: max_delta=%04f"%(max_delta))
plt.show()

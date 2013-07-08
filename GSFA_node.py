
import numpy
import scipy
import scipy.optimize
#import scipy.optimize
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import PIL
#import Image
import mdp
#import os
#import glob
#import random
##For patching MDP
#from mdp import numx
#from mdp.utils import (mult, pinv, symeig, CovarianceMatrix, QuadraticForm,
#                       SymeigException)
from mdp.utils import (mult, pinv, symeig, CovarianceMatrix, SymeigException)
#from mdp.utils import (mult, pinv, symeig, CovarianceMatrix, SymeigException)

import copy
#import time
#import SystemParameters
#import hashlib
#
import sys
##sys.path.append("/home/escalafl/workspace/hiphi/src/hiphi/utils")
##import misc
#import object_cache
#import more_nodes
from sfa_libs import select_rows_from_matrix
#from inversion import invert_exp_funcs2
import inspect

##################### GSFA functions/classes ################

#Special purpose object to compute the covariance matrices used by SFA.
#It efficiently supports special training methods: clustered, serial, mixed
#TODO: Remove falsely global unneeded variables
#TODO: 
class CovDCovMatrix(object):
    def __init__(self, block_size):
        self.block_size = block_size
        self.sum_x = None
        self.sum_prod_x = None
        self.num_samples = 0
        self.sum_diffs = None 
        self.sum_prod_diffs = None  
        self.num_diffs = 0
        self.last_block = None
#Permanent Results Storage
        self.cov_mtx = None
        self.avg = None
        self.dcov_mtx = None
        self.tlen = 0            

    def AddSamples(self, sum_prod_x, sum_x, num_samples, weight=1.0):
        weighted_sum_x = sum_x * weight
        weighted_sum_prod_x = sum_prod_x * weight
        weighted_num_samples = num_samples * weight
        
        if self.sum_prod_x is None:
            self.sum_prod_x = weighted_sum_prod_x
            self.sum_x = weighted_sum_x
        else:
            self.sum_prod_x = self.sum_prod_x + weighted_sum_prod_x
            self.sum_x = self.sum_x + weighted_sum_x

        self.num_samples = self.num_samples + weighted_num_samples

    def AddDiffs(self, sum_prod_diffs, num_diffs, weight=1.0):
        #weighted_sum_diffs = sum_diffs * weight
        weighted_sum_prod_diffs = sum_prod_diffs * weight
        weighted_num_diffs = num_diffs * weight
        
        if self.sum_prod_diffs is None:
            self.sum_prod_diffs = weighted_sum_prod_diffs
#            self.sum_diffs = weighted_sum_diffs
        else:
            self.sum_prod_diffs = self.sum_prod_diffs + weighted_sum_prod_diffs
#            self.sum_diffs = self.sum_diffs + weighted_sum_diffs

        self.num_diffs = self.num_diffs + weighted_num_diffs   
       
    #TODO:Add option to skip last sample from Cov part.
    #Add unlabeled samples to Cov matrix (DCov remains unmodified)
    def updateUnlabeled(self, x, weight=1.0):
        num_samples, dim = x.shape
                       
        sum_x = x.sum(axis=0) 
        sum_prod_x = mdp.utils.mult(x.T, x)
        self.AddSamples(sum_prod_x, sum_x, num_samples, weight)
    
    #TODO:Add option to skip last sample from Cov part.
    def updateRegular(self, x, weight=1.0):
        num_samples, dim = x.shape
        
        #Update Cov Matrix
        sum_x = x.sum(axis=0) 
        sum_prod_x = mdp.utils.mult(x.T, x)            
        self.AddSamples(sum_prod_x, sum_x, num_samples, weight)

        #Update DCov Matrix
        diffs = x[1:,:]-x[:-1,:]
        num_diffs = num_samples - 1
#        sum_diffs = diffs.sum(axis=0)
        sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
        self.AddDiffs(sum_prod_diffs, num_diffs, weight)

    #Usually: sum(node_weights)=num_samples 
    def updateGraph(self, x, node_weights=None, edge_weights=None, ignore_edge_avg=True):
        num_samples, dim = x.shape

        if node_weights==None:
            node_weights = numpy.ones(num_samples)

        if len(node_weights) != num_samples:
            er = "Node weights should be the same length as the number of samples"
            raise Exception(er)

        if edge_weights==None:
            er = "edge_weights should be a dictionary with entries: d[(i,j)] = w_{i,j}"
            raise Exception(er)

        node_weights_column = node_weights.reshape((num_samples, 1))                
        #Update Cov Matrix
        weighted_x = x * node_weights_column
    
        weighted_sum_x = weighted_x.sum(axis=0) 
        weighted_sum_prod_x = mdp.utils.mult(x.T, weighted_x)
        weighted_num_samples = node_weights.sum()
        print "weighted_num_samples=",  weighted_num_samples         
        self.AddSamples(weighted_sum_prod_x, weighted_sum_x, weighted_num_samples)

        #Update DCov Matrix
        num_diffs = len(edge_weights)
        diffs = numpy.zeros((num_diffs, dim))
        weighted_diffs = numpy.zeros((num_diffs, dim))
        weighted_num_diffs = 0
        for ii, (i,j) in enumerate(edge_weights.keys()):
            diff = x[j,:]-x[i,:]
            diffs[ii] = diff
            w_ij = edge_weights[(i,j)]
            weighted_diff = diff * w_ij
            weighted_diffs[ii] = weighted_diff
            weighted_num_diffs += w_ij

        weighted_sum_prod_diffs = mdp.utils.mult(diffs.T, weighted_diffs)
        self.AddDiffs(weighted_sum_prod_diffs, weighted_num_diffs)

    #Note, this method makes sense from consistency constraints for "larger" windows. 
    def updateMirroringSlidingWindow(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth #window_halfwidth is way too long to write it
        if 2*width >= num_samples:
            ex = "window_halfwidth %d not supported for %d samples!"%(width, num_samples)
            raise Exception(ex)

        #Update Cov Matrix
        #All samples have same weight
        sum_x = x.sum(axis=0) 
        sum_prod_x = mdp.utils.mult(x.T, x)       
        self.AddSamples(sum_prod_x, sum_x, num_samples, weight)
        
        #Update DCov Matrix
        #First mirror the borders
        x_mirror = numpy.zeros((num_samples+2*width, dim))
        x_mirror[width:-width] = x #center part
        x_mirror[0:width, :] = x[0:width, :][::-1,:] #first end
        x_mirror[-width:, :] = x[-width:, :][::-1,:] #second end

        #Center part
        x_full = x
        sum_prod_x_full = mdp.utils.mult(x_full.T, x_full)
#        print "sum_prod_x_full[0]=", sum_prod_x_full[0]
#        print "(2*width+1)*sum_prod_x_full=", (2*width+1)*sum_prod_x_full[0:3,0:3]

        Aacc123 = numpy.zeros((dim, dim))
        for i in range(0, 2*width): # [0, 2*width-1]
            Aacc123 += (i+1)*mdp.utils.mult(x_mirror[i:i+1,:].T, x_mirror[i:i+1,:]) #(i+1)=1,2,3..., 2*width

        for i in range(num_samples, num_samples+2*width): # [num_samples-width, num_samples-1]
            Aacc123 += (num_samples+2*width-i)*mdp.utils.mult(x_mirror[i:i+1,:].T, x_mirror[i:i+1,:])
#            print (num_samples+2*width-i)
        #Warning [-1,0] does not work!!!    
        #for i in range(0, 2*width): # [0, 2*width-1]
        #    Aacc123 += (i+1)*mdp.utils.mult(x_mirror[-(i+1):-i,:].T, x_mirror[-(i+1):-i,:]) #(i+1)=1,2,3..., 2*width
        x_middle = x_mirror[2*width:-2*width,:] #intermediate values of x, which are connected 2*width+1 times
        Aacc123 += (2*width+1)*mdp.utils.mult(x_middle.T, x_middle)
#        print "Aacc123[0:3,0:3]=", Aacc123[0:3,0:3]

        b = numpy.zeros((num_samples+1+2*width, dim))
        b[1:] = x_mirror.cumsum(axis=0)
        B = b[2*width+1:]-b[0:-2*width-1]
        Bprod = mdp.utils.mult(x_full.T, B)
#        print "Bprod[0:3,0:3]=", Bprod[0:3,0:3]
         
        sum_prod_diffs_full = (2*width+1)*sum_prod_x_full + (Aacc123) - Bprod - Bprod.T
        num_diffs = num_samples*(2*width) # removed zero differences
#         print "N3=", num_diffs
        self.AddDiffs(sum_prod_diffs_full, num_diffs, weight)        
    
    #Note, this method makes sense from consistency constraints for "larger" windows. 
    def updateSlowMirroringSlidingWindow(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth #window_halfwidth is way too long to write it
        if 2*width >= num_samples:
            ex = "window_halfwidth %d not supported for %d samples!"%(width, num_samples)
            raise Exception(ex)

      
        #Update Cov Matrix
        #All samples have same weight
        sum_x = x.sum(axis=0) 
        sum_prod_x = mdp.utils.mult(x.T, x)       
        self.AddSamples(sum_prod_x, sum_x, num_samples, weight)
                
        #Update DCov Matrix
        #window = numpy.ones(2*width+1) # Rectangular window
        x_mirror = numpy.zeros((num_samples+2*width, dim))
        x_mirror[width:-width] = x #center part
        x_mirror[0:width, :] = x[0:width, :][::-1,:] #first end
        x_mirror[-width:, :] = x[-width:, :][::-1,:] #second end
                
        diffs = numpy.zeros((num_samples, dim))
        #print "width=", width
        for offset in range(-width, width+1):
            if offset == 0:
                pass
            else:
                diffs = x_mirror[offset+width:offset+width+num_samples,:]-x
                                 
#                sum_diffs = diffs.sum(axis=0) 
                sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
                num_diffs = len(diffs)
# WARNING!
#                self.AddDiffs(sum_prod_diffs, sum_diffs, num_diffs, weight * window[width+offset])
                self.AddDiffs(sum_prod_diffs, num_diffs, weight)               

################## Truncating Window ####################
    def updateSlowTruncatingSlidingWindow(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth #window_halfwidth is way too long to write it
        if 2*width >= num_samples:
            ex = "window_halfwidth %d not supported for %d samples!"%(width, num_samples)
            raise Exception(ex)
      
        #Update Cov Matrix
        #All samples have same weight
        sum_x = x.sum(axis=0) 
        sum_prod_x = mdp.utils.mult(x.T, x)       
        self.AddSamples(sum_prod_x, sum_x, num_samples, weight)
                
        #Update DCov Matrix
        #window = numpy.ones(2*width+1) # Rectangular window
        x_extended = numpy.zeros((num_samples+2*width, dim))
        x_extended[width:-width] = x #center part is preserved, extreme samples are zero
                
        diffs = numpy.zeros((num_samples, dim))
        #print "width=", width
        for offset in range(1, width+1): #Negative offset is not considered because it is equivalent to the positive one, thereore the factor 2
            diffs = x_extended[offset+width:width+num_samples,:]-x[0:-offset,:]
            sum_prod_diffs = 2*mdp.utils.mult(diffs.T, diffs)
            num_diffs = 2*(num_samples-offset)
            self.AddDiffs(sum_prod_diffs, num_diffs, weight)               
#            diffs = x_extended[offset+width:offset+width+num_samples,:]-x
#            sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
#            num_diffs = 2*(num_samples-offset)
#            self.AddDiffs(sum_prod_diffs, num_diffs, weight)               

    def updateFastTruncatingSlidingWindow(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth #window_halfwidth is way too long to write it everytime
        if 2*width >= num_samples:
            ex = "window_halfwidth %d not supported for %d samples!"%(width, num_samples)
            raise Exception(ex)

        #Update Cov Matrix
        #All samples have same weight
        sum_x = x.sum(axis=0) 
        sum_prod_x = mdp.utils.mult(x.T, x)       
        self.AddSamples(sum_prod_x, sum_x, num_samples, weight)
        
        #Update DCov Matrix
        #First mirror the borders
        x_extended = numpy.zeros((num_samples+2*width, dim))
        x_extended[width:-width] = x #center part, the extremes are zero

        #Center part
        x_full = x #Also called y
        sum_prod_x_full = mdp.utils.mult(x_full.T, x_full)
#        print "sum_prod_x_full[0]=", sum_prod_x_full[0]
#        print "(2*width+1)*sum_prod_x_full=", (2*width+1)*sum_prod_x_full[0:3,0:3]

        Aacc123 = numpy.zeros((dim, dim)) 
        for i in range(0, 2*width): # [0, 2*width-1]
            Aacc123 += (i+1)*mdp.utils.mult(x_extended[i:i+1,:].T, x_extended[i:i+1,:]) #(i+1)=1,2,3..., 2*width

        for i in range(num_samples, num_samples+2*width): # [num_samples-width, num_samples-1]
            Aacc123 += (num_samples+2*width-i)*mdp.utils.mult(x_extended[i:i+1,:].T, x_extended[i:i+1,:])
#            print (num_samples+2*width-i)
        #Warning [-1,0] does not work!!!    
        #for i in range(0, 2*width): # [0, 2*width-1]
        #    Aacc123 += (i+1)*mdp.utils.mult(x_mirror[-(i+1):-i,:].T, x_mirror[-(i+1):-i,:]) #(i+1)=1,2,3..., 2*width
        x_middle = x_extended[2*width:-2*width,:] #intermediate values of x, which are connected 2*width+1 times
        Aacc123 += (2*width+1)*mdp.utils.mult(x_middle.T, x_middle)
#        print "Aacc123[0:3,0:3]=", Aacc123[0:3,0:3]

        b = numpy.zeros((num_samples+1+2*width, dim))
        b[1:] = x_extended.cumsum(axis=0)
        B = b[2*width+1:]-b[0:-2*width-1]
        Bprod = mdp.utils.mult(x_full.T, B)
#        print "Bprod[0:3,0:3]=", Bprod[0:3,0:3]
         
        sum_prod_diffs_full = (2*width+1)*sum_prod_x_full + (Aacc123) - Bprod - Bprod.T
        num_diffs = 2*(num_samples*width-width*(width+1)/2) # num_samples*(2*width) # removed zero differences
        self.AddDiffs(sum_prod_diffs_full, num_diffs, weight)  


    def updateFastTruncatingSlidingWindow2(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth #window_halfwidth is way too long to write it everytime
        if 2*width >= num_samples:
            ex = "window_halfwidth %d not supported for %d samples!"%(width, num_samples)
            raise Exception(ex)

        #Update Cov Matrix
        #All samples have same weight
        sum_x = x.sum(axis=0) 
        sum_prod_x = mdp.utils.mult(x.T, x)       
        self.AddSamples(sum_prod_x, sum_x, num_samples, weight)
        
        #Update DCov Matrix
        #First mirror the borders
        y_extended = numpy.zeros((num_samples+2*width, dim)) 
        y_extended[width:-width] = x #center part, the extremes are zero
        y = x
        
        y_ext_padded = numpy.zeros((num_samples+2*width+1, dim))
        y_ext_padded [1:]=y_extended
        y_ext_cumsum = y_ext_padded.cumsum(axis=0)
        e_hat = y_ext_cumsum[2*width+1:,:]-y_ext_cumsum[:-2*width-1,:]
        
        #Center part
        t1 = mdp.utils.mult(x.T, x) * (2*width+1)     
        t2 = mdp.utils.mult(e_hat.T, x)     

        t6 = numpy.zeros((dim, dim)) 
        for i in range(0, width): 
            t6 += (2*i+1)*mdp.utils.mult(y[i:i+1,:].T, y[i:i+1,:]) 
        t6 += (2*width+1)*mdp.utils.mult(y[width:num_samples-width].T,y[width:num_samples-width])
        for i in range(num_samples-width,num_samples): 
            t6 += (2*(num_samples-width-1)-2*i+2*width+1)*mdp.utils.mult(y[i:i+1,:].T, y[i:i+1,:]) 
                 
        sum_prod_diffs_full = t1-t2-t2.T+t6
        num_diffs = (num_samples*(2*width)-2*width*(width+1)/2) # not counting zero differences, unverified
        self.AddDiffs(sum_prod_diffs_full, num_diffs, weight)  
        print ":|"

################### Sliding window with node-weight correction #################    
    def updateFastSlidingWindow(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth
        if 2*width >= num_samples:
            ex = "window_halfwidth %d not supported for %d samples!"%(width, num_samples)
            raise Exception(ex)

#MOST CORRECT VERSION
        x_sel = x+0.0
        w_up = numpy.arange(width, 2*width) / (2.0*width)
        w_up = w_up.reshape((width, 1))
        w_down = numpy.arange(2*width-1,width-1,-1) / (2.0*width)
        w_down = w_down.reshape((width, 1))
        x_sel[0:width, :] =  x_sel[0:width, :] * w_up
        x_sel[-width:, :] =  x_sel[-width:, :] * w_down
        
        sum_x = x_sel.sum(axis=0) 
        sum_prod_x = mdp.utils.mult(x_sel.T, x)       #There was a bug here, x_sel used twice!!!
        self.AddSamples(sum_prod_x, sum_x, num_samples - (0.5 *window_halfwidth-0.5), weight)
        
        #Update DCov Matrix
        #First we compute the borders.
        #Left border
        for i in range(0, width): # [0, width -1]
            diffs = x[0:width+i+1,:]-x[i,:]
            sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
            num_diffs = len(diffs) - 1 # removed zero differences
#            print "N1=", num_diffs
#            print "sum_prod_diffs[0]=", sum_prod_diffs[0]
            self.AddDiffs(sum_prod_diffs, num_diffs, weight)
        #Right border
        for i in range(num_samples-width, num_samples): # [num_samples-width, num_samples-1]
            diffs = x[i-width:num_samples,:]-x[i,:]                                 
            sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
            num_diffs = len(diffs) - 1 # removed zero differences
#            print "N2=", num_diffs
#            print "sum_prod_diffs[0]=", sum_prod_diffs[0]
            self.AddDiffs(sum_prod_diffs, num_diffs, weight)


        #Center part
        x_full = x[width:num_samples-width,:]
        sum_prod_x_full = mdp.utils.mult(x_full.T, x_full)
#        print "sum_prod_x_full[0]=", sum_prod_x_full[0]

        Aacc123 = numpy.zeros((dim, dim))
        for i in range(0, 2*width): # [0, 2*width-1]
            Aacc123 += (i+1)*mdp.utils.mult(x[i:i+1,:].T, x[i:i+1,:]) #(i+1)=1,2,3..., 2*width

        for i in range(num_samples-2*width, num_samples): # [num_samples-width, num_samples-1]
            Aacc123 += (num_samples-i)*mdp.utils.mult(x[i:i+1,:].T, x[i:i+1,:]) #(num_samples-1)=2*width,...,3,2,1

        x_middle = x[2*width:num_samples-2*width,:] #intermediate values of x, which are connected 2*width+1 times

        Aacc123 += (2*width+1)*mdp.utils.mult(x_middle.T, x_middle)

##        a = numpy.zeros((num_samples+1, dim, dim))
##        x1 = numpy.reshape(x, (num_samples, dim, 1))
##        x2 = numpy.reshape(x, (num_samples, 1, dim))
##        pp =numpy.multiply(x1, x2)
##        #print "pp[0]", pp[0], pp.shape
##        #for i in range(1,num_samples+1):
##        #    #reimplement this using sumcum?
##        #    a[i] = a[i-1] + mdp.utils.mult(x[i-1:i,:].T, x[i-1:i,:])
##        #print "a[1]", a[1]
##        #
##        #a2 = a + 0.0
##        a[1:] = pp.cumsum(axis=0)
        #print "a[-1]", a[-1]
        #print "a2[-1]", a2[-1]
            #print "a[i]", a[i]
#        print "a[0:2]", a[0:2]
        b = numpy.zeros((num_samples+1, dim))
        b[1:] = x.cumsum(axis=0)
#        for i in range(1,num_samples+1):
#            b[i] = b[i-1] + x[i-1,:]
#        A = a[2*width+1:]-a[0:-2*width-1]
        B = b[2*width+1:]-b[0:-2*width-1]
#        Aacc = A.sum(axis=0)
#        print "Aacc[0]=", Aacc[0]
        Bprod = mdp.utils.mult(x_full.T, B)
#        print "Bprod[0]=", Bprod[0]
#        print sum_prod_x_full.shape, Aacc.shape, Bprod.shape
        sum_prod_diffs_full = (2*width+1)*sum_prod_x_full + (Aacc123) - Bprod - Bprod.T
        num_diffs = (num_samples-2*width)*(2*width) # removed zero differences
#         print "N3=", num_diffs
        sum_diffs = numpy.zeros(dim)
        self.AddDiffs(sum_prod_diffs_full, num_diffs, weight)        
    #Add sliding window data to covariance matrices
    def updateSlidingWindow(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth
        if 2*width >= num_samples:
            ex = "window_halfwidth %d not supported for %d samples!"%(width, num_samples)
            raise Exception(ex)

        #Update Cov Matrix
        #Warning, truncating samples to avoid edge problems
#TRUNCATED VERSION
#        x_sel = x[window_halfwidth:-window_halfwidth,:]
#        sum_x = x_sel.sum(axis=0) 
#        sum_prod_x = mdp.utils.mult(x_sel.T, x_sel)       
#        self.AddSamples(sum_prod_x, sum_x, num_samples-2*window_halfwidth, weight)

#MOST CORRECT VERSION
        x_sel = x+0.0
        w_up = numpy.arange(width, 2*width) / (2.0 * width)
        w_up = w_up.reshape((width, 1))
        w_down = numpy.arange(2*width-1,width-1,-1) / (2.0 * width)
        w_down = w_down.reshape((width, 1))
        x_sel[0:width, :] =  x_sel[0:width, :] * w_up
        x_sel[-width:, :] =  x_sel[-width:, :] * w_down
        
        sum_x = x_sel.sum(axis=0)
        #print "F:)",
        sum_prod_x = mdp.utils.mult(x_sel.T, x)       #Bug fixed!!! computing w * X^T * X, with X=(x1,..xN)^T
        self.AddSamples(sum_prod_x, sum_x, num_samples - (0.5 *window_halfwidth-0.5), weight) # weights verified
        
        #Update DCov Matrix
        #window = numpy.ones(2*width+1) # Rectangular window, used always here!
        
        diffs = numpy.zeros((num_samples-2*width, dim))
        #print "width=", width
        #This can be made faster (twice) due to symmetry
        for offset in range(-width, width+1):
            if offset == 0:
                pass
            else:
                if offset > 0:
                    diffs = x[offset:,:]-x[0:num_samples-offset,:]
#                else: #offset < 0, only makes sense if window asymmetric!
#                    abs_offset = -1 * offset
#                    diffs = x[0:num_samples-abs_offset,:] - x[abs_offset:,:]                                 
                    sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
                    num_diffs = len(diffs)
                    self.AddDiffs(sum_prod_diffs, num_diffs, weight)

# ERRONEOUS VERSION USED FOR PERFORMANCE REPORT: (leaves out several differences due to extreme samples)
#                diffs = x[width+offset:num_samples-width+offset,:]-x[width:num_samples-width,:]
#                                 
#                sum_diffs = diffs.sum(axis=0) 
#                sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
#                num_diffs = len(diffs)
## WARNING!
##                self.AddDiffs(sum_prod_diffs, sum_diffs, num_diffs, weight * window[width+offset])
#                self.AddDiffs(sum_prod_diffs, sum_diffs, num_diffs, weight)

#            def AddDiffs(self, sum_prod_diffs, sum_diffs, num_diffs, weight=1.0):
    #Add samples belonging to a serial training graph
    #TODO:Remove torify option 
    #NOTE: include_last_block not needed
    #Torify: copy first block to the end of the serial
    def updateSerial(self, x, block_size, Torify=False, weight=1.0, include_last_block=True):
        num_samples, dim = x.shape
        if block_size == None:
            er = "block_size must be specified"
            raise Exception(er)
            block_size = self.block_size
        if isinstance(block_size, (numpy.ndarray)):
            err = "Inhomogeneous block sizes not yet supported in updateSerial"
            raise Exception(err)
                    
        if Torify is True:    
            x2 = numpy.zeros((num_samples+block_size, dim))
            x2[0:num_samples] = x
            x2[num_samples:] = x[0:block_size]
            x = x2
            num_samples = num_samples + block_size
                       
        if num_samples % block_size > 0:
            err = "Consistency error: num_samples is not a multiple of block_size"
            raise Exception(err)
        num_blocks = num_samples / block_size

        #warning, plenty of dtype missing!!!!!!!!
        #Optimize computation of x.T ???
        #Warning, remove last element of x (incremental computation)!!!

        #Correlation Matrix. Computing sum of outer products (the easy part)
        xp = x[block_size:num_samples-block_size]
        x_b_ini = x[0:block_size]
        x_b_end =  x[num_samples-block_size:]
        sum_x = x_b_ini.sum(axis=0) + 2 * xp.sum(axis=0) + x_b_end.sum(axis=0)

#            print "Sum_x[0:3] is ", sum_x[0:3]
        sum_prod_x = mdp.utils.mult(x_b_ini.T, x_b_ini) + 2 *  mdp.utils.mult(xp.T, xp) + mdp.utils.mult(x_b_end.T, x_b_end)        
        num_samples = 2 * block_size + 2 * (num_samples- 2 * block_size)
        
        self.AddSamples(sum_prod_x, sum_x, num_samples, weight)
        
        self.last_block = x[num_samples-block_size:]

        #DCorrelation Matrix. Compute medias signal
        media = numpy.zeros((num_blocks, dim))
        for i in range(num_blocks):
            media[i] = x[i*block_size:(i+1)*block_size].sum(axis=0) * (1.0 / block_size)

        media_a = media[0:-1]
        media_b = media[1:]
        sum_prod_mixed_meds = (mdp.utils.mult(media_a.T, media_b) + mdp.utils.mult(media_b.T, media_a))
#        prod_first_media = numpy.outer(media[0], media[0]) * block_size
#        prod_last_media = numpy.outer(media[num_blocks-1], media[num_blocks-1]) * block_size
        prod_first_block = mdp.utils.mult(x[0:block_size].T, x[0:block_size])
        prod_last_block = mdp.utils.mult(x[num_samples-block_size:].T, x[num_samples-block_size:])

        #next line causes an exception index out of bounds: n_threads=10
#        sum_diffs = (media[num_blocks-1] - media[0]) * (block_size * block_size)  * (1.0 / block_size)
#        num_diffs = (block_size * block_size) * (num_blocks - 1)
#       WARNING? why did I remove one factor block_size?
        num_diffs = block_size * (num_blocks - 1)

#warning with factors!!!! they should be 2.0, 1.0, -1.0
#        sum_prod_diffs = block_size * (2.0 * sum_prod_x + 2.0 * prod_last_block - 0.0 *prod_first_block) - (block_size * block_size) * sum_prod_mixed_meds
        sum_prod_diffs = (block_size * (sum_prod_x) - (block_size * block_size) * sum_prod_mixed_meds) * (1.0 / block_size)
#        sum_prod_diffs = block_size * (2.0 * sum_prod_x + 2.0 * prod_last_block + 0.0 *prod_first_block) - (block_size * block_size) * sum_prod_mixed_meds
        self.AddDiffs(sum_prod_diffs, num_diffs, weight)

#Here block_size must be an array or list
#why block_sizes is not a parameter???
#This will be changed to updateClustered
#Weight should refer to node weights
    def update_clustered(self, x, block_sizes = None, weight=1.0):       
        num_samples, dim = x.shape

        if isinstance(block_sizes, (int)):
            return self.updateClustered_homogeneous_blocks(x, weight=weight, block_sizes=block_sizes)
        
        if block_sizes == None:
            er = "error, block_size not specified!!!!"
            raise Exception(er)
            block_sizes = self.block_size

        if num_samples != block_sizes.sum():
            err = "Inconsistency error: num_samples (%d) is not equal to sum of block_sizes:"%num_samples, block_sizes
            raise Exception(err)

#        num_blocks = len(block_sizes)
        
        counter_sample=0
        for block_size in block_sizes:
            #Note, here a sqrt might be useful to compensate for very unbalanced datasets
            #normalized_weight = weight * block_size * 1.0 / num_samples
            #Warning Warning
#            normalized_weight = weight / block_size #Warning! is this necessary!!??? I do sample balancing, in general what should be done???
            normalized_weight = weight
            self.update_clustered_homogeneous_block_sizes(x[counter_sample:counter_sample+block_size], weight=normalized_weight, block_size=block_size)
            counter_sample += block_size

# If the input is an array, the inhomogeneous function is used
# Create true updateClustered
# Change to updateClustered
#TODO: For consisency with paper: make sure edge weights are = 1/Ns, and that self-loops are not counted, so divide by Ns * (Ns-1)
  
    def update_clustered_homogeneous_block_sizes(self, x, weight=1.0, block_size=None):
        if block_size == None:
            er = "error, block_size not specified!!!!"
            raise Exception(er)
            block_size = self.block_size

        if isinstance(block_size, (numpy.ndarray)):
            er = "Error: inhomogeneous block sizes not supported by this function"
            raise Exception(er)
#            return self.updateClustered_inhomogeneous_blocks(x, weight=weight, block_sizes=block_size)
        
        #Assuming block_size is an integer:
        num_samples, dim = x.shape
        if num_samples % block_size > 0:
            err = "Inconsistency error: num_samples (%d) is not a multiple of block_size (%d)"%(num_samples, block_size)
            raise Exception(err)
        num_blocks = num_samples / block_size
        num_neighbours = block_size-1

        #warning, plenty of dtype missing!!!!!!!!

        #Optimize computation of x.T ???
        #Correlation Matrix. Computing sum of outer products, all elements are equally likely
        sum_x = x.sum(axis=0)
#            print "Sum_x[0:3] is ", sum_x[0:3]
        sum_prod_x = mdp.utils.mult(x.T, x)
        
        #Note that each node is in 2 * (N-1) links, half of it beginning the link, half ending it.
        #the number of nodes is N * B
        #WARNING THEORY; IT WAS:
        #weighted_sum_x = (2 * num_neighbours) * sum_x * weight
        #weighted_sum_prod_x = (2 * num_neighbours) * sum_prod_x * weight
        self.AddSamples(sum_prod_x, sum_x, num_samples, weight)

        self.last_block = None
        #DCorrelation Matrix. Compute medias signal
        media = numpy.zeros((num_blocks, dim))
        for i in range(num_blocks):
            media[i] = x[i*block_size:(i+1)*block_size].sum(axis=0) * (1.0 / block_size)
    
        sum_prod_meds = mdp.utils.mult(media.T, media)
        sum_diffs = numpy.zeros((1,dim))
        #note there are N * (N-1) * B links
        #WARNING!
        #num_diffs = (block_size * (block_size-1)) * num_blocks
        #WARNING!!!!!
#        num_diffs = (block_size-1) * num_blocks
#TODO: why such factor 0.5???
        num_diffs = block_size * 0.5 * num_blocks

        #WARNING!
        #sum_prod_diffs = (2 * block_size) * sum_prod_x - 2 * (block_size * block_size) * sum_prod_meds
#TODO: why the extra factor block_size in both terms, why divide here by num_neighbors??? both terms almost cancel.
        sum_prod_diffs = ((2 * block_size) * sum_prod_x - 2 * (block_size * block_size) * sum_prod_meds)/num_neighbours
        
        self.AddDiffs(sum_prod_diffs, num_diffs, weight)
        
    def addCovDCovMatrix(self, cov_dcov_mat, adding_weight=1.0, own_weight=1.0):
        if self.sum_prod_x is None:
            self.sum_prod_x = cov_dcov_mat.sum_prod_x * adding_weight
            self.sum_x = cov_dcov_mat.sum_x * adding_weight
        else:
            self.sum_prod_x = self.sum_prod_x * own_weight + cov_dcov_mat.sum_prod_x * adding_weight
            self.sum_x = self.sum_x * own_weight + cov_dcov_mat.sum_x * adding_weight
        self.num_samples = self.num_samples * own_weight + cov_dcov_mat.num_samples * adding_weight
        if self.sum_prod_diffs is None:
            self.sum_diffs = cov_dcov_mat.sum_diffs * adding_weight
            self.sum_prod_diffs = cov_dcov_mat.sum_prod_diffs * adding_weight
        else:
            self.sum_diffs = self.sum_diffs * own_weight + cov_dcov_mat.sum_diffs * adding_weight
            self.sum_prod_diffs = self.sum_prod_diffs * own_weight + cov_dcov_mat.sum_prod_diffs * adding_weight
        self.num_diffs = self.num_diffs * own_weight + cov_dcov_mat.num_diffs * adding_weight

    #This is not a good place to select whether to include the last sample or not... move it somewhere or remove it if not needed
    def fix(self, divide_by_num_samples_or_differences=True, include_tail=False, verbose=False, center_dcov=False):
        if verbose:
            print "Fixing CovDCovMatrix, with block_size=", self.block_size
  
        #Finalize covariance matrix of x
        if include_tail is True:
            print "Including data tail into computation of covariance matrix of shape:" + str(self.last_block.shape)
            self.sum_x = self.sum_x + self.last_block.sum(axis=0)
            self.sum_prod_x = self.sum_prod_x + mdp.utils.mult(self.last_block.T, self.last_block )
            self.num_samples = self.num_samples + self.block_size
            
        avg_x = self.sum_x * (1.0 / self.num_samples)
#        avg_x = avg_x.reshape((1,self.input_dim))

        #TEORY; This computation has a bias            
        #exp_prod_x = self.sum_prod_x * (1.0 / self.num_samples) 
        #prod_avg_x = numpy.outer(avg_x, avg_x)
        #cov_x = exp_prod_x - prod_avg_x
        prod_avg_x = numpy.outer(avg_x, avg_x)
        if divide_by_num_samples_or_differences: # as specified by the theory on Training Graphs
            cov_x = (self.sum_prod_x - self.num_samples * prod_avg_x) / (1.0 * self.num_samples)
        else: # standard unbiased estimation
            cov_x = (self.sum_prod_x - self.num_samples * prod_avg_x) / (self.num_samples-1.0)
        
        #Finalize covariance matrix of dx
        if divide_by_num_samples_or_differences:
            cov_dx = self.sum_prod_diffs / (1.0 * self.num_diffs)
        else:
            cov_dx = self.sum_prod_diffs / (self.num_diffs-1.0)
            
        self.cov_mtx = cov_x
        self.avg = avg_x
        self.tlen = self.num_samples            
        self.dcov_mtx = cov_dx
        
#        print "cov_mtx[0]", self.cov_mtx[0]
#        print "avg", self.avg
#        print "dcov_mtx[0]", self.dcov_mtx[0]
#        quit()

#Safely decomment for debugging
#        print "Finishing training CovDcovMtx: ",  self.num_samples, " num_samples, and ", self.num_diffs, " num_diffs"
#        print "Avg[0:3] is", self.avg[0:4]
#        print "Prod_avg_x[0:3,0:3] is", prod_avg_x[0:3,0:3]
#        print "Cov[0:3,0:3] is", self.cov_mtx[0:3,0:3]
#        print "DCov[0:3,0:3] is", self.dcov_mtx[0:3,0:3]
#        print "AvgDiff[0:4] is", avg_diff[0:4]
#        print "Prod_avg_diff[0:3,0:3] is", prod_avg_diff[0:3,0:3]
#        print "Sum_prod_diffs[0:3,0:3] is", self.sum_prod_diffs[0:3,0:3]
#        print "exp_prod_diffs[0:3,0:3] is", exp_prod_diffs[0:3,0:3]
      
        return self.cov_mtx, self.avg, self.dcov_mtx 

######## Helper functions for parallel processing and CovDcovMatrices #########
#This function appears to be obsolete
def ComputeCovMatrix(x, verbose=False):
    print "PCov",
    if verbose:
        print "Computation Began!!! **********************************************************8"
        sys.stdout.flush()
    covmtx = CovarianceMatrix(bias=True)
    covmtx.update(x)
    if verbose:
        print "Computation Ended!!! **********************************************************8"
        sys.stdout.flush()
    return covmtx

def ComputeCovDcovMatrixClustered(params, verbose=False):
    print "PComp",
    if verbose:
        print "Computation Began!!! **********************************************************8"
        sys.stdout.flush()
    x, block_size, weight = params
    covdcovmtx = CovDCovMatrix(block_size)
    covdcovmtx.update_clustered_homogeneous_block_sizes(x, block_size=block_size, weight=weight)
    if verbose:
        print "Computation Ended!!! **********************************************************8"
        sys.stdout.flush()
    return covdcovmtx

def ComputeCovDcovMatrixSerial(params, verbose=False):
    print "PSeq",
    if verbose:
        print "Computation Began!!! **********************************************************8"
        sys.stdout.flush()
    x, block_size = params
    Torify= False
    covdcovmtx = CovDCovMatrix(block_size)
    covdcovmtx.updateSerial(x, block_size = block_size, Torify=Torify)
    if verbose:
        print "Computation Ended!!! **********************************************************8"
        sys.stdout.flush()
    return covdcovmtx

def ComputeCovDcovMatrixMixed(params, verbose=False):
    print "PMixed",
    if verbose:
        print "Computation Began!!! **********************************************************8"
        sys.stdout.flush()
    x, block_size = params
    bs=block_size
    covdcovmtx = CovDCovMatrix(block_size)
    covdcovmtx.update_clustered_homogeneous_block_sizes(x[0:bs], block_size=block_size, weight=0.5)
    covdcovmtx.update_clustered_homogeneous_block_sizes(x[bs:-bs], block_size=block_size, weight=1.0)
    covdcovmtx.update_clustered_homogeneous_block_sizes(x[-bs:], block_size=block_size, weight=0.5)
    covdcovmtx.updateSerial(x, block_size = block_size, Torify=False)            
    if verbose:
        print "Computation Ended!!! **********************************************************8"
        sys.stdout.flush()
    return covdcovmtx

#This class is derived from SFANode only because I need the function _set_range
class GSFANode(mdp.nodes.SFANode):
    def __init__(self, input_dim=None, output_dim=None, dtype=None, block_size=None, train_mode=None, sfa_expo=None, pca_expo=None, magnitude_sfa_biasing=None):
        super(GSFANode, self).__init__(input_dim, output_dim, dtype)
        #Warning! bias activated, "courtesy" of Alberto
        # init two covariance matrices
        # one for the input data
###        self._cov_mtx = CovarianceMatrix(dtype, bias=True)
        # one for the derivatives
###        self._dcov_mtx = CovarianceMatrix(dtype, bias=True)
    
        self.pinv = None
###        self._symeig = symeig
        self.block_size= block_size
        self.train_mode = train_mode
    
        self.sum_prod_x = None
        self.sum_x = None
        self.num_samples = 0
        self.sum_diff = None 
        self.sum_prod_diff = None  
        self.num_diffs = 0
        
###        self.sfa_expo=sfa_expo # 1.2: 1=Regular SFA directions, >1: schrink derivative in the direction of the principal components
###        self.pca_expo=pca_expo # 0.25: 0=No magnitude reduction 1=Whitening
###        self.magnitude_sfa_biasing = magnitude_sfa_biasing
        self._myvar = None
        self._covdcovmtx = CovDCovMatrix(block_size)
        self.list_train_params = ["scheduler", "n_parallel", "train_mode", "block_size"] #Parameters accepted during training
        
    def _inverse(self, y):
        #code for storing pseudoinverse courtesy of Alberto Escalante
        if self.pinv is None:
            self.pinv = pinv(self.sf)
    #        print "SFA.pinv = ", self.pinv
    #        print "Shape of SFA.pinv = ", self.pinv.shape
    #        print "SFA.sf = ", self.sf
    #        print "Shape of sf = ", self.sf.shape        
    #        sf_t = self.sf.T
    #        print "sf_t= ", sf_t
    #        
    #        m2 = mult(sf_t, self.sf)
    #        print "m2 = ", m2
    #        print "For orthonormal sf, m2 is the identity"
    #
    #        m3 = mult(self.sf, sf_t)
    #        print "m3 = ", m3
    #        print "just curiosity"
    #        
    #        s_mod = (self.sf * self.sf).sum(axis=0)
    #        print "s_mod = ", s_mod
    #        print "(sf/s_mod).T= ", (self.sf / s_mod).T
        return mult(y, self.pinv)+self.avg

    def _train_with_scheduler(self, x, block_size=None, train_mode = None, node_weights=None, edge_weights=None, scheduler = None, n_parallel=None):      
        self._train_phase_started = True
        if train_mode == None:
            train_mode = self.train_mode
        if block_size == None:
            block_size = self.block_size
        if scheduler == None or n_parallel == None or train_mode == None:
    #        print "NO parallel sfa done...  scheduler=", ,uler, " n_parallel=", n_parallel
            return self._train(self, x, block_size=block_size, train_mode=train_mode, node_weights=node_weights, edge_weights=edge_weights)
        else:
    #        self._covdcovmtx.update_clustered_homogeneous_block_sizes(x, weight=1.0)
    
    #        chunk_size=None 
    #        num_chunks = n_parallel
            num_chunks = min(n_parallel, x.shape[0]/block_size) #WARNING, cover case where the division is not exact
            #here chunk_size is given in blocks!!!
    #        chunk_size = int(numpy.ceil((x.shape[0]/block_size)*1.0/num_chunks))
            chunk_size = int((x.shape[0]/block_size)/num_chunks)
            
            #Notice that parallel training doesn't work with clustered mode and inhomogeneous blocks
            #TODO:Fix this
            print "%d chunks, of size %d blocks, last chunk contains %d blocks"%(num_chunks, chunk_size, (x.shape[0]/block_size)%chunk_size)
            if train_mode == 'clustered':
                #TODO: Implement this
                for i in range(num_chunks):
                    print "Adding scheduler task //////////////////////"
                    sys.stdout.flush()
                    if i < num_chunks - 1:
                        scheduler.add_task((x[i*block_size*chunk_size:(i+1)*block_size*chunk_size], block_size, 1.0), ComputeCovDcovMatrixClustered)
                    else: # i == num_chunks - 1
                        scheduler.add_task((x[i*block_size*chunk_size:], block_size, 1.0), ComputeCovDcovMatrixClustered)
                    print "Done Adding scheduler task ///////////////////"
                    sys.stdout.flush()
            elif train_mode == 'serial':
                for i in range(num_chunks):
                    if i==0:
                        print "adding task %d from sample "%i, i*block_size*chunk_size, "to", (i+1)*block_size*chunk_size
                        scheduler.add_task((x[i*block_size*chunk_size:(i+1)*block_size*chunk_size], block_size), ComputeCovDcovMatrixSerial)
                    elif i==num_chunks-1: #Add one previous block to the chunk
                        print "adding task %d from sample "%i, i*block_size*chunk_size-block_size, "to the end"
                        scheduler.add_task((x[i*block_size*chunk_size-block_size:], block_size), ComputeCovDcovMatrixSerial)
                    else:
                        print "adding task %d from sample "%i, i*block_size*chunk_size-block_size, "to", (i+1)*block_size*chunk_size
                        scheduler.add_task((x[i*block_size*chunk_size-block_size:(i+1)*block_size*chunk_size], block_size), ComputeCovDcovMatrixSerial)
                        
            elif train_mode == 'mixed':
                bs = block_size
                self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[0:bs], weight=0.5)
    
    #           xxxx self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[bs:-bs], weight=1.0)
                x2 = x[bs:-bs]
    #            num_chunks2 = int(numpy.ceil((x2.shape[0]/block_size-2)*1.0/chunk_size))
                num_chunks2 = int((x2.shape[0]/block_size-2)/chunk_size)
                for i in range(num_chunks2):
                    if i < num_chunks2-1:
                        scheduler.add_task((x2[i*block_size*chunk_size:(i+1)*block_size*chunk_size], block_size, 1.0), ComputeCovDcovMatrixClustered)
                    else:
                        scheduler.add_task((x2[i*block_size*chunk_size:], block_size, 1.0), ComputeCovDcovMatrixClustered)
                        
                self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[-bs:], weight=0.5)
    
    #           xxxx self._covdcovmtx.updateSerial(x, Torify=False)            
                for i in range(num_chunks):
                    if i==0:
                        scheduler.add_task((x[i*block_size*chunk_size:(i+1)*block_size*chunk_size], block_size), ComputeCovDcovMatrixSerial)
                    elif i==num_chunks-1: #Add one previous block to the chunk
                        scheduler.add_task((x[i*block_size*chunk_size-block_size:], block_size), ComputeCovDcovMatrixSerial)
                    else: #Add one previous block to the chunk
                        scheduler.add_task((x[i*block_size*chunk_size-block_size:(i+1)*block_size*chunk_size], block_size), ComputeCovDcovMatrixSerial)
            elif train_mode == 'unlabeled':
                #TODO: IMPLEMENT THIS
                for i in range(num_chunks):
                    print "Adding scheduler task //////////////////////"
                    sys.stdout.flush()
                    #BUG:Why am I adding here the clustered graph!!!
                    scheduler.add_task((x[i*block_size*chunk_size:(i+1)*block_size*chunk_size], block_size, 1.0), ComputeCovDcovMatrixClustered)
                    print "Done Adding scheduler task ///////////////////"
                    sys.stdout.flush()
            #TODO:ADD support for regular/standar mode
            else:
                ex = "Unknown training method:", self.train_mode
                raise Exception(ex)
            
            print "Getting results"
            sys.stdout.flush()
    
            results = scheduler.get_results()
    #        print "Shutting down scheduler"
            sys.stdout.flush()
    
            for covdcovmtx in results:
                self._covdcovmtx.addCovDCovMatrix(covdcovmtx)

    def _train(self, x, block_size=None, train_mode = None, node_weights=None, edge_weights=None):
        if train_mode == None:
            train_mode = self.train_mode
#            train_mode = ""
        if block_size == None:
#            er="SFA no block_size"
#            raise Exception(er)
            block_size = self.block_size
        
        self._myvar=1
        self.set_input_dim(x.shape[1])
    
        ## update the covariance matrices
        # cut the final point to avoid a trivial solution in special cases
        # WARNING: Force artificial training
        print "train_mode=", train_mode
        if train_mode == 'unlabeled':
            print "updateUnlabeled"
            self._covdcovmtx.updateUnlabeled(x, weight=0.00015) #Warning, set this weight appropiately!
        elif train_mode == "regular":
            print "updateRegular"
            self._covdcovmtx.updateRegular(x, weight=1.0)
        elif train_mode == 'clustered':
            print "update_clustered_homogeneous_block_sizes"
            self._covdcovmtx.update_clustered_homogeneous_block_sizes(x, weight=1.0, block_size=block_size)
        elif train_mode == 'serial':
            print "updateSerial"
            self._covdcovmtx.updateSerial(x, Torify=False, block_size=block_size)
        elif train_mode == 'mixed':
            print "update mixed"
            bs = block_size
    # WARNING: THIS Generates degenerated solutions!!! Check code!! it should actually work fine??!!
            self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[0:bs], weight=2.0, block_size=block_size)
            self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[bs:-bs], weight=1.0, block_size=block_size)
            self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[-bs:], weight=2.0, block_size=block_size)
    #        self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[0:bs], weight=0.5, block_size=block_size)
    #        self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[bs:-bs], weight=1.0, block_size=block_size)
    #        self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[-bs:], weight=0.5, block_size=block_size)
            self._covdcovmtx.updateSerial(x, Torify=False, block_size=block_size)            
        elif train_mode.startswith('window'):
            window_halfwidth = int(train_mode[len('window'):])
            print "Window (%d)"%window_halfwidth
            self._covdcovmtx.updateSlidingWindow(x, weight=1.0, window_halfwidth=window_halfwidth)
        elif train_mode.startswith('fwindow'):
            window_halfwidth = int(train_mode[len('fwindow'):])
            print "Fast Window (%d)"%window_halfwidth
            self._covdcovmtx.updateFastSlidingWindow(x, weight=1.0, window_halfwidth=window_halfwidth)
        elif train_mode.startswith('mirror_window'):
            window_halfwidth = int(train_mode[len('mirror_window'):])
            print "Mirroring Window (%d)"%window_halfwidth
            self._covdcovmtx.updateMirroringSlidingWindow(x, weight=1.0, window_halfwidth=window_halfwidth)
        elif train_mode.startswith('smirror_window'):
            window_halfwidth = int(train_mode[len('smirror_window'):])
            print "Slow Mirroring Window (%d)"%window_halfwidth
            self._covdcovmtx.updateSlowMirroringSlidingWindow(x, weight=1.0, window_halfwidth=window_halfwidth)
        elif train_mode.startswith('ftruncate_window'):
            window_halfwidth = int(train_mode[len('ftruncate_window'):])
            print "Fast Truncating Window (%d)"%window_halfwidth
            self._covdcovmtx.updateFastTruncatingSlidingWindow2(x, weight=1.0, window_halfwidth=window_halfwidth)
        elif train_mode.startswith('truncate_window'):
            window_halfwidth = int(train_mode[len('truncate_window'):])
            print "Slow Truncating Window (%d)"%window_halfwidth
            self._covdcovmtx.updateSlowTruncatingSlidingWindow(x, weight=1.0, window_halfwidth=window_halfwidth)
        elif train_mode == 'graph':
            print "updateGraph"
            self._covdcovmtx.updateGraph(x, node_weights=node_weights, edge_weights=edge_weights)
        else:
            ex = "Unknown training method %s or window halfwidth not specified"%train_mode
            raise Exception(ex)

    def _stop_training(self, debug=False, verbose=False, pca_term = 0.995, pca_exp=2.0):
    #    self._myvar = self._myvar * 10
    #    Warning, changed block_size to artificial_training
    #    if (self.block_size == None) or (isinstance(self.block_size, int) and self.block_size == 1):
    #   WARNING!!! WARNING!!!
    #TODO: Define a proper way to fix training matrices... add training mode "regular" to SFA???
        if ((self.block_size == None) or (isinstance(self.block_size, int) and self.block_size == 1)) and False:
            ##### request the covariance matrices and clean up
            self.cov_mtx, self.avg, self.tlen = self._cov_mtx.fix()
            del self._cov_mtx
            self.dcov_mtx, davg, dtlen = self._dcov_mtx.fix()
            del self._dcov_mtx
            print "Finishing training (regular2222). %d tlen, %d tlen of diff"%(self.tlen, dtlen)
            print "Avg[0:3] is", self.avg[0:3]
            print "Cov[0:3,0:3] is", self.cov_mtx[0:3,0:3]
            print "DCov[0:3,0:3] is", self.dcov_mtx[0:3,0:3]
        else:
            if verbose or True:
                print "stop_training: Warning, using experimental SFA training method, with self.block_size=", self.block_size
            print "self._covdcovmtx.num_samples = ", self._covdcovmtx.num_samples 
            print "self._covdcovmtx.num_diffs= ", self._covdcovmtx.num_diffs
            self.cov_mtx, self.avg, self.dcov_mtx = self._covdcovmtx.fix()
                   
            print "Finishing SFA training: ",  self.num_samples, " num_samples, and ", self.num_diffs, " num_diffs"
    #        print "Avg[0:3] is", self.avg[0:4]
    #        print "Prod_avg_x[0:3,0:3] is", prod_avg_x[0:3,0:3]
    #        print "Cov[0:3,0:3] is", self.cov_mtx[0:3,0:3]
            print "DCov[0:3,0:3] is", self.dcov_mtx[0:3,0:3]
    #        quit()
    
        if pca_term != 0.0 and False:
            signs = numpy.sign(self.dcov_mtx)
            self.dcov_mtx = ((1.0-pca_term) * signs * numpy.abs(self.dcov_mtx) ** (1.0/pca_exp) + pca_term * numpy.identity(self.dcov_mtx.shape[0])) ** pca_exp          
            self.dcov_mtx = numpy.identity(self.dcov_mtx.shape[0])
        rng = self._set_range()
        
            
        #### solve the generalized eigenvalue problem
        # the eigenvalues are already ordered in ascending order
        #SUPERMEGAWARNING, moved dcov to the second argument!!!
        #self.d, self.sf = self._symeig(self.dcov_mtx, self.cov_mtx, range=rng, overwrite=(not debug))
        try:
            print "***Range used=", rng
###            if self.sfa_expo != None and self.pca_expo!=None:
###                self.d, self.sf = _symeig_fake_regularized(self.dcov_mtx, self.cov_mtx, range=rng, overwrite=(not debug), sfa_expo=self.sfa_expo, pca_expo=self.pca_expo, magnitude_sfa_biasing=self.magnitude_sfa_biasing)
###            else:
            self.d, self.sf = self._symeig(self.dcov_mtx, self.cov_mtx, range=rng, overwrite=(not debug))
            d = self.d
            # check that we get only *positive* eigenvalues
            if d.min() < 0:
                raise SymeigException("Got negative eigenvalues: %s." % str(d))
        except SymeigException, exception:
            errstr = str(exception)+"\n Covariance matrices may be singular."
            raise Exception(errstr)
    
        # delete covariance matrix if no exception occurred
###        del self.cov_mtx
###        del self.dcov_mtx
        del self._covdcovmtx
        # store bias
        self._bias = mult(self.avg, self.sf)
        print "shape of SFANode.sf is=", self.sf.shape

def graph_delta_values(y, edge_weights):
    R = 0
    deltas = 0
    for (i, j) in edge_weights.keys():
        w_ij = edge_weights[(i,j)]
        deltas += w_ij * (y[j]-y[i])**2
        R += w_ij
    return deltas/R

def comp_delta(x):
    xderiv = x[1:, :]-x[:-1, :]
    return (xderiv**2).mean(axis=0)

#Basic experiment
#from GSFA_node import *
basic_test_GSFA_edge_dict=True and False
if basic_test_GSFA_edge_dict:
    print "******************************************************************"
    print "*Basic test of GSFA on random data and graph, edge dictionary mode"
    x = numpy.random.normal(size=(200,15))
    v = numpy.ones(200)
    e = {}
    for i in range(1500):
        n1 = numpy.random.randint(200)
        n2 = numpy.random.randint(200)
        e[(n1,n2)] = numpy.random.normal()+1.0
    n = GSFANode(output_dim=5)
    n.train(x, train_mode="graph", node_weights=v, edge_weights=e)
    n.stop_training()
    
    y = n.execute(x)
    print "Graph delta values of training data", graph_delta_values(y, e)
    
    x2 = numpy.random.normal(size=(200,15))
    y2 = n.execute(x2)
    print "Graph delta values of test data (should be larger than for training)", graph_delta_values(y2, e)

test_equivalence_SFA_GSFA_linear_graph=True and False
if test_equivalence_SFA_GSFA_linear_graph:
    print ""
    print "*********************************************************************"
    print "Testing equivalence of Standard SFA and an appropriate graph for GSFA"
    x = numpy.random.normal(size=(200,15))
    x2 = numpy.random.normal(size=(200,15))
    
    v = numpy.ones(200)
    e = {}
    for t in range(199):
        e[(t,t+1)] = 1.0
    print "Training GSFA:"
    n = GSFANode(output_dim=5)
    n.train(x, train_mode="graph", node_weights=v, edge_weights=e)
    n.stop_training()
    
    print "Training SFA:"
    n_sfa = mdp.nodes.SFANode(output_dim=5)
    n_sfa.train(x)
    n_sfa.stop_training()
    
    print "/"*10, "Brute delta values of GSFA features (training/test):"
    y = n.execute(x)
    print graph_delta_values(y,e)
    
    y2 = n.execute(x2)
    print graph_delta_values(y2,e)
    
    print "-"*10, "Brute delta values of SFA features (training/test):"
    
    y_sfa = n_sfa.execute(x)
    print comp_delta(y)
    
    y2_sfa = n_sfa.execute(x2)
    print comp_delta(y2)

test_fast_windows=True #and False
if test_fast_windows:
    print ""
    print "***********************************************************************"
    print "Testing equivalence of slow and fast mirroring sliding windows for GSFA"
    x = numpy.random.normal(size=(200,15))
    
    training_modes = ("window3", "fwindow3", "smirror_window3", "mirror_window3", "truncate_window3", "ftruncate_window3")
#    training_modes = ("truncate_window3", "ftruncate_window3",)
#    training_modes = ("smirror_window32", "mirror_window32") #Test passed
    
    delta_values = []
    for training_mode in training_modes:
        n = GSFANode(output_dim=5)
        n.train(x, train_mode=training_mode)
        n.stop_training()
        
        y = n.execute(x)
        delta = comp_delta(y)
        print "**Brute Delta Values of mode %s are: "%(training_mode), delta
        delta_values.append(delta)
    
    print delta_values
    quit()

test_pathological_outputs=True #and False
if test_pathological_outputs:
    print ""
    print "**************************************************************************"
    print "*Pathological responses. Experiment on graph with weakly connected samples"
    x = numpy.random.normal(size=(20,19))
    x2 = numpy.random.normal(size=(20,19))
    
    l = numpy.random.normal(size=(20))
    l -= l.mean()
    l /= l.std()
    l.sort()

    half_width=3 
    
    v = numpy.ones(20)
    e = {}
    for t in range(19):
        e[(t,t+1)] = 1.0
    experiment = 11
    if experiment==0:
        exp_title = "Original linear SFA graph"
    elif experiment==1:
        v[0] = 10.0
        v[10] = 0.1
        v[19] = 10.0
        exp_title = "Modified node weights 1"
    elif experiment==2:
        v[0] = 10.0
        v[19] = 0.1
        exp_title = "Modified node weights 2"
    elif experiment==3:
        e[(0,1)] = 0.1
        e[(18,19)] = 10.0
        exp_title = "Modified edge weights 3"
    elif experiment==4:
        e[(10,11)] = 0.1
        e[(1,2)] = 0.1
        exp_title = "Modified edge weights 4"
    elif experiment==4.5:
        e[(10,11)] = 0.0
        e[(1,2)] = 0.0
        e[(3,5)] = 1.0
        e[(7,9)] = 1.0
        e[(17,19)] = 1.0
        e[(14,16)] = 1.0
    
        exp_title = "Modified edge weights 4.5"
    elif experiment==5:
        e[(6,7)] = 0.1
        e[(5,6)] = 0.1
        exp_title = "Modified edge weights 5"
    elif experiment==6:
        e = {}
        for j1 in range(19):
            for j2 in range(j1+1, 20):
                e[(j1,j2)] = 1/(l[j2]-l[j1]+0.00005)
        exp_title = "Modified edge weights for labels as w12 = 1/(l2-l1+0.00005) 6"
    elif experiment==7:
        e = {}
        for j1 in range(19):
            for j2 in range(j1+1, 20):
                e[(j1,j2)] = numpy.exp(-0.25*(l[j2]-l[j1])**2)
        exp_title = "Modified edge weights for labels as w12 = exp(-0.25*(l2-l1)**2) 7"
    elif experiment==8:
        e = {}
        for j1 in range(19):
            for j2 in range(j1+1, 20):
                if l[j2]-l[j1] < 0.6:
                    e[(j1,j2)] = 1/(l[j2]-l[j1]+0.00005)
        exp_title = "Modified edge weights w12 = 1/(l2-l1+0.00005), for l2-l1<0.6 8"
    elif experiment==9:
        exp_title = "Mirroring training graph, w=%d"%half_width
        train_mode = "smirror_window%d"%half_width
        e = {}
    elif experiment==10:
        exp_title = "Node weight adjustment training graph, w=%d"%half_width
        train_mode = "window%d"%half_width
        e = {}
    elif experiment==11:
        exp_title = "Truncating training graph, w=%d"%half_width
        train_mode = "truncate_window%d"%half_width
        e = {}
    else:
        print "Unknown experiment"
        quit()
    
    n = GSFANode(output_dim=5)
    if experiment in (9,10,11):
        n.train(x, train_mode=train_mode)
    else:
        n.train(x, train_mode="graph", node_weights=v, edge_weights=e)
    n.stop_training()

    print "/"*20, "Brute delta values of GSFA features (training/test):"
    y = n.execute(x)
    y2 = n.execute(x2)
    if e != {}:
        print graph_delta_values(y,e)    
        print graph_delta_values(y2,e)
    
    D = numpy.zeros(20)
    for (j1,j2) in e:
        D[j1] += e[(j1,j2)]/2.0 
        D[j2] += e[(j1,j2)]/2.0 
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    f1 = plt.figure()
    plt.title("Overfitted outputs on training data,v="+str(v))
    plt.xlabel(exp_title + "\n With D=" + str(D))
    plt.xticks(numpy.arange(0, 20, 1))
    plt.plot(y)
    if experiment in (6,7,8):
        if y[0,0] > 0:
            l *= -1
        plt.plot(l, "*")
    plt.show()
#    f1a11 = plt.subplot(2,5,1)
#    f1a11.set_position([0.05, 0.5, 0.04, 0.4])
#    plt.title("Layer Sel.")


#SUMMARY:
#Mirroring windows work fine in slow and fast versions
#Truncating window does not work in optimized version, I need to clarify the algebraic optimization
#Plain Sliding window (node-weight adjusting) seems to be broken in optimized version. For simplicity would be desirable
#Is it worth it to have so many methods? I guess the mirroring windows are enough, they have constant node weights and edge weights almost fulfill consistency

#TESTS:
#All training modes should return something reasonable.
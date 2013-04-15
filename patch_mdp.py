import numpy
import mdp
from mdp import numx
from mdp.utils import (mult, pinv, symeig, CovarianceMatrix, SymeigException)
import more_nodes
from sfa_libs import select_rows_from_matrix
import inversion
#import mdp.parallel.makeparallel
import sys
import time
import object_cache as misc
import inspect


mdp.nodes.RandomizedMaskNode = more_nodes.RandomizedMaskNode
mdp.nodes.GeneralExpansionNode = more_nodes.GeneralExpansionNode
mdp.nodes.PointwiseFunctionNode = more_nodes.PointwiseFunctionNode
mdp.nodes.PInvSwitchboard = more_nodes.PInvSwitchboard 
mdp.nodes.RandomPermutationNode = more_nodes.RandomPermutationNode
mdp.nodes.HeadNode = more_nodes.HeadNode
mdp.nodes.SFAPCANode = more_nodes.SFAPCANode
mdp.nodes.IEVMNode = more_nodes.IEVMNode
mdp.nodes.IEVMLRecNode = more_nodes.IEVMLRecNode
mdp.nodes.SFAAdaptiveNLNode = more_nodes.SFAAdaptiveNLNode

#print "Adding localized inverse support..."
mdp.Flow.localized_inverse = inversion.localized_inverse

mdp.nodes.SFANode.localized_inverse = inversion.linear_localized_inverse
mdp.nodes.PCANode.localized_inverse = inversion.linear_localized_inverse
mdp.nodes.WhiteningNode.localized_inverse = inversion.linear_localized_inverse
#RandomizedtationNode.localized_inverse = lambda self, x, y, y_prime: self.inverse(y_prime)

mdp.hinet.Layer.localized_inverse = inversion.layer_localized_inverse

mdp.nodes.GeneralExpansionNode.localized_inverse = inversion.general_expansion_node_localized_inverse 

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

        weighted_sum_diffs = weighted_diffs.sum(axis=0)
        weighted_sum_prod_diffs = mdp.utils.mult(diffs.T, weighted_diffs)
        self.AddDiffs(weighted_sum_prod_diffs, weighted_num_diffs)

    #Note, this method makes sense from consistency constraints for "larger" windows. 
    def updateMirroringSlidingWindow(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth #window_halfwidth is way too long to write it

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
        sum_diffs = numpy.zeros(dim)
        self.AddDiffs(sum_prod_diffs_full, num_diffs, weight)        
    
    #Note, this method makes sense from consistency constraints for "larger" windows. 
    def updateSlowMirroringSlidingWindow(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth #window_halfwidth is way too long to write it
      
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
                                 
                sum_diffs = diffs.sum(axis=0) 
                sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
                num_diffs = len(diffs)
# WARNING!
#                self.AddDiffs(sum_prod_diffs, sum_diffs, num_diffs, weight * window[width+offset])
                self.AddDiffs(sum_prod_diffs, sum_diffs, num_diffs, weight)               
    
    def updateFastSlidingWindow(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth

#MOST CORRECT VERSION
        x_sel = x+0.0
        w_up = numpy.arange(width, 2*width) / (2.0*width)
        w_up = w_up.reshape((width, 1))
        w_down = numpy.arange(2*width-1,width-1,-1) / (2.0*width)
        w_down = w_down.reshape((width, 1))
        x_sel[0:width, :] =  x_sel[0:width, :] * w_up
        x_sel[-width:, :] =  x_sel[-width:, :] * w_down
        
        sum_x = x_sel.sum(axis=0) 
        sum_prod_x = mdp.utils.mult(x_sel.T, x_sel)       #There is also a bug here!!!! fix me!!!!
        self.AddSamples(sum_prod_x, sum_x, num_samples - (0.5 *window_halfwidth-0.5), weight)
        
        #Update DCov Matrix
        #First we compute the borders.
        #Left border
        for i in range(0, width): # [0, width -1]
            diffs = x[0:width+i+1,:]-x[i,:]
            sum_diffs = diffs.sum(axis=0) 
            sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
            num_diffs = len(diffs) - 1 # removed zero differences
#            print "N1=", num_diffs
#            print "sum_prod_diffs[0]=", sum_prod_diffs[0]
            self.AddDiffs(sum_prod_diffs, num_diffs, weight)
        #Right border
        for i in range(num_samples-width, num_samples): # [num_samples-width, num_samples-1]
            diffs = x[i-width:num_samples,:]-x[i,:]                                 
            sum_diffs = diffs.sum(axis=0) 
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
        print "F:)",
        sum_prod_x = mdp.utils.mult(x_sel.T, x)       #Bug fixed!!! computing w * X^T * X, with X=(x1,..xN)^T
        self.AddSamples(sum_prod_x, sum_x, num_samples - (0.5 *window_halfwidth-0.5), weight)
        
        #Update DCov Matrix
        window = numpy.ones(2*width+1) # Rectangular window
        
        diffs = numpy.zeros((num_samples-2*width, dim))
        #print "width=", width
        for offset in range(-width, width+1):
            if offset == 0:
                pass
            else:
                if offset > 0:
                    diffs = x[offset:,:]-x[0:num_samples-offset,:]
                else: #offset < 0, only makes sense if window asymetric! (be careful about mean, however)
                    abs_offset = -1 * offset
                    diffs = x[0:num_samples-abs_offset,:] - x[abs_offset:,:]
                                 
                sum_diffs = diffs.sum(axis=0) 
                sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
                num_diffs = len(diffs)
# WARNING!
#                self.AddDiffs(sum_prod_diffs, sum_diffs, num_diffs, weight * window[width+offset])
                self.AddDiffs(sum_prod_diffs, num_diffs, weight)

# ERRONEOUS VERSION USED FOR PERFORMANCE REPORT:
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
    


def SFANode_inverse(self, y):
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

#print "Rewritting SFANode._inverse, __init__, _train, _stop_training..."
mdp.nodes.SFANode._inverse = SFANode_inverse

# TODO: Remove block_size=None, train_mode=None  
def SFANode__init__(self, input_dim=None, output_dim=None, dtype=None, block_size=None, train_mode=None, sfa_expo=None, pca_expo=None, magnitude_sfa_biasing=None):
    super(mdp.nodes.SFANode, self).__init__(input_dim, output_dim, dtype)
    #Warning! bias activated, "courtesy" of Alberto
    # init two covariance matrices
    # one for the input data
    self._cov_mtx = CovarianceMatrix(dtype, bias=True)
    # one for the derivatives
    self._dcov_mtx = CovarianceMatrix(dtype, bias=True)

    self.pinv = None
    self._symeig = symeig
    self.block_size= block_size
    self.train_mode = train_mode

    self.sum_prod_x = None
    self.sum_x = None
    self.num_samples = 0
    self.sum_diff = None 
    self.sum_prod_diff = None  
    self.num_diffs = 0
    
    self.sfa_expo=sfa_expo # 1.2: 1=Regular SFA directions, >1: schrink derivative in the direction of the principal components
    self.pca_expo=pca_expo # 0.25: 0=No magnitude reduction 1=Whitening
    self.magnitude_sfa_biasing = magnitude_sfa_biasing
    self._myvar = None
    self._covdcovmtx = CovDCovMatrix(block_size)
    
mdp.nodes.SFANode.__init__ = SFANode__init__
               
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

def SFANode_train_scheduler(self, x, block_size=None, train_mode = None, node_weights=None, edge_weights=None, scheduler = None, n_parallel=None):      
    self._train_phase_started = True
    if train_mode == None:
        train_mode = self.train_mode
    if block_size == None:
        block_size = self.block_size
    if scheduler == None or n_parallel == None or train_mode == None:
#        print "NO parallel sfa done...  scheduler=", ,uler, " n_parallel=", n_parallel
        return SFANode_train(self, x, block_size=block_size, train_mode=train_mode, node_weights=node_weights, edge_weights=edge_weights)
    else:
#        self._covdcovmtx.update_clustered_homogeneous_block_sizes(x, weight=1.0)

#        chunk_size=None 
#        num_chunks = n_parallel
        num_chunks = min(n_parallel, x.shape[0]/block_size)
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
                scheduler.add_task((x[i*block_size*chunk_size:(i+1)*block_size*chunk_size], block_size, 1.0), ComputeCovDcovMatrixClustered)
                print "Done Adding scheduler task ///////////////////"
                sys.stdout.flush()
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

               
def SFANode_train(self, x, block_size=None, train_mode = None, node_weights=None, edge_weights=None):
    if train_mode == None:
        train_mode = self.train_mode
    if block_size == None:
        er="SFA no block_size"
        raise Exception(er)
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
    elif train_mode[0:6] == 'window':
        window_halfwidth = int(train_mode[6:])
        print "Window (%d)"%window_halfwidth
        self._covdcovmtx.updateSlidingWindow(x, weight=1.0, window_halfwidth=window_halfwidth)
    elif train_mode[0:7] == 'fwindow':
        window_halfwidth = int(train_mode[7:])
        print "Fast Window (%d)"%window_halfwidth
        self._covdcovmtx.updateFastSlidingWindow(x, weight=1.0, window_halfwidth=window_halfwidth)
    elif train_mode[0:13] == 'mirror_window':
        window_halfwidth = int(train_mode[13:])
        print "Mirroring Window (%d)"%window_halfwidth
        self._covdcovmtx.updateMirroringSlidingWindow(x, weight=1.0, window_halfwidth=window_halfwidth)
    elif train_mode[0:14] == 'smirror_window':
        window_halfwidth = int(train_mode[14:])
        print "Slow Mirroring Window (%d)"%window_halfwidth
        self._covdcovmtx.updateSlowMirroringSlidingWindow(x, weight=1.0, window_halfwidth=window_halfwidth)
    elif train_mode == 'graph':
        print "updateGraph"
        self._covdcovmtx.updateGraph(x, weight=1.0, node_weights=node_weights, edge_weights=edge_weights)
    else:
        ex = "Unknown training method"
        raise Exception(ex)

#mdp.nodes.SFANode._train = SFANode_train
#UPDATE WARNING, this should be mdp.nodes.SFANode._train not mdp.nodes.SFANode.train
mdp.nodes.SFANode.train = SFANode_train_scheduler



def SFANode_stop_training(self, debug=False, verbose=False, pca_term = 0.995, pca_exp=2.0):
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
        if self.sfa_expo != None and self.pca_expo!=None:
            self.d, self.sf = _symeig_fake_regularized(self.dcov_mtx, self.cov_mtx, range=rng, overwrite=(not debug), sfa_expo=self.sfa_expo, pca_expo=self.pca_expo, magnitude_sfa_biasing=self.magnitude_sfa_biasing)
        else:
            self.d, self.sf = self._symeig(self.dcov_mtx, self.cov_mtx, range=rng, overwrite=(not debug))
        d = self.d
        # check that we get only *positive* eigenvalues
        if d.min() < 0:
            raise SymeigException("Got negative eigenvalues: %s." % str(d))
    except SymeigException, exception:
        errstr = str(exception)+"\n Covariance matrices may be singular."
        raise Exception(errstr)

    # delete covariance matrix if no exception occurred
    del self.cov_mtx
    del self.dcov_mtx
    del self._covdcovmtx
    # store bias
    self._bias = mult(self.avg, self.sf)
    print "shape of SFANode.sf is=", self.sf.shape
    
mdp.nodes.SFANode._stop_training = SFANode_stop_training

#sfa_train_params = ["block_size", "training_mode", "include_last"]
#mdp.nodes.SFANode.train_params = sfa_train_params

#mdp.nodes.SFANode.list_train_params = ["scheduler", "n_parallel", "train_mode", "include_latest", "block_size"]
mdp.nodes.SFANode.list_train_params = ["scheduler", "n_parallel", "train_mode", "block_size"] # "sfa_expo", "pca_expo", "magnitude_sfa_biasing"
mdp.nodes.SFAPCANode.list_train_params = ["scheduler", "n_parallel", "train_mode", "block_size"] # "sfa_expo", "pca_expo", "magnitude_sfa_biasing"
mdp.nodes.PCANode.list_train_params = ["scheduler", "n_parallel"]
mdp.nodes.IEVMNode.list_train_params = ["scheduler", "n_parallel", "train_mode", "block_size"]
mdp.nodes.IEVMLRecNode.list_train_params = ["scheduler", "n_parallel", "train_mode", "block_size"]
mdp.nodes.SFAAdaptiveNLNode.list_train_params = ["scheduler", "n_parallel", "train_mode", "block_size"]
#This function replaces the tradicional Node.train
#It extracts the relevant parameters from params according to Node.list_train_params (if available)
#and passes it to train
def node_train_params(self, data, params=None):
    all_params = {}
    if isinstance(self, (mdp.hinet.Layer, mdp.hinet.CloneLayer)):
        er = "Should never reach this point, we do specify here the train_params method for Layers..."
        raise Exception(er)
    elif "list_train_params" in dir(self):
        #list_train_params = self.list_train_params
        for par, val in params.items():
            if par in self.list_train_params:
                all_params[par] = val
        #print "parameters given to node:", all_params
        print "NODE: ", self
        print "with args: ", inspect.getargspec(self.train)
        #print "and doc:", self.train.__doc__
        return self.train(data, **all_params)
    else:
        if isinstance(self, mdp.nodes.SFANode):
            print "wrong wrong...", dir(self)
            quit()
        print self
        print "wrong wrong 2...", dir(self)
        print "data_tp=", data
 #       quit()
        return self.train(data) #no Node.train_params found

#Patch all nodes to offer train_params
#Additionally, list_train_params should be added to relevant nodes
for node_str in dir(mdp.nodes):
    node_class = getattr(mdp.nodes, node_str)
    if inspect.isclass(node_class) and issubclass(node_class, mdp.Node):
# This assumes that is_trainable() is a class function, however this might be false in some cases
#        if node_class.is_trainable():
        print "Adding train_params to mdp.Node: ", node_str
        node_class.train_params = node_train_params
#Do the same with nodes contained in hinet
for node_str in dir(mdp.hinet):
    node_class = getattr(mdp.hinet, node_str)
    if inspect.isclass(node_class) and issubclass(node_class, mdp.Node):
        if not issubclass(node_class, (mdp.hinet.Layer, mdp.hinet.CloneLayer)):
# This assumes that is_trainable() is a class function, however this might be false in some cases
#        if node_class.is_trainable():
            print "Adding train_params to (hinet) mdp.Node: ", node_str
            node_class.train_params = node_train_params
#quit()

#Execute preserving structure of input (data vector) (and in the future passing exec_params) 
#Note: exec_params not yet supported
def node_execute_data_vec(self, data_vec, exec_params=None):
    
    if isinstance(data_vec, numx.ndarray):
        return self.execute(data_vec) # **exec_params)

    res = []
    print "data_vect is:", data_vec
    for i, data in enumerate(data_vec):
        #print "data=", data
        print "data.shape=", data.shape
        #print "self=", self
        res.append(self.execute(data)) # **exec_params)
    return res

#Patch all nodes to offer execute_data_vec
for node_str in dir(mdp.nodes):
    node_class = getattr(mdp.nodes, node_str)
    print node_class 
    if inspect.isclass(node_class) and issubclass(node_class, mdp.Node):
        print "Adding execute_data_vec to node:", node_str
        node_class.execute_data_vec = node_execute_data_vec
    else:
        print "Not a node:", node_str

for node_str in dir(mdp.hinet):
    node_class = getattr(mdp.hinet, node_str)
    print node_class 
    if inspect.isclass(node_class) and issubclass(node_class, mdp.Node):
        print "Adding execute_data_vec to (hinet) node:", node_str
        node_class.execute_data_vec = node_execute_data_vec
    else:
        print "Not a node:", node_str

def ParallelSFANode_join(self, forked_node):
    print "_joining ParallelSFANode"
    """Combine the covariance matrices."""
    print "old _myvar=", self._myvar
    if self._myvar is None:
        self._myvar = forked_node._myvar
    else:
        self._myvar = self._myvar + forked_node._myvar
    print "new _myvar=", self._myvar

    if self.block_size == None:      
        if self._cov_mtx._cov_mtx is None:
            self.set_dtype(forked_node._cov_mtx._dtype)
            self._cov_mtx = forked_node._cov_mtx
            self._dcov_mtx = forked_node._dcov_mtx
        else:
            self._cov_mtx._cov_mtx += forked_node._cov_mtx._cov_mtx
            self._cov_mtx._avg += forked_node._cov_mtx._avg
            self._cov_mtx._tlen += forked_node._cov_mtx._tlen
            self._dcov_mtx._cov_mtx += forked_node._dcov_mtx._cov_mtx
            self._dcov_mtx._avg += forked_node._dcov_mtx._avg
            self._dcov_mtx._tlen += forked_node._dcov_mtx._tlen
    else:
        if self._covdcovmtx is None:
            self._covdcovmtx = forked_node._covdcovmtx
        else:
            self._covdcovmtx.addCovDCovMatrix(forked_node._covdcovmtx)

mdp.parallel.ParallelSFANode._join = ParallelSFANode_join


def PCANode_train_scheduler(self, x, scheduler = None, n_parallel=None):
    if scheduler == None or n_parallel == None:
        # update the covariance matrix
        self._cov_mtx.update(x)
    else:
        num_chunks = n_parallel
        chunk_size_samples= int(numpy.ceil(x.shape[0] *1.0/ num_chunks))

        print "%d chunks, of size %d samples"%(num_chunks, chunk_size_samples)
        for i in range(num_chunks):
            scheduler.add_task(x[i*chunk_size_samples:(i+1)*chunk_size_samples], ComputeCovMatrix)
        
        print "Getting results"
        sys.stdout.flush()

        results = scheduler.get_results()
#        print "Shutting down scheduler"
        sys.stdout.flush()

        if self._cov_mtx._cov_mtx == None:
            self._cov_mtx._cov_mtx = 0.0
            self._cov_mtx._avg = 0
            self._cov_mtx._tlen = 0

        for covmtx in results:
            self._cov_mtx._cov_mtx += covmtx._cov_mtx
            self._cov_mtx._avg += covmtx._avg
            self._cov_mtx._tlen += covmtx._tlen

mdp.nodes.PCANode._train = PCANode_train_scheduler

#def apply_permutation_to_signal(x, permutation, output_dim):
#    xt = x.T
#    yt = numpy.zeros((x.shape[1], output_dim))
#    for i in permutation:
#        yt[i] = xt[i]
#    y = yt.T
#    return y

#Make switchboard faster!!!
def switchboard_new_execute(self, x):
    return select_rows_from_matrix(x, self.connections)

mdp.hinet.Switchboard._execute = switchboard_new_execute

#print "Fixing GaussianClassifierNode_class_probabilities..."


#Verbosity and fix for nan values courtesy of Alberto
def GaussianClassifierNode_class_probabilities(self, x, verbose=True):
        """Return the posterior probability of each class given the input."""
        self._pre_execution_checks(x)

        # compute the probability for each class
        tmp_prob = numpy.zeros((x.shape[0], len(self.labels)),
                              dtype=self.dtype)
        for i in range(len(self.labels)):
            tmp_prob[:, i] = self._gaussian_prob(x, i)
            tmp_prob[:, i] *= self.p[i]
            
        # normalize to probability 1
        # (not necessary, but sometimes useful)
        tmp_tot = tmp_prob.sum(axis=1)
        tmp_tot = tmp_tot[:, numpy.newaxis]

        # Warning, it can happen that tmp_tot is very small!!!

        probs  = tmp_prob/tmp_tot        
        uniform_amplitude = 1.0 / probs.shape[1]
        #smallest probability that makes sense for the problem...
        #for the true semantics reverse engineer the semantics of _gaussian_prob()
        smallest_probability = 1e-60
        counter = 0
        for i in range(probs.shape[0]):
            if numpy.isnan(probs[i]).any() or tmp_tot[i, 0] < smallest_probability:
#                if verbose:
#                    print "Problematic probs[%d]="%i, probs[i],
#                    print "Problematic tmp_prob[%d]="%i, tmp_prob[i]
#                    print "Problematic tmp_tot[%d]="%i, tmp_tot[i]                
# First attempt to fix it, use uniform_amplitude as guess
                probs[i] = uniform_amplitude
                counter += 1
# Second attempt: Find maximum, and assing 1 to it, otherwise go to previous strategy...
# Seems like it always fails with nan for all entries, thus first measure is better...
        if verbose:
            print "%d probabilities were fixed"%counter

        #just in case at this point something still failed... looks impossible though.
        probs = numpy.nan_to_num(probs)
        
        return probs
    
    
#TODO: Consider adding node GaussianClassifierWithRegressions
#TODO: Consider making the next two functions node independent
def GaussianRegression(self, data, avg_labels):
    """  Use the class probabilities to generate a better label
    If the computation of the class probabilities were perfect,
    and if the Gaussians were just a delta, then the output value
    minimizes the squared error.
    self (mdp node): mdp node providing the class_probabilities function 
    data (2D numpy array): set of samples to do regression on
    avg_labels (1D numpy array): average label for class 0, class 1, ...
    """
    probabilities = self.class_probabilities(data)
    value = numpy.dot(probabilities, avg_labels)
    value[numpy.isnan(value)] = avg_labels.mean() #TODO:compute real mean of all labels
    print "value.shape=", value.shape
    return value


def GaussianRegressionMAE(self, data, avg_labels):
    """  Use the class probabilities to generate a better label
    If the computation of the class probabilities were perfect,
    (and if the Gaussians were just a delta???), then the output value
    minimizes the mean average error
    self (mdp node): mdp node providing the class_probabilities function 
    data (2D numpy array): set of samples to do regression on
    avg_labels (1D numpy array): average label for class 0, class 1, ...
    """
    probabilities = self.class_probabilities(data)
    acc_probs = probabilities.cumsum(axis=1)
    
    print "labels=", self.labels
    print "acc_probs[0:5]", acc_probs[0:5]
    
    acc_index = numpy.ones(acc_probs.shape).cumsum(axis=1)-1
    probability_mask = (acc_probs <= 0.5)
    acc_index[probability_mask] = acc_probs.shape[1]+100 #mark entries with acc_probs <= 0.5 as a large value.
    best_match = numpy.argmin(acc_index, axis=1) #making the smallest entry the first one with acc_prob > 0.5
    
#    probability_deviation_from_mode = numpy.abs(acc_probs-0.5)
#    print "probability_deviation_from_mode[0:5]", probability_deviation_from_mode[0:5]
#    best_match = numpy.argmin(probability_deviation_from_mode, axis=1)
    print "best_match[0:5]",best_match[0:5]
    #print type(best_match)
    #print type(self.labels)
    value = avg_labels[best_match]
    #print "valueMAE.shape=", value.shape
    return value

def GaussianRegressionMAE_uniform_bars(self, data, avg_labels):
    """  Use the class probabilities to generate a better label
    If the computation of the class probabilities were perfect,
    (and if the Gaussians were just a delta???), then the output value
    minimizes the mean average error
    self (mdp node): mdp node providing the class_probabilities function 
    data (2D numpy array): set of samples to do regression on
    avg_labels (1D numpy array): average label for class 0, class 1, ...
    """
    probabilities = self.class_probabilities(data)
    acc_probs = probabilities.cumsum(axis=1)
    
    print "labels (classes)=", self.labels
    print "acc_probs[0:5]", acc_probs[0:5]
    
    acc_index = numpy.ones(acc_probs.shape).cumsum(axis=1)-1
    probability_mask = (acc_probs <= 0.5)
    acc_index[probability_mask] = acc_probs.shape[1]+100 #mark entries with acc_probs <= 0.5 as a large value.
    best_match = numpy.argmin(acc_index, axis=1) #making the smallest entry the first one with acc_prob > 0.5

    print "best_match=", best_match, 
    #Compute x0 and x1, assume symetric bars... do this vectorially
    bar_limit_x0 = numpy.zeros(len(avg_labels))
    bar_limit_x0[1:] =  (avg_labels[1:]+avg_labels[0:-1])/2.0
    bar_limit_x0[0] = avg_labels[0]-(avg_labels[1]-avg_labels[0])/2.0
    
    bar_limit_x1 = numpy.zeros(len(avg_labels))
    bar_limit_x1[:-1] =  (avg_labels[:-1]+avg_labels[1:])/2.0
    bar_limit_x1[-1] = avg_labels[-1]+(avg_labels[-1]-avg_labels[-2])/2.0

    yy = numpy.arange(len(data))
    CP_x0 = acc_probs[yy, best_match-1]
    mask = (best_match==0)
    CP_x0[mask] = 0.0  
    CP_x1 = acc_probs[yy,best_match]

    x0 = bar_limit_x0[best_match]
    x1 = bar_limit_x1[best_match]

    print "x0.shape=", x0.shape, "x1.shape=", x1.shape, "CP_x0.shape=", CP_x0.shape, "CP_x1.shape=", CP_x1.shape  
    value = x0 + (0.5 - CP_x0)*(x1-x0)/(CP_x1-CP_x0)
    print "MAE value=", value
#    probability_deviation_from_mode = numpy.abs(acc_probs-0.5)
#    print "probability_deviation_from_mode[0:5]", probability_deviation_from_mode[0:5]
#    best_match = numpy.argmin(probability_deviation_from_mode, axis=1)
#    print "best_match[0:5]",best_match[0:5]
    #print type(best_match)
    #print type(self.labels)
#    value = avg_labels[best_match]
    #print "valueMAE.shape=", value.shape
    return value


#UPDATE WARNING: Do we need this correction?
#mdp.nodes.GaussianClassifierNode.class_probabilities = GaussianClassifierNode_class_probabilities
mdp.nodes.GaussianClassifier.regression = GaussianRegression
#Original: mdp.nodes.GaussianClassifier.regressionMAE = GaussianRegressionMAE
#Experimental: 
mdp.nodes.GaussianClassifier.regressionMAE = GaussianRegressionMAE #GaussianRegressionMAE_uniform_bars

#def switchboard_execute(self, x):
#    return apply_permutation_to_signal(x, self.connections, self.output_dim)
#
#mdp.hinet.Switchboard._execute = switchboard_execute

#print "Adding linear_localized_inverse to RandomPermutationNode..."

def RandomPermutationNode_linear_localized_inverse(self, x, y, y_prime):
    return self.inverse(y_prime)        

more_nodes.RandomPermutationNode.linear_localized_inverse = RandomPermutationNode_linear_localized_inverse


#This is not needed, as long as execute is executed for setting such dim sizes
#print "Making IdentityNode trainable (to remember input/output dim sizes)"
#def IdentityNode_is_trainable(self):
#    return True
#
#def IdentityNode_train(self, x):
#    self.output_dim = x.shape[1]
#    
#mdp.nodes.IdentityNode.is_trainable = IdentityNode_is_trainable
#mdp.nodes.IdentityNode._train = IdentityNode_train

#PATCH for layer.py
patch_layer = True
if patch_layer:
    def Layer_new__init__(self, nodes, dtype=None, homogeneous=False):
        """Setup the layer with the given list of nodes.
        
        The input and output dimensions for the nodes must be already set 
        (the output dimensions for simplicity reasons). The training phases for 
        the nodes are allowed to differ.
        
        Keyword arguments:
        nodes -- List of the nodes to be used.
        """
        self.nodes = nodes
#WARNING
        self.homogeneous=homogeneous
        # check nodes properties and get the dtype
        dtype = self._check_props(dtype)
        # calculate the the dimensions
        self.node_input_dims = numx.zeros(len(self.nodes))
        #WARNING: Difference between "==" and "is"???
        if self.homogeneous == False:
            input_dim = 0
            for index, node in enumerate(nodes):
                input_dim += node.input_dim
                self.node_input_dims[index] = node.input_dim
        else:
            input_dim = None
            for index, node in enumerate(nodes):
                self.node_input_dims[index] = None
            
        output_dim = self._get_output_dim_from_nodes()
        super(mdp.hinet.Layer, self).__init__(input_dim=input_dim,
                                    output_dim=output_dim,
                                    dtype=dtype)
        
    def Layer_new_check_props(self, dtype=None):
        """Check the compatibility of the properties of the internal nodes.
        
        Return the found dtype and check the dimensions.
        
        dtype -- The specified layer dtype.
        """
        dtype_list = []  # the dtypes for all the nodes
        for i, node in enumerate(self.nodes):
            # input_dim for each node must be set
#WARNING!!!
            if self.homogeneous is False:
                if node.input_dim is None:
                    err = ("input_dim must be set for every node. " +
                           "Node #%d (%s) does not comply." % (i, node))
                    raise mdp.NodeException(err)

            if node.dtype is not None:
                dtype_list.append(node.dtype)
        # check that the dtype is None or the same for every node
        nodes_dtype = None
        nodes_dtypes = set(dtype_list)
        nodes_dtypes.discard(None)
        if len(nodes_dtypes) > 1:
            err = ("All nodes must have the same dtype (found: %s)." % 
                   nodes_dtypes)
            raise mdp.NodeException(err)
        elif len(nodes_dtypes) == 1:
            nodes_dtype = list(nodes_dtypes)[0]
        else:
            nodes_dtype = None
#            er = "Error, no dtypes found!!!"
#            raise Exception(er)

        # check that the nodes dtype matches the specified dtype
        if nodes_dtype and dtype:
            if not numx.dtype(nodes_dtype) == numx.dtype(dtype):
                err = ("Cannot set dtype to %s: " %
                       numx.dtype(nodes_dtype).name +
                       "an internal node requires %s" % numx.dtype(dtype).name)
                raise mdp.NodeException(err)
        elif nodes_dtype and dtype == None:
            dtype = nodes_dtype
        return dtype
    
    def Layer_new_train(self, x, scheduler=None, n_parallel=0, *args, **kwargs):
        """Perform single training step by training the internal nodes."""
        start_index = 0
        stop_index = 0

        if self.homogeneous is True:
            layer_input_dim = x.shape[1]
            self.set_input_dim(layer_input_dim)
            num_nodes = len(self.nodes)
            print "Training homogeneous layer with input_dim %d and %d nodes"%(layer_input_dim, num_nodes)
            for node in self.nodes:
                node.set_input_dim(layer_input_dim / num_nodes)
            input_dim = 0
            for index, node in enumerate(self.nodes):
                input_dim += node.input_dim
                self.node_input_dims[index] = node.input_dim
 #           print "input dim is: %d, should be %d"%(input_dim, self.input_dim)


        for node in self.nodes:
            start_index = stop_index
            stop_index += node.input_dim
#           print "stop_index = ", stop_index
            if node.is_training():
                if isinstance(node, (mdp.nodes.SFANode, mdp.nodes.PCANode, mdp.nodes.WhiteningNode, mdp.hinet.CloneLayer, mdp.hinet.Layer)) and node.input_dim >= 45:
                    print "Attempting node parallel training in Layer..."
                    node.train(x[:, start_index : stop_index], scheduler=scheduler, n_parallel=n_parallel, *args, **kwargs)
                else:
                    node.train(x[:, start_index : stop_index], *args, **kwargs)




    def Layer_new_train_params(self, x, params = None):
        """Perform single training step by training the internal nodes."""
        start_index = 0
        stop_index = 0

        if self.homogeneous is True:
            layer_input_dim = x.shape[1]
            self.set_input_dim(layer_input_dim)
            num_nodes = len(self.nodes)
            print "Training homogeneous layer with input_dim %d and %d nodes"%(layer_input_dim, num_nodes)
            for node in self.nodes:
                node.set_input_dim(layer_input_dim / num_nodes)
            input_dim = 0
            for index, node in enumerate(self.nodes):
                input_dim += node.input_dim
                self.node_input_dims[index] = node.input_dim
#           print "input dim is: %d, should be %d"%(input_dim, self.input_dim)

        for node in self.nodes:
            start_index = stop_index
            stop_index += node.input_dim
#           print "stop_index = ", stop_index
            if node.is_training():
                node.train_params(x[:, start_index : stop_index], params)
#                if isinstance(node, (mdp.nodes.SFANode, mdp.nodes.PCANode, mdp.nodes.WhiteningNode, mdp.hinet.CloneLayer, mdp.hinet.Layer)) and node.input_dim >= 45:
#                    print "Attempting node parallel training in Layer..."
#                    node.train(x[:, start_index : stop_index], scheduler=scheduler, n_parallel=n_parallel, *args, **kwargs)
#                else:
#                    node.train(x[:, start_index : stop_index], *args, **kwargs)


    def Layer_new_pre_execution_checks(self, x):
        """Make sure that output_dim is set and then perform normal checks."""
        if self.output_dim is None:
            # first make sure that the output_dim is set for all nodes
            in_start = 0
            in_stop = 0
 
            
            #Warning!!! add code to support homogeneous input sizes
            if self.homogeneous is True:
                layer_input_dim = x.shape[1]
                self.set_input_dim(layer_input_dim)
                num_nodes = len(self.nodes)
                print "Pre_Execution of homogeneous layer with input_dim %d and %d nodes"%(layer_input_dim, num_nodes)
                for node in self.nodes:
                    node.set_input_dim(layer_input_dim / num_nodes)
                    input_dim = 0
                    for index, node in enumerate(self.nodes):
                        input_dim += node.input_dim
                        self.node_input_dims[index] = node.input_dim
#                    print "input dim is: %d, should be %d"%(input_dim, self.input_dim)
            
            
            for node in self.nodes:
                in_start = in_stop
                in_stop += node.input_dim
                node._pre_execution_checks(x[:,in_start:in_stop])
            self.output_dim = self._get_output_dim_from_nodes()
            if self.output_dim is None:
                err = "output_dim must be set at this point for all nodes"
                raise mdp.NodeException(err)  
        super(mdp.hinet.Layer, self)._pre_execution_checks(x)


    def CloneLayer_new__init__(self, node, n_nodes=1, dtype=None):
        """Setup the layer with the given list of nodes.
        
        Keyword arguments:
        node -- Node to be cloned.
        n_nodes -- Number of repetitions/clones of the given node.
        """
#WARNING!!!
        super(mdp.hinet.CloneLayer, self).__init__((node,) * n_nodes, dtype=dtype, homogeneous=True)
        self.node = node  # attribute for convenience
        
        

    mdp.hinet.Layer.__init__ = Layer_new__init__
    mdp.hinet.Layer._check_props = Layer_new_check_props

    mdp.hinet.Layer._train = Layer_new_train_params
#ATTENTION, modification for backwards compatibility!!!!!!!
#    mdp.hinet.Layer._train = Layer_new_train

#    mdp.hinet.Layer.train_params = Layer_new_train_params
#    mdp.hinet.Layer.train_scheduler = Layer_new_train
    mdp.hinet.Layer._pre_execution_checks = Layer_new_pre_execution_checks      
    mdp.hinet.CloneLayer.__init__ = CloneLayer_new__init__
    #print "mdp.Layer was patched"

    
def HiNetParallelTranslator_translate_layer(self, layer):
    """Replace a Layer with its parallel version."""
    parallel_nodes = super(mdp.parallel.makeparallel.HiNetParallelTranslator, 
                           self)._translate_layer(layer)
#Warning, it was: return parallelhinet.ParallelLayer(parallel_nodes)
    return mdp.parallel.ParallelLayer(parallel_nodes, homogeneous=layer.homogeneous)
    
#UPDATE WARNING: Is this still needed?
#mdp.parallel.makeparallel.HiNetParallelTranslator._translate_layer = HiNetParallelTranslator_translate_layer



#LINEAR_FLOW FUNCTIONS


#Code courtesy of Alberto Escalante
#improves the training time in a linear factor on the number of nodes
#but is less general than the usual procedure
#Now supporting list based training data for different layers
#not needed now: signal_read_enabled=False, signal_write_enabled=False
patch_flow = True
if patch_flow:
    def flow_special_train_cache_scheduler(self, data, verbose=False, benchmark=None, node_cache_read = None, signal_cache_read=None, node_cache_write=None, signal_cache_write=None, scheduler=None, n_parallel=None):
        # train each Node successively
        min_input_size_for_parallel = 45

        print data.__class__, data.dtype, data.shape
        print data

        data_loaded = True
        for i in range(len(self.flow)):
            trained_node_in_cache = False
            exec_signal_in_cache = False
                
            if benchmark != None:
                ttrain0 = time.time()
            if self.verbose:
                print "*****************************************************************"
                print "Training node #%d (%s)..." % (i, str(self.flow[i])),
                if isinstance(self.flow[i], mdp.hinet.Layer):
                    print "of [%s]"%str(self.flow[i].nodes[0])
                elif isinstance(self.flow[i], mdp.hinet.CloneLayer):
                    print "of cloned [%s]"%str(self.flow[i].nodes[0])
                
            hash_verbose=False
            if str(self.flow[i]) == "CloneLayer":
                if str(self.flow[i].nodes[0]) == "RandomPermutationNode":
                    print "BINGO, RandomPermutationNode Found!"
                    hash_verbose=False
                
            if node_cache_write or node_cache_read or signal_cache_write or signal_cache_read:
                untrained_node_hash = misc.hash_object(self.flow[i], verbose=hash_verbose).hexdigest()
                print "untrained_node_hash[%d]=" % i, untrained_node_hash
                node_ndim = str(data.shape[1])
                data_in_hash = misc.hash_object(data).hexdigest()
                print "data_in_hash[%d]=" % i, data_in_hash
    
    
                    
            if self.flow[i].is_trainable():
                #look for trained node in cache
                if node_cache_read:
                    node_base_filename = "node_%s_%s_%s"%((node_ndim, untrained_node_hash, data_in_hash))
                    print "Searching for Trained Node:", node_base_filename
                    if node_cache_read.is_file_in_filesystem(base_filename=node_base_filename):
                        print "Trained node FOUND in cache..."
                        self.flow[i] = node_cache_read.load_obj_from_cache(base_filename=node_base_filename)
                        trained_node_in_cache = True
                    else:
                        print "Trained node NOT found in cache..."

                if trained_node_in_cache == False:
                    #Not in cache, then train the node
                    if isinstance(self.flow[i], (mdp.hinet.CloneLayer, mdp.hinet.Layer)):
    #                       print "Here it should be doing parallel training 1!!!"
#WARNING: REMOVED scheduler & n_parallel
#                        self.flow[i].train(data, scheduler=scheduler, n_parallel=n_parallel)
                        self.flow[i].train(data, scheduler=scheduler, n_parallel=n_parallel)
                    elif isinstance(self.flow[i], (mdp.nodes.SFANode, mdp.nodes.PCANode, mdp.nodes.WhiteningNode)) and \
                    self.flow[i].input_dim >= min_input_size_for_parallel:
    #                       print "Here it should be doing parallel training 2!!!"
#WARNING: REMOVED scheduler & n_parallel
                        self.flow[i].train(data, scheduler=scheduler, n_parallel=n_parallel)
                    else:
                        print "Input_dim was: ", self.flow[i].input_dim, "or unknown parallel method, thus I didn't go parallel"
                        self.flow[i].train(data)
                    self.flow[i].stop_training()
            ttrain1 = time.time()
    
            if node_cache_write or node_cache_read or signal_cache_write or signal_cache_read:
                trained_node_hash = misc.hash_object(self.flow[i],verbose=hash_verbose).hexdigest()
                print "trained_node_hash[%d]="%i, trained_node_hash
    
            print "++++++++++++++++++++++++++++++++++++ Executing..."
            if signal_cache_read:
                signal_base_filename = "signal_%s_%s_%s"%((node_ndim, data_in_hash, trained_node_hash))
                print "Searching for Executed Signal: ", signal_base_filename
                if signal_cache_read.is_splitted_file_in_filesystem(base_filename=signal_base_filename):
                    print "Executed signal FOUND in cache..."
                    data = signal_cache_read.load_array_from_cache(base_filename=signal_base_filename, verbose=True)
                    exec_signal_in_cache = True
                else:
                    print "Executed signal NOT found in cache..."
    
            print data.__class__, data.dtype, data.shape
            print "supported types:", self.flow[i].get_supported_dtypes()
            print data
            
            if exec_signal_in_cache == False:  
                data = self.flow[i].execute(data)
    
            ttrain2 = time.time()
            if verbose == True:
                print "Training finished in %0.3f s, execution in %0.3f s"% ((ttrain1-ttrain0), (ttrain2-ttrain1))          
            else:
                print "Training finished"
            if benchmark != None:
                benchmark.append(("Train node #%d (%s)" % (i, str(self.flow[i])), ttrain1-ttrain0))
                benchmark.append(("Execute node #%d (%s)" % (i, str(self.flow[i])), ttrain2-ttrain1))
                
            #Add to Cache Memory: Executed Signal
            if node_cache_write and (trained_node_in_cache == False) and self.flow[i].is_trainable():              
                print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Caching Trained Node..."
                node_base_filename = "node_%s_%s_%s"%((node_ndim, untrained_node_hash, data_in_hash))
                node_cache_write.update_cache(self.flow[i], base_filename=node_base_filename, overwrite=True, verbose=True)
    
            if signal_cache_write and (exec_signal_in_cache == False):              
                print "####################################### Caching Executed Signal..."
                data_ndim = node_ndim
                data_base_filename = "signal_%s_%s_%s"%((data_ndim, data_in_hash, trained_node_hash))
                signal_cache_write.update_cache(data, base_filename=data_base_filename, overwrite=True, verbose=True)
        return data

#The functions generate the training data when executed, while parameters are used for training the corresponding node
#Function that given a training data description (func and param sets) extracts the relevant function and parameters vector for the given node
#The catch is that funcs_sets and param_sets are (usually) bidimensional arrays. 
#The first dimension is used for the particular node
#The second dimension is used in case there is more than one training data set for the node, and mmight be None, which means that the data/parameters from the previous node is used
#The output is the data functions and parameters needed to train a particular node
#Add logic for data_params
    def extract_node_funcs(funcs_sets, param_sets, node_nr):
        print "funcs_sets is:", funcs_sets
        if isinstance(funcs_sets, list):
            #Find index of last data_vec closer to the requested node 
            if node_nr >= len(funcs_sets):
                index = len(funcs_sets)-1
            else:
                index = node_nr
    
            #Find index of last data_vec with not None data
            while funcs_sets[index] == None and index > 0:
                index -= 1
            
            node_funcs = funcs_sets[index]
            if param_sets == None: 
                node_params = [None] * len(funcs_sets[index])
                for i in len(node_params):
                    node_params[i] = {} 
            else:
                print "param_sets =", param_sets
                node_params = param_sets[index]
            
#            #TODO: More robust compatibility function required            
##            if len(node_params) != len(node_funcs):
##                er = "node_funcs and node_params are not compatible: "+str(node_funcs)+str(node_params)
##                raise Exception(er)
            print "node_funcs and node_params:", node_funcs, node_params
            return node_funcs, node_params
        else: #Not a data set, use data itself as training data
            if param_sets == None:
                param_sets = {}
            print "param_sets =", param_sets
            return funcs_sets, param_sets

#If the input is a list of functions, execute them to generate the data_vect, otherwise use node_funcs directly as array data        
    def extract_data_from_funcs(node_funcs):
        print "node_funcs is:", node_funcs
        if isinstance(node_funcs, list):
            node_data = []
            for func in node_funcs:
                node_data.append(func())
            return node_data
        elif inspect.isfunction(node_funcs):
            return node_funcs()
        else:
            return node_funcs

    #Tells the dimensionality of the data (independent of number of samples or number of data arrays)
    def data_vec_ndim(data_vec):
        if isinstance(data_vec, list):
            return data_vec[0].shape[1]
        else:
            return data_vec.shape[1]

    #As the previous train method, however here shape variable data is allowed
    #Perhaps the data should be loaded dynamically???
    #TODO: find nicer name!
    #In theory, there should  be a data_in_hash for data & another for (data, params), however we only use the last one
    def flow_special_train_cache_scheduler_sets(self, funcs_sets, params_sets=None, verbose=True, benchmark=None, node_cache_read = None, signal_cache_read=None, node_cache_write=None, signal_cache_write=None, scheduler=None, n_parallel=None, memory_save=False):
        # train each Node successively
        #Set smalles dimensionality for which parallel training is worth doing
        min_input_size_for_parallel = 45

#        print data.__class__, data.dtype, data.shape
#        print data
#        print data_params

#        if memory_save == True:
#            node_cache_read = None
#            signal_cache_read = None
#            node_cache_write = None
#            signal_cache_write = None
            
        #indicates whether node_data and node_params are valid 
        node_funcs = node_data = node_params = None
#        data_loaded = False
        for i in range(len(self.flow)):
            #indicates whether the node or the exec_signal were loaded from cache
            trained_node_in_cache = False
            exec_signal_in_cache = False

            #Extract data and data_params, integrity check
            #if data_loaded == False:
            new_node_funcs, new_node_params = extract_node_funcs(funcs_sets, params_sets, node_nr=i)
            print "new_node_funcs = ", new_node_funcs 
            #quit()"list_train_params" in dir(self)
            execute_node_data = False
            if isinstance(new_node_funcs, numpy.ndarray):
                comp = (node_funcs != new_node_funcs)
                if isinstance(comp, bool) and (node_funcs != new_node_funcs):
                    execute_node_data = True
                elif isinstance(comp, numpy.ndarray) and (node_funcs != new_node_funcs).any():
                    execute_node_data = True
            else:
                if not (node_funcs == new_node_funcs): #New data loading needed
                    execute_node_data = True

            if execute_node_data: #data should be extracted from new_node_funcs and propagated just before the current node 
                node_data = extract_data_from_funcs(new_node_funcs)
                print "node_data is:", node_data
                data_vec = self.execute_data_vec(node_data, node_nr=i-1)  
            #else:
            #    data_vec already contains valid data

            node_funcs = new_node_funcs
            node_params = new_node_params
            
            if self.flow[i].input_dim >= min_input_size_for_parallel:              
                if isinstance(data_vec, list):
                    for j in range(len(data_vec)):
                        new_node_params[j]["scheduler"] = scheduler
                        new_node_params[j]["n_parallel"] = n_parallel
                else:
                    new_node_params["scheduler"] = scheduler
                    new_node_params["n_parallel"] = n_parallel
                    
            
            if benchmark != None:
                ttrain0 = time.time()
            if self.verbose:
                print "*****************************************************************"
                print "Training node #%d (%s)..." % (i, str(self.flow[i])),
                if isinstance(self.flow[i], mdp.hinet.Layer):
                    print "of [%s]"%str(self.flow[i].nodes[0])
                elif isinstance(self.flow[i], mdp.hinet.CloneLayer):
                    print "of cloned [%s]"%str(self.flow[i].nodes[0])
                
            hash_verbose=False
            if str(self.flow[i]) == "CloneLayer":
                if str(self.flow[i].nodes[0]) == "RandomPermutationNode":
                    print "BINGO, RandomPermutationNode Found!"
                    hash_verbose=False

            #Compute hash of Node and Training Data if needed
            #Where to put/hash data_parameters?????? in data "of_course"!
            #Notice, here Data should be taken from appropriate component, and also data_params
            if node_cache_write or node_cache_read or signal_cache_write or signal_cache_read:
                untrained_node_hash = misc.hash_object(self.flow[i], verbose=hash_verbose).hexdigest()
                print "untrained_node_hash[%d]=" % i, untrained_node_hash
                node_ndim = str(data_vec_ndim(data_vec))
                #hash data and node parameters for handling the data!
                print "node_params to be hashed: ", node_params
                data_in_hash = misc.hash_object((data_vec, node_params)).hexdigest()
                print "data_in_hash[%d]=" % i, data_in_hash
                           
            if self.flow[i].is_trainable():
                #look for trained node in cache
                if node_cache_read:
                    node_base_filename = "node_%s_%s_%s"%((node_ndim, untrained_node_hash, data_in_hash))
                    print "Searching for Trained Node:", node_base_filename
                    if node_cache_read.is_file_in_filesystem(base_filename=node_base_filename):
                        print "Trained node FOUND in cache..."
                        self.flow[i] = node_cache_read.load_obj_from_cache(base_filename=node_base_filename)
                        trained_node_in_cache = True
                    else:
                        print "Trained node NOT found in cache..."
#                else:
#                    er = "What happened here???"
#                    raise Exception(er)   
                

                #If trained node not found in cache, then train!
                #Here, however, parse data and data_params appropriately!!!!!
                if trained_node_in_cache == False:
                    #Not in cache, then train the node
                    if isinstance(self.flow[i], (mdp.hinet.CloneLayer, mdp.hinet.Layer)):
                        print "First step"
                        if isinstance(data_vec, list):
                            print "First step 2"
                            for j, data in enumerate(data_vec):
                                print "j=", j
                                #Here some logic is expected (list of train parameters on each node???)
                                self.flow[i].train(data, params=node_params[j])
                        else:
                            print "First step 3"                            
                            self.flow[i].train(data_vec, params=node_params)
                            
                    elif isinstance(self.flow[i], (mdp.nodes.SFANode, mdp.nodes.PCANode, mdp.nodes.WhiteningNode)):
                        print "Second Step"
                        if isinstance(data_vec, list):                          
                            for j, data in enumerate(data_vec):
                                print "Parameters used for training node (L)=", node_params
                                print "Pre self.flow[i].output_dim=", self.flow[i].output_dim
                                self.flow[i].train_params(data, params=node_params[j])
                                print "Post self.flow[i].output_dim=", self.flow[i].output_dim
                        else:
                            print "Parameters used for training node=", node_params
                            self.flow[i].train_params(data_vec, params=node_params)
                    else: #Other node which does not have parameters nor parallelization
#                        print "Input_dim ", self.flow[i].input_dim, "<", min_input_size_for_parallel, ", or unknown parallel method, thus I didn't go parallel"
                        if isinstance(data_vec, list):                          
                            for j, data in enumerate(data_vec):
                                self.flow[i].train_params(data, params=node_params[j])
                        else:
                            self.flow[i].train_params(data_vec, params=node_params)                            
                    print "Finishing training of node %d:"%i, self.flow[i] 
                    self.flow[i].stop_training()
                    print "Post2 self.flow[i].output_dim=", self.flow[i].output_dim

            ttrain1 = time.time()
    
            #is hash of trained node needed??? of course, this is redundant if no training
            if node_cache_write or node_cache_read or signal_cache_write or signal_cache_read:
                trained_node_hash = misc.hash_object(self.flow[i],verbose=hash_verbose).hexdigest()
                print "trained_node_hash[%d]="%i, trained_node_hash

            print "++++++++++++++++++++++++++++++++++++ Executing..."
            #Look for excecuted signal in cache
            if signal_cache_read:
                signal_base_filename = "signal_%s_%s_%s"%((node_ndim, data_in_hash, trained_node_hash))
                print "Searching for Executed Signal: ", signal_base_filename
                if signal_cache_read.is_splitted_file_in_filesystem(base_filename=signal_base_filename):
                    print "Executed signal FOUND in cache..."
                    data_vec = signal_cache_read.load_array_from_cache(base_filename=signal_base_filename, verbose=True)
                    exec_signal_in_cache = True
                else:
                    print "Executed signal NOT found in cache..."
    
#            print data.__class__, data.dtype, data.shape
#            print "supported types:", self.flow[i].get_supported_dtypes()
#            print data

            #However, excecute should preserve shape here!           
            if exec_signal_in_cache == False:
                
                data_vec = self.flow[i].execute_data_vec(data_vec)
                print "Post3 self.flow[i].output_dim=", self.flow[i].output_dim
                print "data_vec", data_vec
                print "len(data_vec)", len(data_vec)
                print "data_vec[0].shape", data_vec[0].shape
                 
            ttrain2 = time.time()
            if verbose == True:
                print "Training finished in %0.3f s, execution in %0.3f s"% ((ttrain1-ttrain0), (ttrain2-ttrain1))          
            else:
                print "Training finished"
            if benchmark != None:
                benchmark.append(("Train node #%d (%s)" % (i, str(self.flow[i])), ttrain1-ttrain0))
                benchmark.append(("Execute node #%d (%s)" % (i, str(self.flow[i])), ttrain2-ttrain1))
                
            #Add to Cache Memory: Trained Node
            if node_cache_write and (trained_node_in_cache == False) and self.flow[i].is_trainable():              
                print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Caching Trained Node..."
                node_base_filename = "node_%s_%s_%s"%((node_ndim, untrained_node_hash, data_in_hash))
                node_cache_write.update_cache(self.flow[i], base_filename=node_base_filename, overwrite=True, verbose=True)

            #Add to Cache Memory: Executed Signal
            if signal_cache_write and (exec_signal_in_cache == False):
                #T: Perhaps data should be written based on mdp.array, not on whole data.
                print "####################################### Caching Executed Signal..."
                data_ndim = node_ndim
                data_base_filename = "signal_%s_%s_%s"%((data_ndim, data_in_hash, trained_node_hash))
                signal_cache_write.update_cache(data_vec, base_filename=data_base_filename, overwrite=True, verbose=True)
        if isinstance(data_vec, list):
            return numpy.concatenate(data_vec, axis=0)
        else:
            return data_vec

    #Supports mainly single array data or merges iterator data 
    def flow__execute_seq(self, x, node_nr = None, benchmark=None):
        # Filters input data 'x' through the nodes 0..'node_nr' included
        flow = self.flow
        if node_nr is None:
            node_nr = len(flow)-1
        
        if benchmark != None:
            t0 = time.time()
        for i in range(node_nr+1):  
            try:
                x = flow[i].execute(x)
            except Exception, e:
                self._propagate_exception(e, i)
            if benchmark != None:
                t1 = time.time()
                benchmark.append(("Node %d (%s)Excecution"%(i,str(flow[i])), t1-t0))
                t0 = t1
        return x

    #Supports single array data but also iterator
    #However result is concatenated! Not PowerTraining Compatible???
    #Thus this function is usually called for each single chunk 
    def flow_execute(self, iterable, node_nr = None, benchmark=None):
        """Process the data through all nodes in the flow.
            
        'iterable' is an iterable or iterator (note that a list is also an
        iterable), which returns data arrays that are used as input to the flow.
        Alternatively, one can specify one data array as input.
            
        If 'nodenr' is specified, the flow is executed only up to
        node nr. 'nodenr'. This is equivalent to 'flow[:nodenr+1](iterable)'.
        """
        #Strange bug when nodenr is None, it seems like somethimes None<0 is True!!!
        if node_nr < 0 and node_nr!=None:
            return iterable
        
        if isinstance(iterable, numx.ndarray):
            return self._execute_seq(iterable, node_nr, benchmark)
        res = []
        empty_iterator = True
        for x in iterable:
            empty_iterator = False
            res.append(self._execute_seq(x, node_nr))
        if empty_iterator:
            errstr = ("The execute data iterator is empty.")
    #        raise mdp.MDPException(errstr)
            raise mdp.linear_flows.FlowException(errstr)
        res = numx.concatenate(res)
        print "result shape is:", res.shape
        return numx.concatenate(res)

    #Supports single array data but also iterator/list
    #Small variation over flow_execute: Result has same shape
    def flow_execute_data_vec(self, iterable, node_nr = None, benchmark=None):
        """Process the data through all nodes in the flow.
        keeping the structure
        If 'nodenr' is specified, the flow is executed only up to
        node nr. 'nodenr'. This is equivalent to 'flow[:nodenr+1](iterable)'.
        """
        if isinstance(iterable, numx.ndarray):
            return self._execute_seq(iterable, node_nr, benchmark)
        if node_nr == -1:
            return iterable        
        res = []
        empty_iterator = True
        for x in iterable:
            empty_iterator = False
            res.append(self._execute_seq(x, node_nr))
        if empty_iterator:
            errstr = ("The execute data iterator is empty.")
    #        raise mdp.MDPException(errstr)
            raise mdp.linear_flows.FlowException(errstr)
        return res

    mdp.Flow.special_train_cache_scheduler = flow_special_train_cache_scheduler
    mdp.Flow.special_train_cache_scheduler_sets = flow_special_train_cache_scheduler_sets
    mdp.Flow.execute = flow_execute
    mdp.Flow.execute_data_vec = flow_execute_data_vec
    mdp.Flow._execute_seq = flow__execute_seq
    

numx_linalg  = mdp.numx_linalg
#sfa_expo: eigenvalues wB of B are corrected with a factor sqrt(wB**sfa_expo) instead of sqrt(wB)
#pca_expo: whithening is done using corrected eigenvalues sqrt(wB**pca_expo) instead of sqrt(wB)   
def _symeig_fake_regularized(A, B = None, eigenvectors = True, range = None,
                 type = 1, overwrite = False, sfa_expo=1.0, pca_expo=1.0, magnitude_sfa_biasing=False):
    """Solve standard and generalized eigenvalue problem for symmetric
(hermitian) definite positive matrices.
This function is a wrapper of LinearAlgebra.eigenvectors or
numarray.linear_algebra.eigenvectors with an interface compatible with symeig.

    Syntax:

      w,Z = symeig(A) 
      w = symeig(A,eigenvectors=0)
      w,Z = symeig(A,range=(lo,hi))
      w,Z = symeig(A,B,range=(lo,hi))

    Inputs:

      A     -- An N x N matrix.
      B     -- An N x N matrix.
      eigenvectors -- if set return eigenvalues and eigenvectors, otherwise
                      only eigenvalues 
      turbo -- not implemented
      range -- the tuple (lo,hi) represent the indexes of the smallest and
               largest (in ascending order) eigenvalues to be returned.
               1 <= lo < hi <= N
               if range = None, returns all eigenvalues and eigenvectors. 
      type  -- not implemented, always solve A*x = (lambda)*B*x
      overwrite -- not implemented
      
    Outputs:

      w     -- (selected) eigenvalues in ascending order.
      Z     -- if range = None, Z contains the matrix of eigenvectors,
               normalized as follows:
                  Z^H * A * Z = lambda and Z^H * B * Z = I
               where ^H means conjugate transpose.
               if range, an N x M matrix containing the orthonormal
               eigenvectors of the matrix A corresponding to the selected
               eigenvalues, with the i-th column of Z holding the eigenvector
               associated with w[i]. The eigenvectors are normalized as above.
    """

    dtype = numx.dtype(_greatest_common_dtype([A, B]))
    try:
        if B is None:
            w, Z = numx_linalg.eigh(A)
        else:
            # make B the identity matrix
            wB, ZB = numx_linalg.eigh(B)
            _assert_eigenvalues_real_and_positive(wB, dtype)

            if sfa_expo < 1:
                ex = "sfa_expo should be at least 1.0"
                raise Exception(ex)
            if pca_expo > 1:
                ex = "pca_expo should be at most 1.0"
                raise Exception(ex)
            
#            pca_expo = 0.25
#            sfa_expo = 1.1

#TODO: SMART SFA_PCA WITHOUT STRANGE EXPONENTS
            wB_pca_mapped = wB.real
            #wB_pca_mapped = wB.real**pca_expo
            ZB_pca = ZB.real / numx.sqrt(wB_pca_mapped)
            if magnitude_sfa_biasing:
                ZB_sfa = ZB.real / numx.sqrt(wB.real**sfa_expo)
                quit()
            else:
                ZB_sfa = ZB.real / numx.sqrt(wB.real*numpy.linspace(sfa_expo, 1.0, len(wB)))
#            ZB_sfa = ZB.real / numx.sqrt(wB.real**sfa_expo)

##            if magnitude_sfa_biasing:
##                ZB_sfa = ZB.real / numx.sqrt(wB.real**sfa_expo)
##            elif sfa_expo and False:
##                print "sfa_expo=", sfa_expo
##                ZB_sfa = ZB.real / numx.sqrt(wB.real* numpy.linspace(1.33, 1.0, len(wB)))
##                print "len(wB) is", len(wB)
##                print "something went wrong"
###                quit()
##            else:
##                ZB_sfa = ZB.real / numx.sqrt(wB.real)

            # transform A in the new basis: A = ZB^T * A * ZB
            A = mdp.utils.mult(mdp.utils.mult(ZB_sfa.T, A), ZB_sfa)
            # diagonalize A
            w, ZA = numx_linalg.eigh(A)
            Z = mdp.utils.mult(ZB_pca, ZA)
    except numx_linalg.LinAlgError, exception:
        raise SymeigException(str(exception))

    _assert_eigenvalues_real_and_positive(w, dtype)
    w = w.real
    Z = Z.real
    
    idx = w.argsort()
    w = w.take(idx)
    Z = Z.take(idx, axis=1)
    
    # sanitize range:
    n = A.shape[0]
    if range is not None:
        lo, hi = range
        if lo < 1:
            lo = 1
        if lo > n:
            lo = n
        if hi > n:
            hi = n
        if lo > hi:
            lo, hi = hi, lo
        
        Z = Z[:, lo-1:hi]
        w = w[lo-1:hi]

    # the final call to refcast is necessary because of a bug in the casting
    # behavior of Numeric and numarray: eigenvector does not wrap the LAPACK
    # single precision routines
    if eigenvectors:
        return mdp.utils.refcast(w, dtype), mdp.utils.refcast(Z, dtype)
    else:
        return mdp.utils.refcast(w, dtype)



from mdp.utils._symeig import _assert_eigenvalues_real_and_positive, _type_keys, _type_conv, _greatest_common_dtype
#routines



def KNNClassifier_klabels(self,x):
    """Label the data by comparison with the reference points."""
    square_distances = (x*x).sum(1)[:, numx.newaxis] \
                      + (self.samples*self.samples).sum(1)
    square_distances -= 2 * numx.dot(x, self.samples.T)
    min_inds = square_distances.argsort()

#    print "min_inds[:,0:self.k]", min_inds[:,0:self.k]
    min_inds_sel = min_inds[:,0:self.k].astype(int)
#    print "min_inds_sel", min_inds_sel
    my_ordered_labels = numpy.array(self.ordered_labels)
    klabels = my_ordered_labels[min_inds_sel]
    return klabels


def KNNClassifier_klabel_avg(self,x):
    """Label the data by comparison with the reference points."""
    square_distances = (x*x).sum(1)[:, numx.newaxis] \
                      + (self.samples*self.samples).sum(1)
    square_distances -= 2 * numx.dot(x, self.samples.T)
    min_inds = square_distances.argsort()

#    print "min_inds[:,0:self.k]", min_inds[:,0:self.k]
    min_inds_sel = min_inds[:,0:self.k].astype(int)
#    print "min_inds_sel", min_inds_sel
#    klabels = [self.ordered_labels[indices] for indices in min_inds[:,0:self.k]]
    my_ordered_labels = numpy.array(self.ordered_labels)
    klabels = my_ordered_labels[min_inds_sel]
#    klabels = self.ordered_labels[min_inds_sel]
#        win_inds = [numx.bincount(self.sample_label_indices[indices[0:self.k]]).
#                   argmax(0) for indices in min_inds]
#        labels = [self.ordered_labels[i] for i in win_inds]
#    print "klabels", klabels 
    return klabels.mean(axis=1)

mdp.nodes.KNNClassifier.klabel_avg = KNNClassifier_klabel_avg
mdp.nodes.KNNClassifier.klabels = KNNClassifier_klabels

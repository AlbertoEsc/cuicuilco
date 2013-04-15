#Basic Experiments involving SFA applied to sequences of images
#Improving hierarchy to support a lattice based structure
#Allowing for selective masking of input data / channel data
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 14 Mai 2009
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott

import numpy
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import Image
import mdp
import os
import glob
import random
from  eban_SFA_libs import *

##default scales (-1,1) into (0, 255)
##Normalized to [-scale/2, scale/2]
#def scale_to(val, av_in=0.0, delta_in=2.0, av_out=127.5, delta_out=255.0, scale=1.0, transf='lin'):
#    normalized = scale*(val - av_in) / delta_in
#    if transf == 'lin':
#        return normalized * delta_out + av_out
#    elif transf == 'tanh':
#        return numpy.tanh(normalized)*delta_out + av_out 
#    else:
#        raise Exception("Wrong transf in scale_to! (choose from 'lin', 'tanh'")
#
#
#def wider(imag, scale_x=1):
#    z = numpy.zeros((imag.shape[0], imag.shape[1]*scale_x))
#    for i in range(imag.shape[1]):
#        tmp = imag[:,i].reshape(imag.shape[0], 1)
#        z[:,scale_x*i:scale_x*(i+1)] = tmp
#    return z
#
#
#def format_coord(x, y, numcols, numrows, width_factor=1.0, height_factor=1.0):
#    col = int(x/width_factor+0.5)
#    row = int(y/height_factor+0.5)
#    if col>=0 and col<numcols and row>=0 and row<numrows:
#        z = X[row,col]
#        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
#    else:
#        return 'x=%1.4f, y=%1.4f'%(x, y)
#
#
##The following class was borrowed from '/mdp/nodes/expansion_nodes.py'
#class _ExpansionNode(mdp.Node):
#    def __init__(self, input_dim = None, dtype = None):
#        super(_ExpansionNode, self).__init__(input_dim, None, dtype)
#    def expanded_dim(self, dim):
#        return dim
#    def is_trainable(self):
#        return False
#    def is_invertible(self):
#        return False
#    def _set_input_dim(self, n):
#        self._input_dim = n
#        self._output_dim = self.expanded_dim(n)
#    def _set_output_dim(self, n):
#        msg = "Output dim cannot be set explicitly!"
#        raise mdp.NodeException(msg)
#
##using the provided average and standard deviation
#def gauss_noise(x, avg, std):
#    return numpy.random.normal(avg, std, x.shape)
#
##Zero centered
#def additive_gauss_noise(x, avg, std):
#    return x + numpy.random.normal(0, std, x.shape)
#
#    
#class RandomizedMaskNode(mdp.Node):
#    """Selectively mask some components of a random variable by 
#    hiding them with arbitrary noise or by removing them
#    Original code contributed by Alberto Escalante, inspired by NoiseNode
#    """    
#    def __init__(self, remove_mask=None, noise_amount_mask=None, noise_func = gauss_noise, noise_args = (0, 1),
#                 noise_mix_func = None, input_dim = None, dtype = None):
#        self.remove_mask = remove_mask
# 
#        self.noise_amount_mask = noise_amount_mask
#        self.noise_func = noise_func
#        self.noise_args = noise_args
#        self.noise_mix_func = noise_mix_func 
#        self.seen_samples = 0
#        self.x_avg = None
#        self.x_std = None
#        self.type=dtype
#
#        if remove_mask != None and input_dim == None:
#            input_dim = remove_mask.size
#        elif remove_mask == None and input_dim != None: 
#            remove_mask = numpy.zeros(input_dim) > 0.5
#        elif remove_mask != None and input_dim != None:
#            if remove_mask.size != input_dim:
#                err = "size of remove_mask and input_dim not compatible"
#                raise Exception(err)
#        else:
#            err = "At least one of input_dim or remove_mask should be specified"
#            raise Exception(err)
#        
#        if noise_amount_mask is None:
#            print "Signal will be only the computed noise"
#            self.noise_amount_mask = numpy.ones(input_dim)
#        else:
#            self.noise_amount_mask = noise_amount_mask 
#            
#        output_dim = remove_mask.size - remove_mask.sum()
#        print "Output_dim should be:", output_dim 
#        super(RandomizedMaskNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
#                   
#    def is_trainable(self):
#        return True
#
#    def _train(self, x):
#        if self.x_avg == None:
#            self.x_avg = numpy.zeros(self.input_dim, dtype=self.type)
#            self.x_std = numpy.zeros(self.input_dim, dtype=self.type)
#        new_samples = x.shape[0]
#        self.x_avg = (self.x_avg * self.seen_samples + x.sum(axis=0)) / (self.seen_samples + new_samples)
#        self.x_std = (self.x_std * self.seen_samples + x.std(axis=0)*new_samples) / (self.seen_samples + new_samples)
#        self.seen_samples = self.seen_samples + new_samples
#        
#    def is_invertible(self):
#        return False
#
#    def _execute(self, x):
#        vec_noise_func = numpy.vectorize(self.noise_func)
#
#        print "computed X_avg=", self.x_avg
#        print "computed X_std=", self.x_std
#        noise_mat = self.noise_func(x, self.x_avg, self.x_std)
##        noise_mat = self._refcast(self.noise_func(*self.noise_args,
##                                                  **{'size': x.shape}))
#        print "Noise_amount_mask:", self.noise_amount_mask
#        print "Noise_mat:", noise_mat
#        noisy_signal = (1.0 - self.noise_amount_mask) * x + self.noise_amount_mask * noise_mat
#        preserve_mask = self.remove_mask == False
#        return noisy_signal[:, preserve_mask]
#
#
#def extend_channel_mask_to_signal_mask(input_dim, channel_mask):
#    channel_size = channel_mask.size
#    rep = input_dim / channel_size
#    if input_dim % channel_size != 0:
#        err="incompatible channel_mask length and input_dim"
#        raise Exception(err)  
#    res = channel_mask.copy()
#    for iter in range(rep-1):
#        res = numpy.concatenate((res, channel_mask))
#    return res
#
#def identity(x):
#    return x
#
#def abs_dif(x1, x2):
#    return numpy.abs(x1 - x2)
#
#def abs_sum(x1, x2):
#    return numpy.abs(x1 + x2)
#
#def multiply(x1, x2):
#    return x1 * x2
#
#def sqrt_abs_sum(x1, x2):
#    return numpy.sqrt(numpy.abs(x1+x2))
#
#def sqrt_abs_dif(x1, x2):
#    return numpy.sqrt(numpy.abs(x1-x2))
#
##Expansion with terms: f(x1,x1), f(x1,x2), ... f(x1,xn), f(x2,x2), ... f(xn,xn)
##If reflexive=True, skip terms f(xj, xj)
#def pairwise_expansion(x, func, reflexive=True):
#    x_height, x_width = x.shape
#    if reflexive==True:
#        k=0
#        out = numpy.zeros((x_height, x_width*(x_width+1)/2))
#    else:
#        k=1
#        out = numpy.zeros((x_height, x_width*(x_width-1)/2))    
#    mask = numpy.triu(numpy.ones((x_width,x_width)), k) > 0.5
#    for i in range(0, x_height):
#        y1 = x[i].reshape(x_width, 1)
#        y2 = x[i].reshape(1, x_width)
#        yexp = func(y1, y2)
##        print "yexp=", yexp
#        out[i] = yexp[mask]
#    return out    
#
#def pair_abs_dif_ex(x):
#    return pairwise_expansion(x, abs_dif, reflexive=False)
#
#def pair_abs_sum_ex(x):
#    return pairwise_expansion(x, abs_sum)
#
#def pair_prod_ex(x):
#    return pairwise_expansion(x, multiply)
#
#def pair_sqrt_abs_sum_ex(x):
#    return pairwise_expansion(x, sqrt_abs_sum)
#
#def pair_sqrt_abs_dif_ex(x):
#    return pairwise_expansion(x, sqrt_abs_dif, reflexive=False)
#
#
#class GeneralExpansionNode(mdp.Node):
#    def __init__(self, funcs, input_dim = None, dtype = None):
#        self.funcs = funcs
#        super(GeneralExpansionNode, self).__init__(input_dim, dtype)
#    def expanded_dim(self, n):
#        exp_dim=0
#        x = numpy.zeros((1,n))
#        for func in self.funcs:
#            outx = func(x)
#            exp_dim += outx.shape[1]
#        return exp_dim
#    def is_trainable(self):
#        return False
#    def is_invertible(self, x):
#        return False
#    def _set_input_dim(self, n):
#        self._input_dim = n
#        self._output_dim = self.expanded_dim(n)
#    def _execute(self, x):
#        out = []
#        for func in self.funcs:
#            if out==[]:
#                out = func(x)
#            else:
#                out = numpy.concatenate((out, func(x)), axis=1)
#        return out
#
##Computes the coordinates of the lattice points that lie within the image
#def compute_lattice_matrix(v1, v2, mask, x_in_channels, y_in_channels, in_channel_dim=1, n0_1 = 0, n0_2 = 0, wrap_x= False, wrap_y= False, input_dim = None, dtype = None, ignore_cover = True, allow_nonrectangular_lattice=False):
#    if v1[1] != 0 | v1[0] <= 0 | v2[0] < 0 | v2[1] <= 0:
#        err = "v1 must be horizontal: v1[0] > 0, v1[1] = 0, v2[0] >= 0, v2[1] > 0"
#        raise Exception(err)  
#
#    if in_channel_dim != 1:
#        err = "only single channel inputs supported now"
#        raise Exception(err)  
#
##assume no wrapping 
#    image = numpy.array(range(0, x_in_channels * y_in_channels))
#    image.reshape((y_in_channels, x_in_channels))
#    sub_image = numpy.array(range(0, mask.shape[0] * mask.shape[1]))
#    sub_image.reshape((mask.shape[0], mask.shape[1]))
#    mask_i = mask.astype("int")
#    mask_height, mask_width = mask.shape
#    out_channel_dim = mask_i.sum()
##    print "Mask shape is ", mask.shape
#    
#    mat_height = (y_in_channels - mask.shape[0])/v2[1] + 1
#    mat_width = (x_in_channels-mask.shape[1])/v1[0] + 1
#    
#    mat = numpy.ones((mat_height, mat_width, 2)) * -1
##Create Index Matrix, -1 entries equal empty cell
##    print "Mat shape is ", mat.shape
#    ind_y = 0
#    for iy in range(0, mat_height):
#            #x,y are real subimage positions
#            #ix, iy are the coefficients of x,y in base v1 and v2
#            #ind_y, ind_x are the indices in the matrix mat that contains the centers (upper-left corners) of each subimage
#        y = iy * v2[1]
#        min_ix = -1 * numpy.int(iy * v2[0] / v1[0])
#        max_ix = numpy.floor( (x_in_channels - mask.shape[1] - iy *v2[0]) * 1.0 /  v1[0])
#        max_ix = numpy.int(max_ix)
#        ind_x = 0
#        for ix in range(min_ix, max_ix + 1):
#            x = iy *v2[0] + ix * v1[0]
# #           print "value of ind_x, ind_y = ", (ind_x, ind_y)
# #           print "Adding Point (", x, ", ", y, ")"
#            mat[ind_y, ind_x] = (x, y)
#            ind_x = ind_x + 1
#        ind_y = ind_y + 1
#
#    if not allow_nonrectangular_lattice:
#        if (-1, -1) in mat[:,mat_width-1]:
#            mat = mat[:,:mat_width-2]
#        
#    return mat
#
#def compute_lattice_matrix_connections(v1, v2, preserve_mask, x_in_channels, y_in_channels, in_channel_dim=1, allow_nonrectangular_lattice=False):
#        if in_channel_dim > 1:
#            err = "Error, feature not supported in_channel_dim > 1"
#            raise Exception(err)
##
#        lat_mat = compute_lattice_matrix(v1, v2, preserve_mask, x_in_channels, y_in_channels, allow_nonrectangular_lattice=allow_nonrectangular_lattice)             
##
#        print "lat_mat =", lat_mat
#        image_positions = numpy.array(range(0, x_in_channels * y_in_channels * in_channel_dim))
#        image_positions = image_positions.reshape(y_in_channels, x_in_channels)
##
##
#        mask_indices = image_positions[0:preserve_mask.shape[0], 0:preserve_mask.shape[1]][preserve_mask].flatten()
##
#        connections = None
#        for ind_y in range(lat_mat.shape[0]):
#            for ind_x in range(lat_mat.shape[1]):
#                if(lat_mat[ind_y, ind_x][0] != -1):
#                    if connections is None:
#                        connections = numpy.array(mask_indices + (lat_mat[ind_y, ind_x][0] + lat_mat[ind_y, ind_x][1]*x_in_channels))
#                    else:
#                        connections = numpy.concatenate((connections, mask_indices + (lat_mat[ind_y, ind_x][0] + lat_mat[ind_y, ind_x][1]*x_in_channels) ))
#                else:
#                    print "Void entry in lattice_matrix skipped"
##
##
#        print "Connections are: ", connections
#        return (connections.astype('int'), lat_mat)
#
##Note: generalize with other functions!!!
##Careful with the sign used!!!
#class BinomialAbsoluteExpansionNode(mdp.Node):
#    def expanded_dim(self, n):
#        return n + n*(n+1)/2
#    def is_trainable(self):
#        return False
#    def is_invertible(self, x):
#        return False
#    def _set_input_dim(self, n):
#        self._input_dim = n
#        self._output_dim = self.expanded_dim(n)
#    def _execute(self, x):
#        out = numpy.concatenate((x, pairwise_expansion(x, abs_sum)), axis=1)
#        return out
#
##Obsolete, there is a method that does this... (with t-1 as denominator)
#def comp_variance(x):
#    t, num_vars = x.shape
#    delta = numpy.zeros(num_vars)
#    for row in x:
#        delta = delta + numpy.square(row)
#    delta = delta / (t)
#    return delta
#
#def comp_delta(x):
#    t, num_vars = x.shape
#    delta = numpy.zeros(num_vars)
#    xderiv = x[1:, :]-x[:-1, :]
#    for row in xderiv:
#        delta = delta + numpy.square(row)
#    delta = delta / (t-1)
#    return delta
#
#def comp_eta(x):
#    t, num_vars = x.shape
#    return t/(2*numpy.pi) * numpy.sqrt(comp_delta(x))
#
#def str2(x):
#    c=""
#    for w in x:
#        if c == "":
#            c+="%.2f" % w
#        else:
#            c+=",%.2f" % w
#    return c
#
#def str3(x):
#    c=""
#    for w in x:
#        if c == "":
#            c+="%.3f" % w
#        else:
#            c+=",%.3f" % w
#    return c


#Renaming filenames => Fix perl script
#image_files = glob.glob(im_seq_base_dir + "/*p?_*tif")
#image_files
#for i in range(0,len(image_files)+0):
# os.rename(prefix + str(i) + sufix, prefix+"%03d"%i + sufix)
#
#image_files = glob.glob(im_seq_base_dir + "/*p??_*tif")
#for i in range(10,len(image_files)+10):
# os.rename(prefix + str(i) + sufix, prefix+"%03d"%i + sufix)



print "TESTING RandomizedMaskNode"
input_dim = 6
channel_size = 3
var_size = (1, input_dim)
remove_channel_mask = numpy.arange(channel_size) < channel_size / 2.5
print "Remove Channel Mask:", remove_channel_mask
remove_mask = extend_channel_mask_to_signal_mask(input_dim, remove_channel_mask)
print "Remove Mask:", remove_mask

#Only noise
noise_amount_mask = numpy.random.normal(0.5, 0.5, var_size)
noise_amount_mask = noise_amount_mask / noise_amount_mask.max()
print "Noise_amount Mask:", noise_amount_mask

avg = numpy.arange(input_dim) * 10 / input_dim + 1
std = numpy.arange(input_dim) * 2.0 / input_dim + 0.01

print "Real X_Avg:", avg
print "Real X_Std:", std

randMaskNode = RandomizedMaskNode(remove_mask, noise_amount_mask)

num_samples = 4
size_samples = (num_samples, input_dim)
x = numpy.random.normal(avg, std, size_samples)
print "sampled x=", x

randMaskNode.train(x)
randMaskNode.stop_training()

print "Using default noise: Gaussian Noise for each component with the same avg and std as the input"
print "For a coefficient noise_amount_mask[i] = 1 output is only this noise" 
print "for noise_amount_mask[i] = 0 output is only the input unaffected" 
z1 = randMaskNode(x)
print "z1=randMaskNode(x)=", z1

y = numpy.random.normal(0, 0.1, size_samples)
print "abnormally sampled y=", y
 
z2 = randMaskNode(y)
print "z2=randMaskNode(y)=", z2


print "TESTING Lattice Generation Functions"
v1 = (4,0)
v2 = (2,5)
mask = numpy.ones((4,4))
print "v1=", v1, ", v2=", v2, ". Notice ordering (x,y) for lattice points"
print "mask size: width=", mask.shape[1], ", height =", mask.shape[0]
print "first lattice. width=", 10, "height=", 12
l1 = compute_lattice_matrix(v1, v2, mask, 10, 12)
print l1
print "second lattice. width=", 9, "height=", 12
l2 = compute_lattice_matrix(v1, v2, mask, 9, 12)
print l2
print "third lattice. width=", 10, "height=", 9
l3 = compute_lattice_matrix(v1, v2, mask, 10, 9)
print l3
print "fourth lattice. width=", 10, "height=", 8
l4 =compute_lattice_matrix(v1, v2, mask, 10, 8)
print l4


scale_disp = 3

#SEQUENCE OF IMAGES
image_width  = 256
image_height = 192

subimage_width  = image_width/5
subimage_height = image_height/2 
subimage_width  = 49
subimage_height = 91
 
#pixelsampling: 1=each pixel, 2= one yes and one not
subimage_pixelsampling= 2

subimage_first_row= image_height/2-subimage_height*subimage_pixelsampling/2
subimage_first_column=image_width/2-subimage_width*subimage_pixelsampling/2

print "Images: width=%d, height=%d, subimage_width=%d,subimage_height=%d"%(image_width,image_height, subimage_width,subimage_height)

#Open sequence of images
im_seq_base_dir = "/home/escalafl/datasets/face_gen"
image_files = glob.glob(im_seq_base_dir + "/*tif")
image_files.sort()
num_images = len(image_files)

subimages = numpy.zeros((num_images, subimage_width * subimage_height))
act_im_num = 0
im_ori = []
for image_file in image_files:
    im = Image.open(image_file)
    im = im.convert("L")
    im_arr = numpy.asarray(im)
    im_small = im_arr[subimage_first_row:(subimage_first_row+subimage_height*subimage_pixelsampling):subimage_pixelsampling,
                              subimage_first_column:(subimage_first_column+subimage_width*subimage_pixelsampling):subimage_pixelsampling].astype(float)
    im_ori.append(im_small)
    subimages[act_im_num] = im_small.flatten()
    act_im_num = act_im_num+1
    

#Add Gaussian noise
#random.gauss(x,x)

#Create Figure
f1 = plt.figure()
plt.suptitle("Sequence of Images...")
  
#display Sequence of images
#Alternative: im1.show(command="xv")
a11 = plt.subplot(2,2,1)
plt.title("Sequence of Images")
subimagesdisp = subimages
a11.imshow(subimagesdisp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)


#display first image
#Alternative: im1.show(command="xv")
f1a12 = plt.subplot(2,2,2)
plt.title("Last image in Sequence")
im_smalldisp = im_small
f1a12.imshow(im_smalldisp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)

#Retrieve Image in Sequence
display = subimages
plot = f1a12
figure = f1
display_length = num_images
display_width  = subimage_width
display_height = subimage_height
def on_press(event):
    global display, figure, plot, display_width, display_height, display_length
    print 'you pressed', event.button, event.xdata, event.ydata
    y = event.ydata
    if y < 0:
        y = 0
    if y >= display_length:
        y =  display_length -1

    display_im = display[y].reshape((display_height, subimage_width))
    plot.imshow(display_im, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)

    
    figure.canvas.draw()
    
f1.canvas.mpl_connect('button_press_event', on_press)


print "Original Rectangular2dSwitchboard processing"
#Create Switchboard L0
#width=51, height=96
#x_field_channels=21
#y_field_channels=
#x_field_spacing=6
#y_field_spacing=6
#in_channel_dim=1
x_field_channels_L0=14
y_field_channels_L0=14
x_field_spacing_L0=7
y_field_spacing_L0=7
in_channel_dim_L0=1
# 6 x 12

print "About to create basic perceptive field of widht=%d, height=%d"%(x_field_channels_L0,y_field_channels_L0) 
print "With a spacing of horiz=%d, vert=%d, and %d channels"%(x_field_spacing_L0,y_field_spacing_L0,in_channel_dim_L0)
switchboard_L0 = mdp.hinet.Rectangular2dSwitchboard(subimage_width, subimage_height,x_field_channels_L0,y_field_channels_L0,x_field_spacing_L0,y_field_spacing_L0,in_channel_dim_L0)
switchboard_L0.connections

#Create single SFA Node
#WARNING!!! SIGNAL TOO SMALL!!! sfa_out_dim_L0 = 10
sfa_out_dim_L0 = 10
sfa_node_L0 = mdp.nodes.SFANode(input_dim=switchboard_L0.out_channel_dim, output_dim=sfa_out_dim_L0)

#Create array of sfa_nodes (just one node, but clonned)
sfa_layer_L0 = mdp.hinet.CloneLayer(sfa_node_L0, n_nodes=switchboard_L0.output_channels)



#Create Switchboard L1
x_field_channels_L1=4
y_field_channels_L1=4
x_field_spacing_L1=2
y_field_spacing_L1=2
in_channel_dim_L1=sfa_out_dim_L0

print "About to create intermediate layer widht=%d, height=%d"%(x_field_channels_L1,y_field_channels_L1) 
print "With a spacing of horiz=%d, vert=%d, and %d channels"%(x_field_spacing_L1,y_field_spacing_L1,in_channel_dim_L1)
switchboard_L1 = mdp.hinet.Rectangular2dSwitchboard(12, 6, x_field_channels_L1,y_field_channels_L1,x_field_spacing_L1,y_field_spacing_L1,in_channel_dim_L1)
switchboard_L1.connections

#Create L1 sfa node
sfa_out_dim_L1 = 12
sfa_node_L1 = mdp.nodes.SFANode(input_dim=switchboard_L1.out_channel_dim, output_dim=sfa_out_dim_L1)

#Create array of sfa_nodes (just one node, but clonned)
#This one should not be clonned!!!
sfa_layer_L1 = mdp.hinet.CloneLayer(sfa_node_L1, n_nodes=switchboard_L1.output_channels)


#Create L2 sfa node
sfa_out_dim_L2 = 4
sfa_node_L2 = mdp.nodes.SFANode(output_dim=sfa_out_dim_L2)


#Join Switchboard and SFA layer in a single flow
flow = mdp.Flow([switchboard_L0, sfa_layer_L0, switchboard_L1, sfa_layer_L1, sfa_node_L2])

flow.train(subimages)
sl_seq = flow.execute(subimages)

a11 = plt.subplot(2,2,3)
plt.title("Output Unit L2. Rectangular 2D Switch")
sl_seqdisp = sl_seq[:, range(0,sfa_out_dim_L2)]
sl_seqdisp = scale_to(sl_seqdisp, sl_seq.mean(), sl_seq.max()-sl_seq.min(), 127.5, 255.0, scale_disp, 'tanh')
a11.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.3f, max=%.3f, scale=%.3f\n e[]=" % (sl_seq.min(), sl_seq.max(), scale_disp)+str3(comp_eta(sl_seq)[0:5]))




print "Lattice based processing"
#Create Switchboard L0
#width=51, height=96
#x_field_channels=21
#y_field_channels=
#x_field_spacing=6
#y_field_spacing=6
#in_channel_dim=1
#subimage_width = subimage_height = 14 + 7
x_field_channels_L0=14
y_field_channels_L0=14
x_field_spacing_L0=7
y_field_spacing_L0=7
in_channel_dim_L0=1

v1 = (x_field_spacing_L0, 0)
v2 = (x_field_spacing_L0, y_field_spacing_L0)

preserve_mask_L0 = numpy.ones((y_field_channels_L0, x_field_channels_L0)) > 0.5
# 6 x 12
print "About to create (lattice based) perceptive field of widht=%d, height=%d"%(x_field_channels_L0,y_field_channels_L0) 
print "With a spacing of horiz=%d, vert=%d, and %d channels"%(x_field_spacing_L0,y_field_spacing_L0,in_channel_dim_L0)

(mat_connections, lat_mat_L0) = compute_lattice_matrix_connections(v1, v2, preserve_mask_L0, subimage_width, subimage_height, in_channel_dim_L0)
print "matrix connections:"
print mat_connections

switchboard_L0 = mdp.hinet.Switchboard(subimage_width * subimage_height, mat_connections)
switchboard_L0.connections

#Create single SFA Node
#Warning Signal too short!!!!!sfa_out_dim_L0 = 20
sfa_out_dim_L0 = 10
sfa_node_L0 = mdp.nodes.SFANode(input_dim=preserve_mask_L0.size, output_dim=sfa_out_dim_L0)

#Create array of sfa_nodes (just one node, but clonned)
sfa_layer_L0 = mdp.hinet.CloneLayer(sfa_node_L0, n_nodes=lat_mat_L0.size / 2)


#Create Switchboard L1
x_field_channels_L1=4
y_field_channels_L1=4
x_field_spacing_L1=2
y_field_spacing_L1=2
in_channel_dim_L1=sfa_out_dim_L0

print "About to create intermediate layer widht=%d, height=%d"%(x_field_channels_L1,y_field_channels_L1) 
print "With a spacing of horiz=%d, vert=%d, and %d channels"%(x_field_spacing_L1,y_field_spacing_L1,in_channel_dim_L1)
switchboard_L1 = mdp.hinet.Rectangular2dSwitchboard(12, 6, x_field_channels_L1,y_field_channels_L1,x_field_spacing_L1,y_field_spacing_L1,in_channel_dim_L1)
switchboard_L1.connections

#Create L1 sfa node
sfa_out_dim_L1 = 12
sfa_node_L1 = mdp.nodes.SFANode(input_dim=switchboard_L1.out_channel_dim, output_dim=sfa_out_dim_L1)

#Create array of sfa_nodes (just one node, but clonned)
#This one should not be clonned!!!
sfa_layer_L1 = mdp.hinet.CloneLayer(sfa_node_L1, n_nodes=switchboard_L1.output_channels)


#Create L1 sfa node
sfa_out_dim_L2 = 4
sfa_node_L2 = mdp.nodes.SFANode(output_dim=sfa_out_dim_L2)


#Join Switchboard and SFA layer in a single flow
flow = mdp.Flow([switchboard_L0, sfa_layer_L0, switchboard_L1, sfa_layer_L1, sfa_node_L2])

flow.train(subimages)
sl_seq = flow.execute(subimages)

a11 = plt.subplot(2,2,4)
plt.title("Output Unit L2. Identical Lattice Switch")
sl_seqdisp = sl_seq[:, range(0,sfa_out_dim_L2)]
sl_seqdisp = scale_to(sl_seqdisp, sl_seq.mean(), sl_seq.max()-sl_seq.min(), 127.5, 255.0, scale_disp, 'tanh')
a11.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.3f, max=%.3f, scale=%.3f\n e[]=" % (sl_seq.min(), sl_seq.max(), scale_disp)+str3(comp_eta(sl_seq)[0:5]))


print "Finished!"

#Create Figure
f2 = plt.figure()
plt.suptitle("Different Masks at input Images...")

print "Changing Lattices..."
x_field_channels_L0=14
y_field_channels_L0=14
x_field_spacing_L0=7
y_field_spacing_L0=7
in_channel_dim_L0=1

v1 = (x_field_spacing_L0, 0)
v2 = (x_field_spacing_L0, y_field_spacing_L0)

print "Creating Masks..."
x_center = (x_field_channels_L0-1) / 2.0
x_dmax = (x_field_channels_L0-1) / 2.0
y_center = (y_field_channels_L0-1) / 2.0
y_dmax = (y_field_channels_L0-1) / 2.0

preserve_mask_L0_circ = numpy.ones((y_field_channels_L0, x_field_channels_L0)) > 0.5
#faktor=1 => ellipse of same height and width, smaller faktor => smaller ellipse
#faktor = 0.60
faktor = 0.50
for xx_mask in range(x_field_channels_L0):
    for yy_mask in range(y_field_channels_L0):
        heights = ((xx_mask - x_center) / x_dmax) ** 2 + \
                  ((yy_mask - y_center) / y_dmax) ** 2
        if heights > 2 * (faktor**2):
            preserve_mask_L0_circ[yy_mask, xx_mask] = False
print "Size of circular perceptive field is ", preserve_mask_L0_circ.sum()

preserve_mask_L0_prism = numpy.ones((y_field_channels_L0, x_field_channels_L0)) > 0.5
#faktor=1 => ellipse of same height and width, smaller faktor => smaller ellipse
#faktor = 0.50
faktor = 0.40
for xx_mask in range(x_field_channels_L0):
    for yy_mask in range(y_field_channels_L0):
        heights = numpy.abs((xx_mask - x_center) / x_dmax) + \
                  numpy.abs((yy_mask - y_center) / y_dmax)
        if heights > 2 * faktor:
            preserve_mask_L0_prism[yy_mask, xx_mask] = False
print "Size of prismal perceptive field is ", preserve_mask_L0_prism.sum()

preserve_mask_L0_rect = numpy.ones((y_field_channels_L0, x_field_channels_L0)) > 0.5
#faktor=1 => ellipse of same height and width, smaller faktor => smaller ellipse
#faktor = 0.70
faktor = 0.50
for xx_mask in range(x_field_channels_L0):
    for yy_mask in range(y_field_channels_L0):
        heights = numpy.abs((xx_mask - x_center) / x_dmax) + \
                  numpy.abs((yy_mask - y_center) / y_dmax)
        if numpy.abs((xx_mask - x_center) / x_dmax) > faktor:
            preserve_mask_L0_rect[yy_mask, xx_mask] = False
        if numpy.abs((yy_mask - y_center) / y_dmax) > faktor:
            preserve_mask_L0_rect[yy_mask, xx_mask] = False
print "Size of rectangular perceptive field is ", preserve_mask_L0_rect.sum()

preserve_mask_L0_cross = numpy.ones((y_field_channels_L0, x_field_channels_L0)) > 0.5
#faktor=1 => ellipse of same height and width, smaller faktor => smaller ellipse
#faktor = 0.25
faktor = 0.20
for xx_mask in range(x_field_channels_L0):
    for yy_mask in range(y_field_channels_L0):
        heights = numpy.abs((xx_mask - x_center) / x_dmax) + \
                  numpy.abs((yy_mask - y_center) / y_dmax)
        if numpy.abs((xx_mask - x_center) / x_dmax) > faktor and \
           numpy.abs((yy_mask - y_center) / y_dmax) > faktor:
            preserve_mask_L0_cross[yy_mask, xx_mask] = False
print "Size of cross perceptive field is ", preserve_mask_L0_cross.sum()
            
preserve_mask_L0_random = numpy.ones((y_field_channels_L0, x_field_channels_L0)) > 0.5
#faktor=1 => all, 0 => nothing, 0.5 half the points
#faktor = 0.50
faktor = 0.30
preserve_mask_L0_random = numpy.random.uniform(size=(y_field_channels_L0, x_field_channels_L0))
#find a value faktor2, so that the proportion of ones in the mask is approx. faktor
i = 1
minf = 0.0
maxf = 1.0
while i < 12:
    faktor2= (minf + maxf) / 2.0
    tmpmat = (preserve_mask_L0_random <= faktor2)
    tmpsum = tmpmat.sum()
    if(tmpsum * 1.0 / preserve_mask_L0_random.size >  faktor):
        maxf = faktor2
    else:
        minf = faktor2
    i = i + 1
#
preserve_mask_L0_random = (preserve_mask_L0_random <= faktor2)
print "Size of random perceptive field is ", preserve_mask_L0_random.sum()


print "Drawing Masks"
im_small_c = im_small.copy()
im_small_c[0*y_field_channels_L0+0:0*y_field_channels_L0+y_field_channels_L0, 0:x_field_channels_L0][preserve_mask_L0] = 255
im_small_c[1*y_field_channels_L0+0:1*y_field_channels_L0+y_field_channels_L0, 0:x_field_channels_L0][preserve_mask_L0_circ] = 220
im_small_c[2*y_field_channels_L0+0:2*y_field_channels_L0+y_field_channels_L0, 0:x_field_channels_L0][preserve_mask_L0_prism] = 180
im_small_c[3*y_field_channels_L0+0:3*y_field_channels_L0+y_field_channels_L0, 0:x_field_channels_L0][preserve_mask_L0_rect] = 140
im_small_c[4*y_field_channels_L0+0:4*y_field_channels_L0+y_field_channels_L0, 0:x_field_channels_L0][preserve_mask_L0_cross] = 220
im_small_c[5*y_field_channels_L0+0:5*y_field_channels_L0+y_field_channels_L0, 0:x_field_channels_L0][preserve_mask_L0_random] = 255


#display first image
#Alternative: im1.show(command="xv")
a11 = plt.subplot(2,3,1)
plt.title("Last image in Sequence")
im_smalldisp = im_small_c
a11.imshow(im_smalldisp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)


print "Applying SFA for each mask"
sfa_number=2
for preserve_mask_L0_form in (preserve_mask_L0_circ, preserve_mask_L0_prism, preserve_mask_L0_rect, \
                              preserve_mask_L0_cross, preserve_mask_L0_random):
    print "About to create perceptive field of widht=%d, height=%d"%(x_field_channels_L0,y_field_channels_L0) 
    print "With a spacing of horiz=%d, vert=%d, and %d channels"%(x_field_spacing_L0,y_field_spacing_L0,in_channel_dim_L0)
    
    (mat_connections, lat_mat_L0) = compute_lattice_matrix_connections_with_input_dim(v1, v2, preserve_mask_L0_form, subimage_width, subimage_height, in_channel_dim_L0)
    print "matrix connections:"
    print mat_connections
    
    switchboard_L0 = mdp.hinet.Switchboard(subimage_width * subimage_height, mat_connections)
    switchboard_L0.connections
    
    #Create single SFA Node
    #Warning!!!! sfa_out_dim_L0 = 20
    sfa_out_dim_L0 = 10
    sfa_node_L0 = mdp.nodes.SFANode(input_dim=preserve_mask_L0_form.sum(), output_dim=sfa_out_dim_L0)
    
    #Create array of sfa_nodes (just one node, but clonned)
    sfa_layer_L0 = mdp.hinet.CloneLayer(sfa_node_L0, n_nodes=lat_mat_L0.size / 2)

    #Create Switchboard L1
    x_field_channels_L1=4
    y_field_channels_L1=4
    x_field_spacing_L1=2
    y_field_spacing_L1=2
    in_channel_dim_L1=sfa_out_dim_L0
    
    print "About to create intermediate layer widht=%d, height=%d"%(x_field_channels_L1,y_field_channels_L1) 
    print "With a spacing of horiz=%d, vert=%d, and %d channels"%(x_field_spacing_L1,y_field_spacing_L1,in_channel_dim_L1)
    switchboard_L1 = mdp.hinet.Rectangular2dSwitchboard(12, 6, x_field_channels_L1,y_field_channels_L1,x_field_spacing_L1,y_field_spacing_L1,in_channel_dim_L1)
    switchboard_L1.connections
    
    #Create L1 sfa node
    sfa_out_dim_L1 = 20
    sfa_node_L1 = mdp.nodes.SFANode(input_dim=switchboard_L1.out_channel_dim, output_dim=sfa_out_dim_L1)
    
    #Create array of sfa_nodes (just one node, but clonned)
    #This one should not be clonned!!!
    sfa_layer_L1 = mdp.hinet.CloneLayer(sfa_node_L1, n_nodes=switchboard_L1.output_channels)
    
    #Create L1 sfa node
    sfa_out_dim_L2 = 4
    sfa_node_L2 = mdp.nodes.SFANode(output_dim=sfa_out_dim_L2)
        
    #Join Switchboard and SFA layer in a single flow
    flow = mdp.Flow([switchboard_L0, sfa_layer_L0, switchboard_L1, sfa_layer_L1, sfa_node_L2])
    
    flow.train(subimages)
    sl_seq = flow.execute(subimages)
    
    a11 = plt.subplot(2,3,sfa_number)
    plt.title("SFA + Lattice. Single Unit L2 (left=slowest)")
    sl_seqdisp = sl_seq[:, range(0,sfa_out_dim_L2)]
    sl_seqdisp = scale_to(sl_seqdisp, sl_seq.mean(), sl_seq.max()-sl_seq.min(), 127.5, 255.0, scale_disp, 'tanh')
    a11.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
#    plt.xlabel("min=%.3f, max=%.3f, scale=%.3f\n e[]=" % (sl_seq.min(), sl_seq.max(), scale_disp)+str3(comp_eta(sl_seq)[0:4]))
    plt.xlabel("Mask Size=%d\n e[]=" % preserve_mask_L0_form.sum() + str3(comp_eta(sl_seq)[0:5]))

    sfa_number = sfa_number + 1

plt.show()
print "Finished showing!"


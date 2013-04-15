#Basic Experiments involving SFA applied to sequences of images
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 29 April 2009
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

#default scales (-1,1) into (0, 255)
#Normalized to [-scale/2, scale/2]
def scale_to(val, av_in=0.0, delta_in=2.0, av_out=127.5, delta_out=255.0, scale=1.0, transf='lin'):
    normalized = scale*(val - av_in) / delta_in
    if transf == 'lin':
        return normalized * delta_out + av_out
    elif transf == 'tanh':
        return numpy.tanh(normalized)*delta_out + av_out 
    else:
        raise Exception("Wrong transf in scale_to! (choose from 'lin', 'tanh'")


def wider(imag, scale_x=1):
    z = numpy.zeros((imag.shape[0], imag.shape[1]*scale_x))
    for i in range(imag.shape[1]):
        tmp = imag[:,i].reshape(imag.shape[0], 1)
        z[:,scale_x*i:scale_x*(i+1)] = tmp
    return z


def format_coord(x, y, numcols, numrows, width_factor=1.0, height_factor=1.0):
    col = int(x/width_factor+0.5)
    row = int(y/height_factor+0.5)
    if col>=0 and col<numcols and row>=0 and row<numrows:
        z = X[row,col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f'%(x, y)


#The following class was borrowed from '/mdp/nodes/expansion_nodes.py'
class _ExpansionNode(mdp.Node):
    def __init__(self, input_dim = None, dtype = None):
        super(_ExpansionNode, self).__init__(input_dim, None, dtype)
    def expanded_dim(self, dim):
        return dim
    def is_trainable(self):
        return False
    def is_invertible(self):
        return False
    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = self.expanded_dim(n)
    def _set_output_dim(self, n):
        msg = "Output dim cannot be set explicitly!"
        raise mdp.NodeException(msg)

#using the provided average and standard deviation
def gauss_noise(x, avg, std):
    return numpy.random.normal(avg, std, x.shape)

#Zero centered
def additive_gauss_noise(x, avg, std):
    return x + numpy.random.normal(0, std, x.shape)

    
class RandomizedMaskNode(mdp.Node):
    """Selectively mask some components of a random variable by 
    hiding them with arbitrary noise or by removing them
    Original code contributed by Alberto Escalante, inspired by NoiseNode
    """    
    def __init__(self, remove_mask=None, noise_amount_mask=None, noise_func = gauss_noise, noise_args = (0, 1),
                 noise_mix_func = None, input_dim = None, dtype = None):
        self.remove_mask = remove_mask
 
        self.noise_amount_mask = noise_amount_mask
        self.noise_func = noise_func
        self.noise_args = noise_args
        self.noise_mix_func = noise_mix_func 
        self.seen_samples = 0
        self.x_avg = None
        self.x_std = None
        self.type=dtype

        if remove_mask != None and input_dim == None:
            input_dim = remove_mask.size
        elif remove_mask == None and input_dim != None: 
            remove_mask = numpy.zeros(input_dim) > 0.5
        elif remove_mask != None and input_dim != None:
            if remove_mask.size != input_dim:
                err = "size of remove_mask and input_dim not compatible"
                raise Exception(err)
        else:
            err = "At least one of input_dim or remove_mask should be specified"
            raise Exception(err)
        
        if noise_amount_mask is None:
            print "Signal will be only the computed noise"
            self.noise_amount_mask = numpy.ones(input_dim)
        else:
            self.noise_amount_mask = noise_amount_mask 
            
        output_dim = remove_mask.size - remove_mask.sum()
        print "Output_dim should be:", output_dim 
        super(RandomizedMaskNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
                   
    def is_trainable(self):
        return True

    def _train(self, x):
        if self.x_avg == None:
            self.x_avg = numpy.zeros(self.input_dim, dtype=self.type)
            self.x_std = numpy.zeros(self.input_dim, dtype=self.type)
        new_samples = x.shape[0]
        self.x_avg = (self.x_avg * self.seen_samples + x.sum(axis=0)) / (self.seen_samples + new_samples)
        self.x_std = (self.x_std * self.seen_samples + x.std(axis=0)*new_samples) / (self.seen_samples + new_samples)
        self.seen_samples = self.seen_samples + new_samples
        
    def is_invertible(self):
        return False

    def _execute(self, x):
        vec_noise_func = numpy.vectorize(self.noise_func)

        print "computed X_avg=", self.x_avg
        print "computed X_std=", self.x_std
        noise_mat = self.noise_func(x, self.x_avg, self.x_std)
#        noise_mat = self._refcast(self.noise_func(*self.noise_args,
#                                                  **{'size': x.shape}))
        print "Noise_amount_mask:", self.noise_amount_mask
        print "Noise_mat:", noise_mat
        noisy_signal = (1.0 - self.noise_amount_mask) * x + self.noise_amount_mask * noise_mat
        return noisy_signal[:, self.remove_mask]


def extend_channel_mask_to_signal_mask(input_dim, channel_mask):
    channel_size = channel_mask.size
    rep = input_dim / channel_size
    if input_dim % channel_size != 0:
        err="incompatible channel_mask length and input_dim"
        raise Exception(err)  
    res = channel_mask.copy()
    for iter in range(rep-1):
        res = numpy.concatenate((res, channel_mask))
    return res

print "************************************************************"
print "Testing RandomizedMaskNode:"
print "        Selective removal/randomization of signal elements"
print "************************************************************"
input_dim = 6
channel_size = 3
var_size = (1, input_dim)
remove_channel_mask = numpy.arange(channel_size) < channel_size / 2.5
print "Channel Mask:", remove_channel_mask
remove_mask = extend_channel_mask_to_signal_mask(input_dim, remove_channel_mask)
print "Remove Mask:", remove_mask

#Only noise
noise_amount_mask = numpy.random.normal(0.5, 0.5, var_size)
print "Noise_amount Mask:", noise_amount_mask

avg = numpy.arange(input_dim) * 10 / input_dim + 1
std = numpy.arange(input_dim) * 2.0 / input_dim + 0.01

print "X_ Avg:", avg
print "X_ Std:", std

randmasknode = RandomizedMaskNode(remove_mask, noise_amount_mask)

num_samples = 10
size_samples = (num_samples, input_dim)
x = numpy.random.normal(avg, std, size_samples)
print "x=", x

randmasknode.train(x)
randmasknode.stop_training()

z1 = randmasknode(x)
print "z1=randmasknode(x)=", z1

y = numpy.random.normal(0, 0.1, size_samples)
print "y=", y
 
z2 = randmasknode(y)
print "z2=randmasknode(y)=", z2

def identity(x):
    return x

def abs_dif(x1, x2):
    return numpy.abs(x1 - x2)

def abs_sum(x1, x2):
    return numpy.abs(x1 + x2)

def multiply(x1, x2):
    return x1 * x2

def sqrt_abs_sum(x1, x2):
    return numpy.sqrt(numpy.abs(x1+x2))

def sqrt_abs_dif(x1, x2):
    return numpy.sqrt(numpy.abs(x1-x2))

#Expansion with terms: f(x1,x1), f(x1,x2), ... f(x1,xn), f(x2,x2), ... f(xn,xn)
#If reflexive=True, skip terms f(xj, xj)
def pairwise_expansion(x, func, reflexive=True):
    x_height, x_width = x.shape
    if reflexive==True:
        k=0
        out = numpy.zeros((x_height, x_width*(x_width+1)/2))
    else:
        k=1
        out = numpy.zeros((x_height, x_width*(x_width-1)/2))    
    mask = numpy.triu(numpy.ones((x_width,x_width)), k) > 0.5
    for i in range(0, x_height):
        y1 = x[i].reshape(x_width, 1)
        y2 = x[i].reshape(1, x_width)
        yexp = func(y1, y2)
#        print "yexp=", yexp
        out[i] = yexp[mask]
    return out    

def pair_abs_dif_ex(x):
    return pairwise_expansion(x, abs_dif, reflexive=False)

def pair_abs_sum_ex(x):
    return pairwise_expansion(x, abs_sum)

def pair_prod_ex(x):
    return pairwise_expansion(x, multiply)

def pair_sqrt_abs_sum_ex(x):
    return pairwise_expansion(x, sqrt_abs_sum)

def pair_sqrt_abs_dif_ex(x):
    return pairwise_expansion(x, sqrt_abs_dif, reflexive=False)


class GeneralExpansionNode(mdp.Node):
    def __init__(self, funcs, input_dim = None, dtype = None):
        self.funcs = funcs
        super(GeneralExpansionNode, self).__init__(input_dim, dtype)
    def expanded_dim(self, n):
        exp_dim=0
        x = numpy.zeros((1,n))
        for func in self.funcs:
            outx = func(x)
            exp_dim += outx.shape[1]
        return exp_dim
    def is_trainable(self):
        return False
    def is_invertible(self, x):
        return False
    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = self.expanded_dim(n)
    def _execute(self, x):
        out = []
        for func in self.funcs:
            if out==[]:
                out = func(x)
            else:
                out = numpy.concatenate((out, func(x)), axis=1)
        return out


def compute_latice_matrix(v1, v2, mask, x_in_channels, y_in_channels, in_channel_dim=1, n0_1 = 0, n0_2 = 0, wrap_x= False, wrap_y= False, input_dim = None, dtype = None, ignore_cover = True):
    if v1[1] != 0 | v1[0] <= 0 | v2[0] < 0 | v2[1] <= 0:
        err = "v1 must be horizontal: v1[0] > 0, v1[1] = 0, v2[0] >= 0, v2[1] > 0"
        raise Exception(err)  
#assume no wrapping 
    image = numpy.array(range(0, x_in_channels * y_in_channels))
    image.reshape((y_in_channels, x_in_channels))
    sub_image = numpy.array(range(0, mask.shape[0] * mask.shape[1]))
    sub_image.reshape((mask.shape[0], mask.shape[1]))
    mask_i = mask.astype("int")
    mask_height, mask_width = mask.shape
    out_channel_dim = mask_i.sum()
#    print "Mask shape is ", mask.shape
    
    mat_height = (y_in_channels - mask.shape[0])/v2[1] + 1
    mat_width = (x_in_channels-mask.shape[1])/v1[0] + 1
    
    mat = numpy.ones((mat_height, mat_width, 2)) * -1
#Create Index Matrix, -1 entries equal empty cell
#    print "Mat shape is ", mat.shape
    ind_y = 0    
    for iy in range(0, mat_height):
            #x,y are real subimage positions
            #ix, iy are the coefficients of x,y in base v1 and v2
            #ind_y, ind_x are the indices in the matrix mat that contains the centers (upper-left corners) of each subimage
        y = iy * v2[1]
        min_ix = -1 * numpy.int(iy * v2[0] / v1[0])
        max_ix = numpy.floor( (x_in_channels - mask.shape[1] - iy *v2[0]) * 1.0 /  v1[0])
        max_ix = numpy.int(max_ix)
        ind_x = 0
        for ix in range(min_ix, max_ix + 1):
            x = iy *v2[0] + ix * v1[0]
 #           print "value of ind_x, ind_y = ", (ind_x, ind_y)
 #           print "Adding Point (", x, ", ", y, ")"
            mat[ind_y, ind_x] = (x, y)
            ind_x = ind_x + 1
        ind_y = ind_y + 1
    return mat


print "***********************************************************************"
print "Testing compute_latice_matrix."
print "    Computes a matrix with the coefficients for each receptive field"
print "***********************************************************************"

v1 = (4,0)
v2 = (2,5)
mask = numpy.ones((4,4))
print "first latice"
compute_latice_matrix(v1, v2, mask, 10, 12)
print "second latice"
compute_latice_matrix(v1, v2, mask, 9, 12)
print "third latice"
compute_latice_matrix(v1, v2, mask, 10, 9)
print "fourth latice"
compute_latice_matrix(v1, v2, mask, 10, 8)


def compute_lattice_matrix_connections(self, v1, v2, mask, x_in_channels, y_in_channels, in_channel_dim=1):
        mat = compute_latice_matrix(v1, v2, mask, x_in_channels, y_in_channels)             
       
        image_positions = numpy.array(range(0, x_in_channels * y_in_channels * in_channel_dim))
        image_positions = image_positions.reshape(x_in_channels, y_in_channels, in_channel_dim)

        if in_channel_dim > 1:
            err = "Warning, feature not supported: in_channel_dim > 1"
            raise Exception(err)
        
        mask_indices = image_positions[0:mask.shape[0], 0:mask.shape[1]][mask].flatten()
        
        connections = []
        for ind_y in range(mat.shape[0]):
            for ind_x in range(mat.shape[1]):
                if(mat[ind_y, ind_x][0] != -1):
                    connections.append(mask_indices + (mat[ind_y, ind_x][0] + mat[ind_y, ind_x][1]*x_in_channels) )
                else:
                    print "Void entry in lattice_matrix skipped"

        return connections



#Note: generalize with other functions!!!
#Careful with the sign used!!!
class BinomialAbsoluteExpansionNode(mdp.Node):
    def expanded_dim(self, n):
        return n + n*(n+1)/2
    def is_trainable(self):
        return False
    def is_invertible(self, x):
        return False
    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = self.expanded_dim(n)
    def _execute(self, x):
        out = numpy.concatenate((x, pairwise_expansion(x, abs_sum)), axis=1)
        return out

#Obsolete, there is a method that does this... (with t-1 as denominator)
def comp_variance(x):
    t, num_vars = x.shape
    delta = numpy.zeros(num_vars)
    for row in x:
        delta = delta + numpy.square(row)
    delta = delta / (t)
    return delta

def comp_delta(x):
    t, num_vars = x.shape
    delta = numpy.zeros(num_vars)
    xderiv = x[1:, :]-x[:-1, :]
    for row in xderiv:
        delta = delta + numpy.square(row)
    delta = delta / (t-1)
    return delta

def comp_eta(x):
    t, num_vars = x.shape
    return t/(2*numpy.pi) * numpy.sqrt(comp_delta(x))

def str2(x):
    c=""
    for w in x:
        if c == "":
            c+="%.2f" % w
        else:
            c+=",%.2f" % w
    return c


#Renaming filenames => Fix perl script
#image_files = glob.glob(im_seq_base_dir + "/*p?_*tif")
#image_files
#for i in range(0,len(image_files)+0):
# os.rename(prefix + str(i) + sufix, prefix+"%03d"%i + sufix)
#
#image_files = glob.glob(im_seq_base_dir + "/*p??_*tif")
#for i in range(10,len(image_files)+10):
# os.rename(prefix + str(i) + sufix, prefix+"%03d"%i + sufix)


print "************************************************************"
print "Creating a 3 Level SFA Hierarchy:"
print "        Using traditional MDP Nodes"
print "************************************************************"

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
a11 = plt.subplot(2,2,2)
plt.title("Last image in Sequence")
im_smalldisp = im_small
a11.imshow(im_smalldisp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)


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
sfa_out_dim_L0 = 20
sfa_node_L0 = mdp.nodes.SFANode(input_dim=switchboard_L0.out_channel_dim, output_dim=sfa_out_dim_L0)

#Create array of sfa_nodes (just one node, but clonned)
sfa_layer_L0 = mdp.hinet.CloneLayer(sfa_node_L0, n_nodes=switchboard_L0.output_channels)



#Create Switchboard L1
x_field_channels_L1=4
y_field_channels_L1=4
x_field_spacing_L1=2
y_field_spacing_L1=2
in_channel_dim_L1=20

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

a11 = plt.subplot(2,2,3)
plt.title("SFA Single Unit L2 (left=slowest)")
sl_seqdisp = sl_seq[:, range(0,sfa_out_dim_L2)]
sl_seqdisp = scale_to(sl_seqdisp, sl_seq.mean(), sl_seq.max()-sl_seq.min(), 127.5, 255.0, scale_disp, 'tanh')
a11.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (sl_seq.min(), sl_seq.max(), scale_disp)+str2(comp_eta(sl_seq)[0:4]))


plt.show()
print "Finished!"


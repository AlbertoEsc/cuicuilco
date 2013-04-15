#Basic Examples of PIL, MDP, and the use of PCA and ICA over the rows of two face images
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 31 March 2009
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

#x = numpy.array([range(-4,4), range(8,16), range(16,24)]) * 1.0
#baenode = BinomialAbsoluteExpansionNode()
#baenode.execute(x)
x = numpy.array([[ -4.,  -3.,  -2.,  -1.,   0.,   1.,   2.,   3.],
       [  8.,   9.,  10.,  11.,  12.,  13.,  14.,  15.],
       [ 16.,  17.,  18.,  19.,  20.,  21.,  22.,  23.]])

comp_delta(x)

#factor to enlarge low-dim images
wide_fac = 10
scale_disp = 3
subimage_width = 12
#pixelsampling: 1=each pixel, 2= one yes and one not
subimage_pixelsampling= 5

#Open first graylevel image
im1ori = Image.open("/home/escalafl/datasets/face/face1.jpg")
#im1ori = Image.open("/home/escalafl/datasets/face/test/cnn1260_beta.gif")
im1ori = im1ori.convert("L")
im1 = numpy.asarray(im1ori)
im1width, im1height = im1ori.size
subimage_height = im1height

subimage_first_column=130
im1small = im1[:, subimage_first_column:(subimage_first_column+subimage_width*subimage_pixelsampling):subimage_pixelsampling].astype(float)

#Create Figure
f1 = plt.figure(1)
plt.suptitle("Experimenting with SFA...")

#display first image
#Alternative: im1.show(command="xv")
a11 = plt.subplot(2,4,1)
plt.title("Original Image 1 small")
im1smalldisp = im1small
a11.imshow(im1smalldisp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
#plt.axis([10, 20, 30, 60])
#def format_coord11(x, y):
#    return format_coord(x,y,subimage_width, im1height, 100)
#
#a11.format_coord = format_coord11
#'v must contain [xmin xmax ymin ymax]'
# a11.axis([0, subimage_width-1, subimage_height-1,0])

#SIMPLE SLOW FEATURE ANALYSIS
#Create SFA Node, train on im1small
sfanode = mdp.nodes.SFANode()
sfanode.train(im1small)
sfanode.stop_training()
sfanode.output_dim
#Note: if array has size a x b (im1[a,b]), then b is the output dimension (variables) and a are observations
#Note: if the observations are not many, you will find trouble when computing the eigenvalues

#Apply SFA to im1small
sl = sfanode.execute(im1small)


#display slowest components SFA
a12 = plt.subplot(2,4,2)
plt.title("SFA: (slowest=left)")
sldisp = scale_to(sl, sl.mean(), sl.max()-sl.min(), 127.5, 255.0, scale_disp, 'tanh')
a12.imshow(sldisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (sl.min(), sl.max(), scale_disp)+str2(comp_eta(sl)[0:3]))


#SLOW FEATURE ANALYSIS ON QUADRATIC EXPANSION
#Create SFA2 Node, train on im1small
sfaqnode = mdp.nodes.SFA2Node(output_dim=sfanode.output_dim)
sfaqnode.train(im1small)
sfaqnode.stop_training()
sfaqnode.output_dim
#Note: if array has size a x b (im1[a,b]), then b is the output dimension (variables) and a are observations
#Note: if the observations are not many, you will find trouble when computing the eigenvalues
slq = sfaqnode.execute(im1small)

#display slowest components SFA2
a21 = plt.subplot(2,4,3)
plt.title("SFA2: (slowest=left)")
slqdisp = scale_to(slq, slq.mean(), slq.max()-slq.min(), 127.5, 255.0, scale_disp, 'tanh')
a21.imshow(slqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (slq.min(), slq.max(), scale_disp)+str2(comp_eta(slq)[0:3]))


#COMPUTE ZERO MEAN SAMPLES: centered
#AverageRemoved + BinomialAbsoluteExpansion + SFA
#Compute averages:
transp = im1small.transpose()
centered = transp.copy()
for i in range(0, subimage_width):
    row = transp[i]
    av = numpy.average(row)
#    print "av=%f"%av
    centered[i] = row - av

centered = centered.transpose()


#Create BinomialAbsoluteExpansionNode
#baenode = BinomialAbsoluteExpansionNode()
#baedataav = baenode.execute(centered) * 1.0

#SLOW FEATURE ANALYSIS ON GENERIC EXPANSIONS 1
funcs1 = [identity, pair_sqrt_abs_sum_ex]
genexpnode1 = GeneralExpansionNode(funcs1)
id__sqr_abs_sum_data = genexpnode1.execute(centered) * 1.0

#feed new SFANode with expanded data
sfexpnode1 = mdp.nodes.SFANode(output_dim=sfanode.output_dim)
sfexpnode1.train(id__sqr_abs_sum_data)
sfexpnode1.stop_training()
sfexpnode1.output_dim

slexp1 = sfexpnode1.execute(id__sqr_abs_sum_data)

#display slowest components zero Mean + BinomialAbsoluteExpansion + SFA
a22 = plt.subplot(2,4,4)
plt.title("SFA: ID+SQR_ABS_SUM")
slexp1disp = scale_to(slexp1, slexp1.mean(), slexp1.max()-slexp1.min(), 127.5, 255.0, scale_disp, 'tanh')
a22.imshow(slexp1disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (slexp1.min(), slexp1.max(), scale_disp)+str2(comp_eta(slexp1)[0:3]))


#SLOW FEATURE ANALYSIS ON GENERIC EXPANSIONS 2
funcs2 = [identity, pair_prod_ex, pair_sqrt_abs_dif_ex, pair_sqrt_abs_sum_ex]
genexpnode2 = GeneralExpansionNode(funcs2)
id__prod_data = genexpnode2.execute(centered) * 1.0

#feed new SFANode with expanded data
sfexpnode2 = mdp.nodes.SFANode(output_dim=sfanode.output_dim)
sfexpnode2.train(id__prod_data)
sfexpnode2.stop_training()
sfexpnode2.output_dim

slexp2 = sfexpnode2.execute(id__prod_data)

#display slowest components zero Mean + BinomialAbsoluteExpansion + SFA
a22 = plt.subplot(2,4,5)
plt.title("SFA: ID+PROD+SQRT_ABS_SUM & DIF")
slexp2disp = scale_to(slexp2, slexp2.mean(), slexp2.max()-slexp2.min(), 127.5, 255.0, scale_disp, 'tanh')
a22.imshow(slexp2disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (slexp2.min(), slexp2.max(), scale_disp)+ str2(comp_eta(slexp2)[0:3]))
print "variance of slexp2 is:"
print comp_variance(slexp2)[0:3]

#SLOW FEATURE ANALYSIS ON GENERIC EXPANSIONS 3
funcs3 = [identity, pair_prod_ex, pair_sqrt_abs_sum_ex]
genexpnode3 = GeneralExpansionNode(funcs3)
exp3_data = genexpnode3.execute(centered) * 1.0

#feed new SFANode with expanded data
sfexpnode3 = mdp.nodes.SFANode(output_dim=sfanode.output_dim)
sfexpnode3.train(exp3_data)
sfexpnode3.stop_training()
sfexpnode3.output_dim

slexp3 = sfexpnode3.execute(exp3_data)

#display slowest components zero Mean + BinomialAbsoluteExpansion + SFA
a22 = plt.subplot(2,4,6)
plt.title("SFA: ID+PROD+SQRT_ABS_SUM")
slexp3disp = scale_to(slexp3, slexp3.mean(), slexp3.max()-slexp3.min(), 127.5, 255.0, scale_disp, 'tanh')
a22.imshow(slexp3disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (slexp3.min(), slexp3.max(), scale_disp)+str2(comp_eta(slexp3)[0:3]))

#WHITHENING + BinomialAbsoluteExpansion + SFA
#Whiten Image:
whitenode = mdp.nodes.WhiteningNode()
whitenode.train(im1small)
whitenode.stop_training()
whitenode.output_dim
whitedata = whitenode.execute(im1small)

#Create BinomialAbsoluteExpansionNode
baenode = BinomialAbsoluteExpansionNode()
baedata = baenode.execute(whitedata)

#feed new SFANode with expanded data
sfaexpnode = mdp.nodes.SFANode(output_dim=sfanode.output_dim)
sfaexpnode.train(baedata)
sfaexpnode.stop_training()
sfaexpnode.output_dim

slexp = sfaexpnode.execute(baedata)

#display slowest components HITHENING + BinomialAbsoluteExpansion + SFA
a22 = plt.subplot(2,4,7)
plt.title("SFA, White, ID+ABS_SUM")
slexpdisp = scale_to(slexp, slexp.mean(), slexp.max()-slexp.min(), 127.5, 255.0, scale_disp, 'tanh')
a22.imshow(slexpdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (slexp.min(), slexp.max(), scale_disp)+str2(comp_eta(slexp)[0:3]))
print "variance of slexp is:"
print comp_variance(slexp)[0:3]


#Create PCA Node, based on info from im1, show projection matrix
pcanode1 = mdp.nodes.PCANode(dtype='float32', svd=True)
pcanode1.train(im1small)
pcanode1.stop_training()
pcanode1.output_dim
avg = pcanode1.avg
v=pcanode1.get_projmatrix()

#Apply PCA to im1: y1 = PCA(im1)
y1=pcanode1.execute(im1small)
y1.shape

a23 = plt.subplot(2,4,8)
plt.title("y1 = PCAN(im1small), principal=left")
scale_disp = 3

#WARNING! y1 IS NORMALIZED
y1 = y1 - y1.mean(axis=0)
y1 = y1 / numpy.sqrt(comp_variance(y1))

y1disp = scale_to(y1, y1.mean(), y1.max()-y1.min(), 127.5, 255.0, scale_disp, 'tanh')
a23.imshow(y1disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (y1.min(), y1.max(), scale_disp)+str2(comp_eta(y1)[0:3]))

print "eta of y1 (normalized to var(y1) = vector of ones, and zero mean) is:"
print comp_eta(y1)[0:3]

########################################################################################
#SECOND IMAGE
#Open second graylevel image
im2ori = Image.open("/home/escalafl/datasets/face/face2.jpg")
#im2ori = Image.open("/home/escalafl/datasets/face/test/cnn1714.gif")
im2ori = im2ori.convert("L")
im2 = numpy.asarray(im2ori)
im2width, im2height = im2ori.size

subimage_first_column=130
#im2small = im2[:,subimage_first_column:subimage_first_column+subimage_width].astype(float)
im2small = im2[:, subimage_first_column:(subimage_first_column+subimage_width*subimage_pixelsampling):subimage_pixelsampling].astype(float)

#Create Figure
f2 = plt.figure(2)
plt.suptitle("Experimenting with SFA (Second Image)...")

#display first image
#Alternative: im2.show(command="xv")
a11 = plt.subplot(2,4,1)
plt.title("Original Image 2 small")
scale_disp = 3
im2smalldisp = im2small
a11.imshow(im2smalldisp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)

#Apply SFA to im2small
sl2 = sfanode.execute(im2small)

#display slowest components SFA
a12 = plt.subplot(2,4,2)
plt.title("SFA: (slowest=left)")
sl2disp = scale_to(sl2, sl2.mean(), sl2.max()-sl2.min(), 127.5, 255.0, scale_disp, 'tanh')
a12.imshow(sl2disp.clip(0,255), aspect='auto',interpolation='nearest',origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (sl2.min(), sl2.max(), scale_disp)+str2(comp_eta(sl2)[0:3]))


#Note: if array has size a x b (im2[a,b]), then b is the output dimension (variables) and a are observations
#Note: if the observations are not many, you will find trouble when computing the eigenvalues
slq2 = sfaqnode.execute(im2small)

#display slowest components SFA2
a21 = plt.subplot(2,4,3)
plt.title("SFA2: (slowest=left)")
slq2disp = scale_to(slq2, slq2.mean(), slq2.max()-slq2.min(), 127.5, 255.0, scale_disp, 'tanh')
a21.imshow(slq2disp.clip(0,255), aspect='auto',interpolation='nearest',origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (slq2.min(), slq2.max(), scale_disp)+str2(comp_eta(slq2)[0:3]))


#COMPUTE ZERO MEAN SAMPLES FOR SECOND IMAGE: centered2
#Compute averages:
transp = im2small.transpose()
centered2 = transp.copy()
for i in range(0, subimage_width):
    row = transp[i]
    av = numpy.average(row)
#    print "av=%f"%av
    centered2[i] = row - av

centered2 = centered2.transpose()


#Create BinomialAbsoluteExpansionNode
#baenode = BinomialAbsoluteExpansionNode()
#baedataav = baenode.execute(centered) * 1.0

#SLOW FEATURE ANALYSIS ON GENERIC EXPANSIONS 1
id__sqr_abs_sum_data2 = genexpnode1.execute(centered2) * 1.0

#feed new SFANode with expanded data
slexp1_2 = sfexpnode1.execute(id__sqr_abs_sum_data2)

#display slowest components zero Mean + BinomialAbsoluteExpansion + SFA
a22 = plt.subplot(2,4,4)
plt.title("SFA: ID+SQR_ABS_SUM")
slexp1_2disp = scale_to(slexp1_2, slexp1_2.mean(), slexp1_2.max()-slexp1_2.min(), 127.5, 255.0, scale_disp, 'tanh')
a22.imshow(slexp1_2disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (slexp1_2.min(), slexp1_2.max(), scale_disp)+str2(comp_eta(slexp1_2)[0:3]))


#SLOW FEATURE ANALYSIS ON GENERIC EXPANSIONS 2
id__prod_data2 = genexpnode2.execute(centered2) * 1.0

#feed new SFANode with expanded data
slexp2_2 = sfexpnode2.execute(id__prod_data2)

#display slowest components zero Mean + BinomialAbsoluteExpansion + SFA
a22 = plt.subplot(2,4,5)
plt.title("SFA: ID+PROD+SQRT_ABS_SUM & DIF")
slexp2_2disp = scale_to(slexp2_2, slexp2_2.mean(), slexp2_2.max()-slexp2_2.min(), 127.5, 255.0, scale_disp, 'tanh')
a22.imshow(slexp2_2disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (slexp2_2.min(), slexp2_2.max(), scale_disp)+str2(comp_eta(slexp2_2)[0:3]))


#SLOW FEATURE ANALYSIS ON GENERIC EXPANSIONS 3
exp3_data2 = genexpnode3.execute(centered2) * 1.0

#feed new SFANode with expanded data
slexp3_2 = sfexpnode3.execute(exp3_data2)

#display slowest components zero Mean + BinomialAbsoluteExpansion + SFA
a22 = plt.subplot(2,4,6)
plt.title("SFA: ID+PROD+SQRT_ABS_SUM")
slexp3_2disp = scale_to(slexp3_2, slexp3_2.mean(), slexp3_2.max()-slexp3_2.min(), 127.5, 255.0, scale_disp, 'tanh')
a22.imshow(slexp3_2disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (slexp3_2.min(), slexp3_2.max(), scale_disp)+str2(comp_eta(slexp3_2)[0:3]))

#WHITHENING + BinomialAbsoluteExpansion + SFA
#Whiten Image:
whitedata2 = whitenode.execute(im2small)

#Create BinomialAbsoluteExpansionNode
baedata2 = baenode.execute(whitedata2)

#feed new SFANode with expanded data
slexp2 = sfaexpnode.execute(baedata2)

#display slowest components HITHENING + BinomialAbsoluteExpansion + SFA
a22 = plt.subplot(2,4,7)
plt.title("SFA, White, ID+ABS_SUM")
slexp2disp = scale_to(slexp2, slexp2.mean(), slexp2.max()-slexp2.min(), 127.5, 255.0, scale_disp, 'tanh')
a22.imshow(slexp2disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (slexp2.min(), slexp2.max(), scale_disp)+str2(comp_eta(slexp2)[0:3]))
print "variance of slexp2 is:"
print comp_variance(slexp2)[0:3]

#Apply PCA to im2: y2 = PCA(im2)
y2=pcanode1.execute(im2small)
y2.shape
#WARNING y2 is normalized!!
y2 = y2 - y2.mean(axis=0)
y2 = y2 / numpy.sqrt(comp_variance(y2))

a22 = plt.subplot(2,4,8)
plt.title("y1 = PCAN(im2small), principal=left")
scale_disp = 3
y2disp = scale_to(y2, y2.mean(), y2.max()-y2.min(), 127.5, 255.0, scale_disp, 'tanh')
a22.imshow(y2disp.clip(0,255), aspect='auto',interpolation='nearest',origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (y2.min(), y2.max(), scale_disp)+str2(comp_eta(y2)[0:3]))
print "variance of y2 is:"
print comp_variance(y2)[0:3]
print "eta of y2 (normalized to var(y2) = vector of ones) is:"

#yy2 = y2 / numpy.sqrt(comp_variance(y2))
print comp_eta(y2)[0:3]


#Renaming filenames
#image_files = glob.glob(im_seq_base_dir + "/*p?_*tif")
#image_files
#for i in range(0,len(image_files)+0):
# os.rename(prefix + str(i) + sufix, prefix+"%03d"%i + sufix)
#
#image_files = glob.glob(im_seq_base_dir + "/*p??_*tif")
#for i in range(10,len(image_files)+10):
# os.rename(prefix + str(i) + sufix, prefix+"%03d"%i + sufix)

#SEQUENCE OF IMAGES
image_width  = 256
image_height = 192

subimage_width  = image_width/5
subimage_height = image_height/2 
#pixelsampling: 1=each pixel, 2= one yes and one not
subimage_pixelsampling= 1

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
f1 = plt.figure(3)
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


#Create Switchboard
switchboard = mdp.hinet.Rectangular2dSwitchboard(subimage_width, subimage_height,x_field_channels=6,y_field_channels=6,x_field_spacing=3,y_field_spacing=3,in_channel_dim=1)
switchboard.connections

#Create single SFA Node
sfa_out_dim = 20
sfa_node = mdp.nodes.SFANode(input_dim=switchboard.out_channel_dim, output_dim=sfa_out_dim)


sfa_layer = mdp.hinet.CloneLayer(sfa_node, n_nodes=switchboard.output_channels)
#Join Switchboard and SFA layer in a single flow
flow = mdp.Flow([switchboard, sfa_layer])


flow.train(subimages)
sl_seq = flow.execute(subimages)

a11 = plt.subplot(2,2,3)
plt.title("SFA Single Unit (left=slowest)")
sl_seqdisp = sl_seq[:, range(0,36)]
sl_seqdisp = scale_to(sl_seqdisp, sl_seq.mean(), sl_seq.max()-sl_seq.min(), 127.5, 255.0, scale_disp, 'tanh')
a11.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (sl_seq.min(), sl_seq.max(), scale_disp)+str2(comp_eta(sl_seq)[0:3]))


plt.show()



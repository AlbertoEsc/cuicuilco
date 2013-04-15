import numpy
from numpy import (cos, sin)
import mdp
import matplotlib as mpl
import matplotlib.pyplot as plt

def neg_expo(x, expo):
    signs = numpy.sign(x)
    y = numpy.abs(x)**expo * signs
    return y

import sys
sys.path.append("/home/escalafl/workspace/cuicuilco/src")
from sfa_libs import ndarray_to_string

num_steps = 1000
t = numpy.linspace(0, numpy.pi, num_steps)
ramp = numpy.linspace(-3,3,num_steps)
numpy.random.seed(123456)

n1 = numpy.random.normal(size=(num_steps))
n1[1:] = (n1[1:] + n1[0:-1])/2
n1[1:] = (n1[1:] + n1[0:-1])/2
n1[1:] = (n1[1:] + n1[0:-1])/2
n1[1:] = (n1[1:] + n1[0:-1])/2
n1[1:] = (n1[1:] + n1[0:-1])/2
n1[1:] = (n1[1:] + n1[0:-1])/2
n1[1:] = (n1[1:] + n1[0:-1])/2
n1[1:] = (n1[1:] + n1[0:-1])/2
n1[1:] = (n1[1:] + n1[0:-1])/2
n1[1:] = (n1[1:] + n1[0:-1])/2
s1 = 0.5*cos(11*t) - 0.2*cos(17*t) + 0.1*cos(20*t) - 1* (t-2.4)**2/15 + 1 * 0.25*cos(10*(t-0.9)**2) + 1*0.08*n1 
#+ 0.4 * cos(cos(t)*5*t)
s1 = (s1*1.0-s1.mean())/s1.std()
#s1 = s1**3
#s1 = (s1*1.0-s1.mean())/s1.std()

#bugg!!!!
n2 = numpy.random.normal(size=(num_steps))
n2[1:] = (n2[1:] + n2[0:-1])/2
n2[1:] = (n2[1:] + n2[0:-1])/2
n2[1:] = (n2[1:] + n2[0:-1])/2
n2[1:] = (n2[1:] + n2[0:-1])/2
n2[1:] = (n2[1:] + n2[0:-1])/2
n2[1:] = (n2[1:] + n2[0:-1])/2
n2[1:] = (n2[1:] + n2[0:-1])/2
n2[1:] = (n2[1:] + n2[0:-1])/2
n2[1:] = (n2[1:] + n2[0:-1])/2
n2[1:] = (n2[1:] + n2[0:-1])/2
n2[1:] = (n2[1:] + n2[0:-1])/2
n2[1:] = (n2[1:] + n2[0:-1])/2
n2[1:] = (n2[1:] + n2[0:-1])/2
n2[1:] = (n2[1:] + n2[0:-1])/2
n2[1:] = (n2[1:] + n2[0:-1])/2
n2[1:] = (n2[1:] + n2[0:-1])/2
n2[1:] = (n2[1:] + n2[0:-1])/2
n2[1:] = (n2[1:] + n2[0:-1])/2
s2 = 0.5*cos(13*t) - 0.2*sin(13*t) + 0.1*cos(23*t) - 1 * (t-2.0)**2/25 + 1*0.05*cos(8*(t-1.2)**2)+1*0.04*n2+0*0.04*cos((cos(10*t)+1)*20*t)
s2 = (s2*1.0 - s2.mean())/s2.std()
print "AAA s2=", s2
s2 = neg_expo(s2,3)

#(s2*1.0 ** 3)
#s2 = 0.5*cos(13*t) + (t-1.0)**3/250 + 0.4*sin((t-2.4)**2/5)+0.1*n2
print "BBB s2=", s2
s2 = (s2*1.0-s2.mean())/s2.std()
print "CCC s2=", s2

s3 = s2 - s1 * numpy.dot(s1,s2)/num_steps
s3 = (s3*1.0-s3.mean())/s3.std()
print "DDD s3=", s2

print "s1 dot s3 =", numpy.dot(s1, s3)
print "s1 dot s2 =", numpy.dot(s1, s2)

del s2


s1 = s1.reshape((num_steps,1))
#s2 = s2.reshape((num_steps,1))
s3 = s3.reshape((num_steps,1))


poly_exp_node = mdp.nodes.PolynomialExpansionNode(20)
sfa_node_1 = mdp.nodes.SFANode(output_dim=20)
sfa_node_2 = mdp.nodes.SFANode(output_dim=20)

poly_exp_1 = poly_exp_node.execute(s1)
sfa_node_1.train(poly_exp_1)
sfa_1 = sfa_node_1.execute(poly_exp_1)

poly_exp_2 = poly_exp_node.execute(s3)
sfa_node_2.train(poly_exp_2)
sfa_2 = sfa_node_2.execute(poly_exp_2)


def comp_delta(x):
    t, num_vars = x.shape
    delta = numpy.zeros(num_vars)
    xderiv = x[1:, :]-x[:-1, :]
    for row in xderiv:
        delta = delta + numpy.square(row)
    delta = delta / (t-1)
    return delta

delta_1 = comp_delta(s1)[0]
delta_2 = comp_delta(s3)[0]
print "orig delta=", delta_1, delta_2

ndarray_to_string(numpy.array([delta_1, delta_2]), prefix="#", col_sep=", ", row_sep="\n", out_filename = "delta_input_signals.txt")


#print delta
#print sfa_node_1.d
#print sfa_node_2.d
ndarray_to_string(sfa_node_1.d, prefix="#", col_sep=", ", row_sep="\n", out_filename = "delta1.txt")
ndarray_to_string(sfa_node_2.d, prefix="#", col_sep=", ", row_sep="\n", out_filename = "delta2.txt")

orig1=numpy.zeros((num_steps,2))
orig1[:,0] = numpy.arange(num_steps)
orig1[:,1] = s1[:,0]

orig2=numpy.zeros((num_steps,2))
orig2[:,0] = numpy.arange(num_steps)
orig2[:,1] = s3[:,0]

#ndarray_to_string(orig1, prefix="#", col_sep=", ", row_sep="\n", out_filename = "orig1.txt")
#ndarray_to_string(orig2, prefix="#", col_sep=", ", row_sep="\n", out_filename = "orig2.txt")

#for i in range(20):
#    signal1=numpy.zeros((num_steps,2))
#    signal1[:,0] = numpy.arange(num_steps)
#    signal1[:,1] = sfa_1[:,i]
#    ndarray_to_string(signal1, prefix="#", col_sep=", ", row_sep="\n", out_filename = "sfa1_%d.txt"%i)
#
#for i in range(20):
#    signal2=numpy.zeros((num_steps,2))
#    signal2[:,0] = numpy.arange(num_steps)
#    signal2[:,1] = sfa_2[:,i]
#    ndarray_to_string(signal2, prefix="#", col_sep=", ", row_sep="\n", out_filename = "sfa2_%d.txt"%i)

#Computing / Storing whole signal t, orig, signal2
orig_n_sfa1n2 = orig1 + 0.0
orig_n_sfa1n2 = numpy.append(orig_n_sfa1n2, sfa_1, axis=1)
orig_n_sfa1n2 = numpy.append(orig_n_sfa1n2, orig2, axis=1)
orig_n_sfa1n2 = numpy.append(orig_n_sfa1n2, sfa_2, axis=1)

print orig_n_sfa1n2, orig_n_sfa1n2.shape
ndarray_to_string(orig_n_sfa1n2, prefix="# sample, original signal 1, sfa signals1, sample, original signal 2, sfa signals 2 \n#", col_sep=", ", row_sep="\n", out_filename = "orig_n_sfa_1n2.txt")
    
#orig_n_sfa2 = orig2 + 0.0
#orig_n_sfa2 = numpy.append(orig_n_sfa2, sfa_2, axis=1)
#print orig_n_sfa2, orig_n_sfa2.shape
#ndarray_to_string(orig_n_sfa2, prefix="# sample, original signal 2, sfa signals \n#", col_sep=", ", row_sep="\n", out_filename = "orig_n_sfa_2.txt")
#    
#Computing function G
ramp = numpy.linspace(-3,3,num_steps)
ramp = ramp.reshape((num_steps,1))

poly_exp_ramp = poly_exp_node.execute(ramp)
sfa_ramp1 = sfa_node_1.execute(poly_exp_ramp)
sfa_ramp2 = sfa_node_2.execute(poly_exp_ramp)

#for i in range(20):
#    g1=numpy.zeros((num_steps,2))
#    g1[:,0] = ramp[:,0]
#    g1[:,1] = sfa_ramp1[:,i]
#    ndarray_to_string(g1, prefix="#", col_sep=", ", row_sep="\n", out_filename = "g1_%d.txt"%i)
#
#for i in range(20):
#    g2=numpy.zeros((num_steps,2))
#    g2[:,0] = ramp[:,0]
#    g2[:,1] = sfa_ramp2[:,i]
#    ndarray_to_string(g1, prefix="#", col_sep=", ", row_sep="\n", out_filename = "g2_%d.txt"%i)

gfuncs_1 = ramp + 0.0
gfuncs_1 = numpy.append(gfuncs_1, sfa_ramp1, axis=1)
print gfuncs_1, gfuncs_1.shape
ndarray_to_string(gfuncs_1, prefix="# x val, g1_0(x) ... g1_19(x) \n#", col_sep=", ", row_sep="\n", out_filename = "gfuncs_1.txt")

gfuncs_2 = ramp + 0.0
gfuncs_2 = numpy.append(gfuncs_2, sfa_ramp2, axis=1)
print gfuncs_2, gfuncs_2.shape
ndarray_to_string(gfuncs_2, prefix="# x val, g2_0(x) ... g2_19(x) \n#", col_sep=", ", row_sep="\n", out_filename = "gfuncs_2.txt")

print "************ Plotting Signals **************"
ax_1 = plt.figure()
ax_1.subplots_adjust(hspace=0.5)
plt.suptitle("First experiment")

sp11 = plt.subplot(3,2,1)
plt.title("Original Signal 1")
sp11.plot(numpy.arange(num_steps), s1, color="b")

sp21 = plt.subplot(3,2,3)
plt.title("Slowest Extracted Signals 1")
sp21.plot(numpy.arange(num_steps), sfa_1[:,0:2])

sp11 = plt.subplot(3,2,2)
plt.title("Original Signal 2")
sp11.plot(numpy.arange(num_steps), s3, color="b")

sp21 = plt.subplot(3,2,4)
plt.title("Slowest Extracted Signals 2")
sp21.plot(numpy.arange(num_steps), sfa_2[:,0:2])

sp31 = plt.subplot(3,2,5)
plt.title("G1_1")
sp31.plot(ramp[:,0], sfa_ramp1[:,0])

sp32 = plt.subplot(3,2,6)
plt.title("G2_1")
sp32.plot(ramp[:,0], sfa_ramp2[:,0])

plt.show()



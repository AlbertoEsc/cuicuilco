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
import time

#Available expansion functions:
# { identity(x), pair_abs_dif_ex(x), pair_abs_sum_ex(x), pair_prod_ex(x), pair_sqrt_abs_sum_ex(x), pair_sqrt_abs_dif_ex(x) }

print "***********************"
print "Testing General Expansion Inversion "
print "        inline code"
print "***********************"
distance = distance_Euclidean
ex_funcs = [identity, pair_prod_ex]

num_samples = 2
dim_x = 4
genexpnode = GeneralExpansionNode(ex_funcs)

x = numpy.random.normal(size=(num_samples,dim_x))
print "x=", x
print "ex_funcs=", ex_funcs

ex_x = genexpnode.execute(x)
print "ex_x=", ex_x
dim_ex_x = ex_x.shape[1]
print "Width of ex_x is ", dim_ex_x, ", height=", ex_x.shape[0]

var_noise = 0.3
ex_x_noisy = ex_x + numpy.random.normal(scale=var_noise, size=(num_samples,dim_ex_x))
print "ex_x_noisy=", ex_x_noisy


if ex_funcs[0] is identity :
    print "Using hint for first approximation!"
    app_x = ex_x_noisy[:,0:dim_x]
else:
    app_x = numpy.random.normal(size=(num_samples,dim_x))
app_ex_x =  genexpnode.execute(app_x)
print "app_ex_x", app_ex_x


max_steady_iter = dim_x * 10
for row in range(num_samples):
    delta = 1.0
    app_x_row = app_x[row].reshape(1, dim_x)
#
    ex_x_noisy_row  = ex_x_noisy[row]
    app_ex_x_row = app_ex_x[row]
    dist = distance(ex_x_noisy_row, app_ex_x_row)
    print "row dist %d is ="%(row), dist
#
    dist = distance(ex_x_noisy_row, app_ex_x_row)
    print "row dist full is =", dist
#   
    while delta > 0.0001:
#        print "Delta Value=", delta
        steady_iter = 0
        while steady_iter < max_steady_iter:
            i = numpy.random.randint(0, high=dim_x)
            app_x_tmp = app_x_row.copy()
#            print "i=", i, 
            app_x_tmp[0,i] = app_x_tmp[0, i] + numpy.random.normal(scale=delta)
#
            app_ex_x_tmp =  genexpnode.execute(app_x_tmp)
            dist_tmp = distance(ex_x_noisy_row, app_ex_x_tmp) 
            if dist_tmp < dist:
                app_x_row = app_x_tmp.copy()
                app_ex_x_row = app_ex_x_tmp.copy()
                dist=dist_tmp
#                print ", Dist=", dist
                steady_iter = 0
            else:
                steady_iter = steady_iter + 1    
        delta = delta * 0.8
    print "row dist is =", dist
    app_x[row]=app_x_row[0].copy()
    app_ex_x[row] = app_ex_x_row[0].copy()

print "Original x was:"
print x
print "Recovered x is:"
print app_x
print "with distance =", distance(x, app_x)

print "Original expansion ex_x was:"
print ex_x

print "Original noisy expansion ex_x_noisy was:"
print ex_x_noisy

print "Approximated expansion app_ex_x is:"
print app_ex_x
print "with image distance =", distance(ex_x_noisy, app_ex_x)



print "***********************"
print "Testing General Expansion Inversion "
print "        function call code"
print "***********************"


t1 = time.time()
app_x_2, app_ex_x_2 = invert_exp_funcs(ex_x_noisy, dim_x, ex_funcs, use_hint=True, max_steady_factor=8.0, delta_factor=0.8, min_delta=0.000001)
t2 = time.time()
print "Original x was:"
print x
print "Recovered x is:"
print app_x_2
print "with distance =", distance(x, app_x_2)
print "and image distance =", distance(ex_x_noisy, app_ex_x_2)
print 'in time %0.3f ms' % ((t2-t1)*1000.0)

#This seems to be the most efficient/precise way to approximate the inverse
t1 = time.time()
app_x_2, app_ex_x_2 = invert_exp_funcs(ex_x_noisy, dim_x, ex_funcs, use_hint=True, max_steady_factor=1.5, delta_factor=0.6, min_delta=0.00001)
t2 = time.time()
print "Original x was:"
print x
print "Recovered x is:"
print app_x_2
print "with distance =", distance(x, app_x_2)
print "and image distance =", distance(ex_x_noisy, app_ex_x_2)
print 'in time %0.3f ms' % ((t2-t1)*1000.0)


print "***********************"
print "Testing General Expansion Node "
print "        including built-in inversion"
print "***********************"
ex_funcs = [identity, pair_prod_ex]
num_samples = 2
dim_x = 4
genexpnode = GeneralExpansionNode(ex_funcs, input_dim = dim_x, use_hint=True, max_steady_factor=2.5, \
                 delta_factor=0.7, min_delta=0.000001)

x = numpy.random.normal(size=(num_samples,dim_x))
print "x=", x
print "ex_funcs=", ex_funcs

ex_x = genexpnode.execute(x)
print "ex_x=", ex_x
dim_ex_x = ex_x.shape[1]
print "Width of ex_x is ", dim_ex_x, ", height=", ex_x.shape[0]

var_noise = 0.1
ex_x_noisy = ex_x + numpy.random.normal(scale=var_noise, size=(num_samples,dim_ex_x))
print "ex_x_noisy=", ex_x_noisy

app_x = genexpnode.inverse(ex_x_noisy)
app_ex_x = genexpnode.execute(app_x)
print "Original x was:"
print x
print "Recovered x is:"
print app_x
print "with distance =", distance(x, app_x)

print "Original expansion ex_x was:"
print ex_x
print "Original noisy expansion ex_x_noisy was:"
print ex_x_noisy
print "Approximated expansion app_ex_x is:"
print app_ex_x
print "with image distance =", distance(ex_x_noisy, app_ex_x)

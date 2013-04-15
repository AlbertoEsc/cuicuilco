#Basic Experiments regarding computation of slow signals
#By Alberto Escalante. Alberto.Escalante@ini.ruhr-uni-bochum.de First Version 8 July 2010
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott

import numpy
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import Image
import mdp
import os
import glob
import random
import sys
sys.path.append("/home/escalafl/workspace/cuicuilco/src")
import sfa_libs 

def new_nl_func(data, expo1=2, expo2=0.5):
    mask = numpy.abs(data) < 1
    res = numpy.zeros(data.shape)

    res[mask] = (numpy.abs(data) ** expo1)[mask]
    res[mask^True] = (numpy.abs(data) ** expo2)[mask^True]   
    return res

def signed_expo(data, expo):
    signs = numpy.sign(data)
    return signs * numpy.abs(data) ** expo


def new_signed_nl_func(data, expo1=2, expo2=0.5):
    mask = numpy.abs(data) < 1
    res = numpy.zeros(data.shape)

    res[mask] = (signed_expo(data, expo1))[mask]
    res[mask^True] = (signed_expo(data, expo2))[mask^True]   
    return res

def other_signed_nl_func(data1, data2, expo1=2, expo2=0.5):
    sqrts = numpy.sqrt(numpy.abs(data1))

    res1 = numpy.sign(data1 * data2)*sqrts
    
    return new_signed_nl_func(res1, 1.0, 1.0)

num_steps = 100000
num_slow_signals = 10
num_blocks = num_steps
block_size = num_steps / num_blocks

std_noise = .15
#0.85=>-0.4954075   0.30499337  0.63831671 -0.36681614  0.05889589
#0.35=>-1.16191841 -0.88211213  5.05820853 -3.33701262  0.65638414
#0.25=>-1.19079117 -1.7948324   6.50985796 -3.90644898  0.69533003
#0.2 =>-1.27244014 -1.02075956  3.87585643 -1.23697957 -0.10522748
#0.15=>-1.38004187  0.16608828  0.15704064  2.48480606 -1.2517549 


t = numpy.linspace(0, numpy.pi, num_steps)

factor = numpy.sqrt(2)
sl = numpy.zeros((num_steps, num_slow_signals))
for i in range(num_slow_signals):
    # *(numpy.random.randint(2)*2 -1)
    sl[:,i] = factor * numpy.cos((i+1)*t)  + std_noise * numpy.random.normal(size=(num_steps)) 
    print "sl[:,i].mean()=", sl[:,i].mean(), "  sl[:,i].var()=", sl[:,i].var()

sl = (sl-sl.mean(axis=0))/sl.std(axis=0)
#Noise-free signals
sl1 = numpy.cos(t)
sl1 = (sl1-sl1.mean(axis=0))/sl1.std(axis=0)
#What function do we want to approximate?
#sl2 = numpy.cos(2*t)
sl2 = -1 * numpy.sin(t)
sl2 = (sl2-sl2.mean(axis=0))/sl2.std(axis=0)

#exponents =  [2.0, 1.0, 0.8, 0.5, -1]
exponents =  [-1]
num_exponents = len(exponents)

#First Experiment From slow signals to fast signals (twice its frequency)
f1 = plt.figure()
plt.suptitle("Approximations to Second slow signal as a*|x|**c+b and a0+a1*|x|+a2*|x|**2+a3*|x|**3+a4*|x|**4")
  
#display Sequence of images
#Alternative: im1.show(command="xv")
enable_fit = True

for i, exponent in enumerate(exponents):
    if exponent != -1:
        ap1 = numpy.abs(sl[:,0]) ** exponent
    else:
        ap1 = new_nl_func(sl[:,0], 2, 0.5)
    print "ap1.shape", ap1.shape

    a = ap1.std()
    ap1 = ap1/a
    b = ap1.mean()
    ap1 = ap1-b

    abssl = numpy.abs(sl[:,0])
    p0 = numpy.ones(len(abssl))
    p1 = abssl
    p2 = new_nl_func(sl[:,0], 2, 0.5)
    p3 = new_nl_func(sl[:,0], 2.0, 0.6)
    p4 = new_nl_func(sl[:,0], 2.0, 0.7)
    p5 = new_nl_func(sl[:,0], 1.5, 0.6)
    p6 = new_nl_func(sl[:,0], 2.5, 0.6)
    p7 = numpy.abs(sl[:,0]+1.0)**0.6
    p8 = numpy.abs(sl[:,0]-1.0)**0.6
    p9 = numpy.abs(sl[:,0]+1.5)**0.6
    p10 = numpy.abs(sl[:,0]-1.5)**0.6
#    p2 = abssl**2
#    p3 = abssl**3
#    p4 = abssl**4
#    p5 = abssl**5
#    p6 = abssl**6
#    p7 = abssl**7
#    p8 = abssl**8
#    p9 = abssl**9
#    p10 = abssl**10

    base_funcs = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
    num_base_funcs = 7   
    mat_p = numpy.zeros((num_steps, num_base_funcs))
    for ii in range(num_base_funcs):
        mat_p[:,ii] = base_funcs[ii]
    
    pinv = numpy.linalg.pinv(mat_p)
    print "pinv.shape is", pinv.shape
    coefs = numpy.dot(pinv, sl2)
    print "coefs: ", coefs
    print "mat_p.shape", mat_p.shape
    ap2 = numpy.dot(mat_p, coefs)
    ap2 = ap2-ap2.mean()
    ap2 = ap2/ap2.std()
    
    print "ap2=", ap2
    print "ap2.shape=", ap2.shape

    ax = plt.subplot(num_exponents,1,i+1)
#    plt.title("c=%f"%exponent)
    plt.plot(t, sl[:,0], "b.")
    plt.plot(t, ap1, "r.")
    if enable_fit:
        plt.plot(t, ap2, "m.")
    plt.plot(t, sl2, "g.")
    if enable_fit:    
        plt.legend(["x(t)=cos(t)+%f*n(t)"%std_noise, "white(|x|^%f)"%exponent, "white(fit(|x|))", "cos(2t)"], loc=4)
    else:
        plt.legend(["x(t)=cos(t)+%f*n(t)"%std_noise, "white(|x|^%f)"%exponent, "cos(2t)"], loc=4)

    error = sl2-ap1
    error2 = sl2-ap2
    if enable_fit:
        plt.xlabel("approximation error white(|x|^%f) and white(fit(|x|)) vs cos(2t): %f and %f"%(exponent, error.std(), error2.std()))
    else:
        plt.xlabel("approximation error white(|x|^%f) vs cos(2t): %f"%(exponent, error.std()))        



f2 = plt.figure()
plt.suptitle("Actual mapping we are interested in to generate cos(2t) from x(t)")
plt.plot(sl[:,0], sl2, "g.")
if enable_fit:
    plt.plot(sl[:,0], ap2, "m.")
    plt.plot(sl[:,0], ap1,"r.")
    plt.legend(["Desired mapping", "fit solution (normalized)", "approx solution (normalized)"], loc=4)

###Code for approximating slower frequency armonics
##f3 = plt.figure()
##for i, exponent in enumerate(exponents):
##    nl1 = numpy.abs(sl[:,1]) ** exponent
##    nl1 = (nl1 - nl1.mean())/nl1.std()
##    nl2 = numpy.abs(sl[:,2]) ** exponent
##    nl2 = (nl2 - nl2.mean())/nl2.std()
##
##    nl3 = numpy.abs(sl[:,1]+sl[:,2]) ** exponent
##    nl3 = (nl3 - nl3.mean())/nl3.std()
##    nl4 = numpy.abs(sl[:,2]+sl[:,3]) ** exponent
##    nl4 = (nl4 - nl4.mean())/nl4.std()
##    nl5 = numpy.abs(sl[:,3]+sl[:,4]) ** exponent
##    nl5 = (nl5 - nl5.mean())/nl5.std()
##    nl6 = numpy.abs(sl[:,4]+sl[:,5]) ** exponent
##    nl6 = (nl6 - nl6.mean())/nl6.std()
##    nl7 = numpy.abs(sl[:,5]+sl[:,6]) ** exponent
##    nl7 = (nl7 - nl7.mean())/nl7.std()
##    nl8 = numpy.abs(sl[:,6]+sl[:,7]) ** exponent
##    nl8 = (nl8 - nl8.mean())/nl8.std()
##    nl9 = numpy.abs(sl[:,7]+sl[:,8]) ** exponent
##    nl9 = (nl9 - nl9.mean())/nl9.std()
##    nl10 = numpy.abs(sl[:,8]+sl[:,9]) ** exponent
##    nl10 = (nl10 - nl10.mean())/nl10.std()
##
##
###    nl3 = numpy.abs(sl[:,3]) ** exponent
###    nl3 = (nl3 - nl3.mean())/nl3.std()
###    nl4 = numpy.abs(sl[:,4]) ** exponent
###    nl4 = (nl4 - nl4.mean())/nl4.std()
##
##
##    mat = numpy.zeros((num_steps, 9))
##    mat[:,0]=1
##    mat[:,1]=nl3
##    mat[:,2]=nl4
##    mat[:,3]=nl5
##    mat[:,4]=nl6
##    mat[:,5]=nl7
##    mat[:,6]=nl8
##    mat[:,7]=nl9
##    mat[:,8]=nl10
##
##
##    pinv = numpy.linalg.pinv(mat)
##    coefs = numpy.dot(pinv, sl[:,0])
##    
##    ap2 = numpy.dot(mat, coefs)
##
##    ax = plt.subplot(num_exponents,1,i+1)
###    plt.title("c=%f"%exponent)
##    plt.plot(t, sl[:,0], "b.")
##    plt.plot(t, ap2, "r.")
###    plt.plot(t, sl[:,1], "g.")
##
##    error = sl[:,0]-ap2
##    plt.xlabel("approximation error from sl[:,1]... to sl[:,0] for c=%f is %f, coefs=%s"%(exponent, error.std(), str(coefs)))





exponents1 = [0.4]
exponents2 = [0.03, 0.04, 0.2]

num_exponents = len(exponents)

#f4 = plt.figure()
for i, exponent1 in enumerate(exponents1):
    for j, exponent2 in enumerate(exponents2):
        nl_func = lambda x: new_signed_nl_func(x, exponent1, exponent2)
    #    nl_func = lambda x: numpy.sign(x)
    #    nl_func = lambda x: x
        
        nl3 = nl_func(sl[:,1]*sl[:,2])
    #    nl3 = numpy.sqrt((1.0/2 + sl[:,1]/(1.4*2)).clip(0,5)) * numpy.sign(sl[:,1]*sl[:,2])+8for i in range(len(divisions_sl2)-1):
    
    #    other_signed_nl_func
    #    nl3 = other_signed_nl_func(sl[:,1], sl[:,2], 1.0, 1.0)
        nl3 = (nl3 - nl3.mean())/nl3.std()
        
        nl4 = nl_func(sl[:,2]*sl[:,3])
        nl4 = (nl4 - nl4.mean())/nl4.std()
        nl5 = nl_func(sl[:,3]*sl[:,4])
        nl5 = (nl5 - nl5.mean())/nl5.std()
        nl6 = nl_func(sl[:,4]*sl[:,5])
        nl6 = (nl6 - nl6.mean())/nl6.std()
        nl7 = nl_func(sl[:,5]*sl[:,6])
        nl7 = (nl7 - nl7.mean())/nl7.std()
        nl8 = nl_func(sl[:,6]*sl[:,7])
        nl8 = (nl8 - nl8.mean())/nl8.std()
        nl9 = nl_func(sl[:,7]*sl[:,8])    
        nl9 = (nl9 - nl9.mean())/nl9.std()
        nl10 = nl_func(sl[:,8]*sl[:,9])    
        nl10 = (nl10 - nl10.mean())/nl10.std()
    
    
    #    nl3 = numpy.abs(sl[:,3]) ** exponent
    #    nl3 = (nl3 - nl3.mean())/nl3.std()
    #    nl4 = numpy.abs(sl[:,4]) ** exponent
    #    nl4 = (nl4 - nl4.mean())/nl4.std()
    
    
        mat = numpy.zeros((num_steps, 9))
        mat[:,0]=1
        mat[:,1]=nl3
        mat[:,2]=nl4
        mat[:,3]=nl5
        mat[:,4]=nl6
        mat[:,5]=nl7
        mat[:,6]=nl8
        mat[:,7]=nl9
        mat[:,8]=nl10
    
    
        pinv = numpy.linalg.pinv(mat)
        coefs = numpy.dot(pinv, sl[:,0])
        
        ap2 = numpy.dot(mat, coefs)
        ap2 = (ap2-ap2.mean())/ap2.std()
        
        delta = sfa_libs.comp_delta(ap2.reshape((num_steps,1)))[0]
          
#        ax = plt.subplot(num_exponents,1,i+1)
#    #    plt.title("c=%f"%exponent)
#        plt.plot(t, sl[:,0], "b.")
#        plt.plot(t, ap2, "r.")
#    #    plt.plot(t, sl[:,1], "g.")
    
        error = sl[:,0]-ap2
#        plt.xlabel("approximation error from sl[:,1]... to sl[:,0] for c=%f is %f, coefs=%s"%(exponent, error.std(), str(coefs)))
        print "approximation error from sl[:,1], sl[:,2] to sl[:,0] for exp1=%f, exp2=%f is %f, sl=%f, coefs=%s"%(exponent1, exponent2, error.std(), delta, str(coefs))



#Conclusions:
#signed expo 0.4 (product) seems to be the best nl function
#signed new_nl(0.4, 1.0, 0.4) is less good ...
#but signed new_nl(0.4, 0.1-0.3, 0.4) improves (thought slowness might be compromised)
#incredible!! also decreasing the second exponent helps!!!
#ex: signed new_nl(0.4, 0.1-0.3, 0.05)  => 0.3974 error
#however numpy.sign() has less good performance (still similar! => 0.404052 error)
#Note: taking the square root of the first one and fixing the sign is poor and has sharp jumps



#f5 = plt.figure()
#plt.suptitle("Actual mapping we are interested in to generate cos(t) from cos(2t) and cos(3t)")
#divisions = [-1.7,-1.25, -0.7, -0.5, -0.25, 0.0]
#enable_fit = True

divisions_sl2 = numpy.arange(-1.45, 1.45, 0.1)
#divisions_sl2 = [-1.0, -0.75]
#[-1.5,-1.0, -0.5, 0.0, 0.5]
#divisions_sl3 = numpy.arange(-1.5, 0.2, 0.2)
#divisions_sl3 = [-1.5,-1.0, -0.5, 0.0, 0.5]
divisions_sl3 = [-1.0, -0.9, -0.8]
divisions_sl3 = numpy.arange(-1.45, 1.45, 0.1)



goal_out = {}
for i in range(len(divisions_sl2)-1):
    for j in range(len(divisions_sl3)-1):
        mask = (sl[:,1] > divisions_sl2[i]) & (sl[:,1]<divisions_sl2[i+1]) & (sl[:,2] > divisions_sl3[j]) & (sl[:,2]<divisions_sl3[j+1])
        mean = sl1[mask].mean()
        if numpy.isnan(mean):
            mean = 0
        goal_out[(i,j)] = mean

fig=f6 = plt.figure()
ax = Axes3D(f6)

centroids_sl2 = (divisions_sl2[:-1]+divisions_sl2[1:])/2
centroids_sl3 = (divisions_sl3[:-1]+divisions_sl3[1:])/2
xx, yy = numpy.meshgrid(centroids_sl2, centroids_sl3)
zz = numpy.zeros((len(centroids_sl2),len(centroids_sl3)))
for i in range(len(centroids_sl2)):
    for j in range(len(centroids_sl3)):
        zz[i,j] = goal_out[(i,j)]
                
surf = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=mpl.cm.jet,
        linewidth=0, antialiased=False)
ax.set_zlim3d(-1.01, 1.01)
ax.w_zaxis.set_major_locator(mpl.ticker.LinearLocator(10))
ax.w_zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.03f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
        
ax.set_xlabel('c(2)')
ax.set_ylabel('c(3)')
ax.set_zlabel('c(1)')


fig=f7 = plt.figure()
ax = Axes3D(f7)
app_out = numpy.zeros((len(centroids_sl2), len(centroids_sl3)))
for i in range(len(centroids_sl2)):
    for j in range(len(centroids_sl3)):
        xx_sl2 = centroids_sl2[i]
        xx_sl3 = centroids_sl3[j]
        www = numpy.array([xx_sl2*xx_sl3]).reshape((1,1))
        www = nl_func(www)
        app_out[i,j] = www[0,0]
      
        
surf = ax.plot_surface(xx, yy, app_out, rstride=1, cstride=1, cmap=mpl.cm.jet,
        linewidth=0, antialiased=False)
ax.set_zlim3d(-1.01, 1.01)
ax.w_zaxis.set_major_locator(mpl.ticker.LinearLocator(10))
ax.w_zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.03f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
        
ax.set_xlabel('c(2)')
ax.set_ylabel('c(3)')
ax.set_zlabel('app c(1)')
plt.show()




plt.show()

plt.suptitle("Actual/approx mapping we need to generate cos(t) from cos(2t) and cos(3t)")
plt.subplot(2,2,1)
for i in range(len(divisions_sl2)-1):
    y_out = []
    for j in range(len(divisions_sl3)-1):
        y_out.append(goal_out[(i,j)])
    plt.plot(divisions_sl3[:-1], y_out)
#plt.legend(["Desired mapping for fixed sl2 >= %f"%divisions_sl2[i] for i in range(len(divisions_sl2)-1)], loc=3)


f7 = plt.figure()
plt.subplot(2,2,2)
#plt.suptitle("Actual mapping we are interested in to generate cos(t) from cos(2t) and cos(3t)")
for j in range(len(divisions_sl3)-1):
    y_out = []
    for i in range(len(divisions_sl2)-1):
        y_out.append(goal_out[(i,j)])
    plt.plot(divisions_sl2[:-1], y_out)
#plt.legend(["Desired mapping for fixed sl3 >= %f"%divisions_sl3[i] for i in range(len(divisions_sl3)-1)], loc=3)




plt.subplot(2,2,3)
for i in range(len(divisions_sl2)-1):
    y_out = []
    for j in range(len(divisions_sl3)-1):
        y_out.append(app_out[i,j])
    plt.plot(divisions_sl3[:-1], y_out)
#plt.legend(["Approximate mapping for fixed sl2 >= %f"%divisions_sl2[i] for i in range(len(divisions_sl2)-1)], loc=3)

plt.subplot(2,2,4)
for j in range(len(divisions_sl3)-1):
    y_out = []
    for i in range(len(divisions_sl2)-1):
        y_out.append(app_out[i,j])
    plt.plot(divisions_sl2[:-1], y_out)
#plt.legend(["Approximate mapping for fixed sl3 >= %f"%divisions_sl3[i] for i in range(len(divisions_sl3)-1)], loc=3)


##fixed_index = 2
##variable_index = 1
##for i in range(len(divisions)-1):
##    mask = (sl[:,fixed_index] > divisions[i]) & (sl[:,fixed_index]<divisions[i+1]) 
##    goal_bin = sl1[mask]
##    sl2_bin = sl[:,1][mask]
##    sl3_bin = sl[:,2][mask]
##
##    if fixed_index == 2:
##        plt.plot(sl2_bin, goal_bin, ".")
##    else:
##        plt.plot(sl3_bin, goal_bin, ".")
##        
##    if enable_fit: #and False:
##        var2_divisions = [-2.0, -1.5, -1, -0.5, 0, 0.5, 1.0, 1.5, 2.0]
##        posx = []
##        posy = []
##        for j in range(len(var2_divisions)-1):
##            if variable_index == 0:
##                mask2 = (sl2_bin > var2_divisions[j]) & (sl2_bin <var2_divisions[j+1]) 
##                sl1_bin_bin = goal_bin[mask2]
##            else:
##                mask2 = (sl3_bin > var2_divisions[j]) & (sl3_bin <var2_divisions[j+1]) 
##                sl1_bin_bin = goal_bin[mask2]
##                
###            plt.plot([(var2_divisions[j]+var2_divisions[j+1])/2], [sl1_bin_bin.mean()], "o")
##            posx.append((var2_divisions[j]+var2_divisions[j+1])/2)
##            posy.append(sl1_bin_bin.mean())
##        plt.plot(posx, posy)
##            
###        plt.plot(sl[:,0], ap1,"r.")
###        plt.legend(["Desired mapping", "fit solution (normalized)", "approx solution (normalized)"], loc=4)
##plt.legend(["Desired mapping for bin %d"%i for i in range(len(divisions)-1)], loc=5)

###Fourth Experiment: Shows taylor approximation of sqrt
##t=numpy.arange(0, 3, 0.01)
##x=t-1
##y=1+0.5*x+(1.0/8)*x**2-(1.0/16)*x**3-(5.0/128)*x**4+(7.0/256)*x**5-(21.0/1024)*x**6
##plt.figure()
##plt.plot(t,y)
##plt.plot(t,numpy.sqrt(t))


plt.show()

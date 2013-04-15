#Scripts that generate some images used for presentations on SFA
#By Alberto Escalante. Alberto.Escalante@ini.ruhr-uni-bochum.de First Version 5 Dec 2011
#Ruhr-University-Bochum, Institute of Neurocomputation, Theory of Neural Systems, Group of Prof. Dr. Wiskott

import numpy
#import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import PIL
import Image
import mdp
import os
import glob
import random
import sys
sys.path.append("/home/escalafl/workspace4/cuicuilco_MDP3.2/src")
from nonlinear_expansion import *
#import sfa_libs
import getopt


plot_SFA_feautre_extraction=True #and False
plot_obsolete_expansion_complexity=True and False
plot_expansion_complexities=True and False
plot_signal_extraction_network=True and False
plot_overfitting_metric=True and False
plot_outlier_amplification_metric=True and False

seed = 0
numpy.random.seed(seed)

def metric_max_var(x):
    return numpy.abs(x).max(axis=1)

def metric_magnitude(x):
    return (x**2).sum(axis=1)**0.5

def metric_RMSA(x):
    return (x**2).mean(axis=1)**0.5

def metric_corrected_RMSA(x):
    return (x**2).mean(axis=1)**0.5 * numpy.sqrt(x.shape[1])

def metric_individual_RMSE(x):
    comp_RMSA =  ((x**2).mean(axis=0))**0.5
    print "comp_RMSA=", comp_RMSA
    x = x / comp_RMSA
    return (x**2).mean(axis=1)**0.5

def sgn_expo(x, expo):
    s = numpy.sign(x)
    y = s * numpy.abs(x)**expo
    return y

if plot_expansion_complexities:
    #Too Simple Expansion Figure
    f0 = plt.figure()
    
    num_steps = 400
    noise_std = 0.15
    max_t_disp = 3.14
    
    t = numpy.linspace(0, numpy.pi, num_steps)
    y_diff = numpy.random.normal(0.0, 1.0, num_steps)
    y = y_diff.cumsum()
    y = y - y.mean()
    y = y / ((y**2).mean()**0.5)
    y = sgn_expo(y, 0.5) + 0.5*numpy.cos(10*t)
    y = y - y.mean()
    y = y / ((y**2).mean()**0.3)
    
    ax =plt.subplot(2,4,1)
    plt.plot(t, y, "b-", linewidth=1.5)
    plt.xlim(0,max_t_disp)
    plt.ylim(-2.0,2.0)
    
    y_noise = y + noise_std * numpy.random.normal(0.0, 1.0, num_steps)
    y_noise = y_noise - y_noise.mean()
    y_noise = y_noise / ((y_noise**2).mean()**0.5)
    
    ax =plt.subplot(2,4,5)
    plt.plot(t, y_noise, "b-", linewidth=1.5)
    plt.xlim(0,max_t_disp)
    plt.ylim(-2.0,2.0)
    
    #Perfect Expansion Complexity
    noise_std = 0.05
    factor = -1*2 ** 0.5
    sl_0 = factor * numpy.cos(t) + noise_std * numpy.random.normal(size=(num_steps)) 
    sl_0 = fix_mean_var(sl_0)
    
    y_diff = numpy.random.normal(0.0, 1.0, num_steps)
    y = y_diff.cumsum()
    y = fix_mean_var(y)
    y = sgn_expo(y, 0.5) + 0.5*numpy.cos(15*t)
    y = fix_mean_var(y)
    sl_1 = sl_0 + 0.15 * y + noise_std * numpy.random.normal(0.0, 1.0, num_steps)
    sl_1 = fix_mean_var(sl_1)
    
    ax =plt.subplot(2,4,2)
    plt.plot(t, sl_1, "b-", linewidth=1.5)
    plt.ylim(-2.0,2.0)
    plt.xlim(0,max_t_disp)
    
    sl_2 = sl_0 + 0.15 * y + noise_std * numpy.random.normal(0.0, 1.0, num_steps)
    sl_2 = fix_mean_var(sl_2)
    ax =plt.subplot(2,4,6)
    plt.plot(t, sl_2, "b-", linewidth=1.5)
    plt.ylim(-2.0,2.0)
    plt.xlim(0,max_t_disp)
    
    #Perfect Expansion Complexity, Bad Expansion
    noise_std = 0.05
    noise_std_out = 0.05
    
    factor = -1*2 ** 0.5
    sl_0 = factor * numpy.cos(t) + noise_std * numpy.random.normal(size=(num_steps)) 
    sl_0 = fix_mean_var(sl_0)
    
    y_diff = numpy.random.normal(0.0, 1.0, num_steps)
    y = y_diff.cumsum()
    y = fix_mean_var(y)
    y = sgn_expo(y, 0.5) + 0.5*numpy.cos(10*t)
    y = fix_mean_var(y)
    sl_1 = sl_0 + 0.15 * y + noise_std * numpy.random.normal(0.0, 1.0, num_steps)
    sl_1 = fix_mean_var(sl_1)
    
    ax =plt.subplot(2,4,3)
    plt.plot(t, sl_1, "b-", linewidth=1.5)
    plt.ylim(-2.0,2.0)
    plt.xlim(0,max_t_disp)
    
    sl_2 = sl_0 + 0.15 * y + noise_std * numpy.random.normal(0.0, 1.0, num_steps)
    sl_2 = fix_mean_var(sl_2)
    sl_2 += noise_std_out * sgn_expo(numpy.random.normal(0.0, 1.0, num_steps), 4)
    ax =plt.subplot(2,4,7)
    plt.plot(t, sl_2, "b-", linewidth=1.5)
    plt.ylim(-2.0,2.0)
    plt.xlim(0,max_t_disp)
    
    
    #Too High Expansion Complexity
    noise_std = 0.005
    noise_std_2 = 0.3
    noise_std_out = 0.07
    
    factor = -1*2 ** 0.5
    sl_0 = factor * numpy.cos(t) + noise_std * numpy.random.normal(size=(num_steps)) 
    sl_0 = fix_mean_var(sl_0)
    
    sl_1 = sl_0
    ax =plt.subplot(2,4,4)
    plt.plot(t, sl_1, "b-", linewidth=1.5)
    plt.xlim(0,max_t_disp)
    plt.ylim(-2.0,2.0)
    
    y_diff = numpy.random.normal(0.0, 1.0, num_steps)
    y = y_diff.cumsum()
    y = fix_mean_var(y)
    y = sgn_expo(y, 0.5) + 0.5*numpy.cos(18*t)
    y = fix_mean_var(y)
    sl_2 = -0.4*sl_0 + 0.85 * y + noise_std_2 * numpy.random.normal(0.0, 1.0, num_steps)
    sl_2 = fix_mean_var(sl_2)
    sl_2 += noise_std_out * sgn_expo(numpy.random.normal(0.0, 1.0, num_steps), 4)
    ax =plt.subplot(2,4,8)
    plt.plot(t, sl_2, "b-", linewidth=1.5)
    plt.ylim(-2.0,2.0)
    plt.xlim(0,max_t_disp)

if plot_SFA_feautre_extraction:
    k=1.0
    d=2.0
    num_steps = 16 #10000   1000 for L, 148 for Q at 6 signals
    num_slow_signals = 10 #60   or 1667 for L and 349 for Q at 10 signals
    num_signals_disp = 4
    max_expanded_dim = 200 #2000   or 
    #noise_dim = 15
    
    std_noise = 0.00125
    seed = None
    #seed = 123456
    selected_function = None
    only_selected_function = False
    text = False
    
    numpy.random.seed(seed)
    t = numpy.linspace(0, numpy.pi, num_steps)
    t_i = numpy.arange(num_steps)
    factor = numpy.sqrt(2)
    
    sl = numpy.zeros((num_steps, num_slow_signals))
    for i in range(num_slow_signals):
        # *(numpy.random.randint(2)*2 -1)
        sl[:,i] = factor * numpy.cos((i+1)*t)  + std_noise * numpy.random.normal(size=(num_steps)) 
        print "sl[:,i].mean()=", sl[:,i].mean(), "  sl[:,i].var()=", sl[:,i].var()
    sl = fix_mean_var(sl)
    
    #Apply random rotation to data
    R = mdp.utils.random_rot(num_slow_signals)
    shuffled = numpy.dot(sl, R)
    
    n = mdp.nodes.SFANode()
    n.train(shuffled)
    extracted = n.execute(shuffled)
    #extracted = extracted * -1 * numpy.sign(extracted[0])
    styles = ["v-", "*--", "o:", "s-."]
    
    f0 = plt.figure()
    if text:
        plt.suptitle("Features Extraction with SFA (%d-Dimensional signal, only %d components shown)"%(num_slow_signals,num_signals_disp))
    ax =plt.subplot(2,1,1)
    for i in numpy.arange(num_signals_disp-1,-1,-1):
        plt.plot(t_i, shuffled[:,i], styles[i], linewidth=1.5)
    if text:
        plt.xlabel("Time (sample)")
        plt.ylabel("Input signal x(t)")
    plt.xlim(0-0.5, num_steps-1+0.5)
    plt.ylim(-2.5, 2.5)
    
    ax =plt.subplot(2,1,2)
    for i in numpy.arange(num_signals_disp-1,-1,-1):
        #plt.plot(t_i, shuffled[:,i], "o"+styles[i])
        plt.plot(t_i, extracted[:,i], styles[i], linewidth=1.5)
    if text:
        plt.xlabel("Time (sample)")
        plt.ylabel("Features extracted y(t)")
    plt.xlim(-0.5, num_steps-1+0.5)
    plt.ylim(-2.5, 2.5)
    
#    plt.suptitle("Expansion functions for different base functions: base (k,d)")
#                ax.axis('off')


#####plt.legend(["Desired mapping for fixed sl3 >= %f"%divisions_sl3[i] for i in range(len(divisions_sl3)-1)], loc=3)
if plot_obsolete_expansion_complexity:
    num_steps = 1000 #10000   1000 for L, 148 for Q at 6 signals
    num_slow_signals = 2 #60   or 1667 for L and 349 for Q at 10 signals
    num_signals_disp = 2
    std_noise_training = 0.0125
    std_noise_test = 0.25
    std_noise_outliers = 0.15
    numpy.random.seed(seed)
    max_amplitude_sfa = 1.5 #2.0
    max_t_disp = 3.0
    t = numpy.linspace(0, numpy.pi, num_steps)
    
    sl_training = numpy.zeros((num_steps, num_slow_signals))
    for i in range(num_slow_signals):
        # *(numpy.random.randint(2)*2 -1)
        sl_training[:,i] = factor * numpy.cos((i+1)*t)  + std_noise_training * numpy.random.normal(size=(num_steps)) 
        print "sl[:,i].mean()=", sl_training[:,i].mean(), "  sl[:,i].var()=", sl_training[:,i].var()
    sl_training = fix_mean_var(sl_training)
    sl_training *= numpy.sign(sl_training[0,0]) * -1
    
    sl_test = numpy.zeros((num_steps, num_slow_signals))
    for i in range(num_slow_signals):
        # *(numpy.random.randint(2)*2 -1)
        sl_test[:,i] = factor * numpy.cos((i+1)*t)  + std_noise_test * numpy.random.normal(size=(num_steps)) 
        print "sl[:,i].mean()=", sl_test[:,i].mean(), "  sl[:,i].var()=", sl_test[:,i].var()
    sl_test = fix_mean_var(sl_test)
    sl_test *= numpy.sign(sl_test[0,0])* -1
    
    sl_outliers = numpy.zeros((num_steps, num_slow_signals))
    for i in range(num_slow_signals):
        # *(numpy.random.randint(2)*2 -1)
        sl_outliers[:,i] = factor * numpy.cos((i+1)*t)  + std_noise_training/2 * numpy.random.normal(size=(num_steps)) + std_noise_test * numpy.random.normal(size=(num_steps)) ** 3
        print "sl[:,i].mean()=", sl_outliers[:,i].mean(), "  sl[:,i].var()=", sl_outliers[:,i].var()
    sl_outliers = fix_mean_var(sl_outliers)
    sl_outliers *= numpy.sign(sl_outliers[0,0])* -1
    
    f0 = plt.figure()
    plt.suptitle("Features Extraction with SFA (%d-Dimensional signal, only %d components shown)"%(num_slow_signals,num_signals_disp))
    
    ax =plt.subplot(3,1,1)
    plt.plot(t, sl_training[:,0], ".r")
    #plt.xlabel("Time (sample)")
    plt.ylabel("Training Data \n Overfitted")
    plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
    plt.xlim(0, max_t_disp)
    
    ax =plt.subplot(3,1,2)
    plt.plot(t, sl_test[:,0], ".r")
    #plt.xlabel("Time (sample)")
    plt.ylabel("Test Data")
    plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
    plt.xlim(0, max_t_disp)
    
    ax =plt.subplot(3,1,3)
    plt.plot(t, sl_outliers[:,0], ".r")
    plt.xlabel("Time (sample)")
    plt.ylabel("Test Data \n Outliers")
    plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
    plt.xlim(0, max_t_disp)


def MyBrownian1D(dim, mass):
    out = numpy.zeros(dim)
    speed = 0.0
    current = 0.0
    
    for i in range(dim):
        current = current + speed 
        force = 0.2*numpy.random.normal()      
#        if speed > 0 and current > 0:
#            force = min(force, force/numpy.e**(0.2+2*current) )
#        elif speed < 0 and current < 0:
#            force = max(force, force/numpy.e**(0.2-2*current) )
        if current >= 0:
            force2 = -0.05 * (0.02+0.1*current**2)
        else:
            force2 = 0.05 * (0.02+0.1*current**2)
        if speed > 0:
            force3 = -0.09 * (0.02+0.1*speed**2)
        elif speed < 0:
            force3 = 0.09 * (0.02+0.1*speed**2)
        else:
            force3=0.0
        force += force2 + force3
        acc = force / mass
        speed += acc 
        out[i] = current
        print speed, acc
    return out

#f0 = plt.figure()
#plt.suptitle("MyBrownian1D")
#ax =plt.subplot(4,1,1)
#plt.plot(MyBrownian1D(1000, 1.0))
#ax =plt.subplot(4,1,2)
#plt.plot(MyBrownian1D(1000, 1.0))
#ax =plt.subplot(4,1,3)
#plt.plot(MyBrownian1D(1000, 1.0))
#ax =plt.subplot(4,1,4)
#plt.plot(MyBrownian1D(1000, 1.0))
#plt.show()
#quit()

if plot_signal_extraction_network:
    num_steps = 1000 #10000   1000 for L, 148 for Q at 6 signals
    num_slow_signals = 3 #60   or 1667 for L and 349 for Q at 10 signals
    std_noise_first = 0.40
    std_noise_second = 0.15
    std_noise_third = 0.0001
    numpy.random.seed(seed)
    max_amplitude_sfa = 1.5 #2.0
    max_t_disp = 3.0
    t = numpy.linspace(0, numpy.pi, num_steps)
    
    input_data = numpy.zeros((num_steps, num_slow_signals))
    for i in range(num_slow_signals):
        input_data[:,i] = MyBrownian1D(num_steps, 1.0)
    
    sl_first = numpy.zeros((num_steps, num_slow_signals))
    for i in range(num_slow_signals):
        # *(numpy.random.randint(2)*2 -1)
        sl_first[:,i] = factor * numpy.cos((i+1)*t)  + std_noise_first * numpy.random.normal(size=(num_steps)) 
        print "sl[:,i].mean()=", sl_first[:,i].mean(), "  sl[:,i].var()=", sl_first[:,i].var()
    sl_first = fix_mean_var(sl_first)
    sl_first *= numpy.sign(sl_first[0,0]) * -1
    
    sl_second = numpy.zeros((num_steps, num_slow_signals))
    for i in range(num_slow_signals):
        # *(numpy.random.randint(2)*2 -1)
        sl_second[:,i] = factor * numpy.cos((i+1)*t)  + std_noise_second * numpy.random.normal(size=(num_steps)) 
        print "sl[:,i].mean()=", sl_second[:,i].mean(), "  sl[:,i].var()=", sl_second[:,i].var()
    sl_second = fix_mean_var(sl_second)
    sl_second *= numpy.sign(sl_second[0,0])* -1
    
    sl_third = numpy.zeros((num_steps, num_slow_signals))
    for i in range(num_slow_signals):
        # *(numpy.random.randint(2)*2 -1)
        sl_third[:,i] = factor * numpy.cos((i+1)*t)  + std_noise_third * numpy.random.normal(size=(num_steps))
        print "sl[:,i].mean()=", sl_third[:,i].mean(), "  sl[:,i].var()=", sl_third[:,i].var()
    sl_third = fix_mean_var(sl_third)
    sl_third *= numpy.sign(sl_third[0,0])* -1
    
    
    f0 = plt.figure()
    plt.suptitle("Features Extraction with SFA through the Network")
    
    ax =plt.subplot(1,4,1)
    plt.plot(t, input_data, ".")
    #plt.xlabel("Time (sample)")
    #plt.ylabel("Layer 1")
    plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
    plt.xlim(0, max_t_disp)
    plt.xlabel("Time (sample)")
    
    
    ax =plt.subplot(1,4,2)
    plt.plot(t, sl_first[:,0:num_slow_signals], ".")
    #plt.xlabel("Time (sample)")
    #plt.ylabel("Layer 1")
    plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
    plt.xlim(0, max_t_disp)
    plt.xlabel("Time (sample)")
    
    ax =plt.subplot(1,4,3)
    plt.plot(t, sl_second[:,0:num_slow_signals], ".")
    #plt.xlabel("Time (sample)")
    #plt.ylabel("Layer N/2")
    plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
    plt.xlim(0, max_t_disp)
    plt.xlabel("Time (sample)")
    
    ax =plt.subplot(1,4,4)
    plt.plot(t, sl_third[:,0:num_slow_signals], ".")
    #plt.ylabel("Layer N")
    plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
    plt.xlim(0, max_t_disp)
    plt.xlabel("Time (sample)")

if plot_overfitting_metric:
    num_steps = 100 #10000   1000 for L, 148 for Q at 6 signals
    numpy.random.seed(seed)
    max_amplitude = 1.5 #2.0
    max_t_disp = num_steps-1
    
    t = numpy.linspace(0, numpy.pi, num_steps)
    c1 = 2**0.5 * numpy.cos(t)
    numpy.random.shuffle(c1)
    
    input_data = c1
    out_data = c1 * (1-1.0/(1+c1**2))*0+0.4*numpy.random.normal(size=num_steps)

    f0 = plt.figure()
    plt.suptitle("Overfitting Metric")
    

    ax =plt.subplot(1,2,1)
    plt.plot(numpy.arange(num_steps), input_data*0, "k-", linewidth=2)
    plt.plot(numpy.arange(num_steps), input_data, "-")
    plt.ylim(-max_amplitude, max_amplitude)
    plt.xlim(0, max_t_disp)
#    ax.axis('off')

    ax =plt.subplot(1,2,2)
    plt.plot(numpy.arange(num_steps), out_data*0, "k-", linewidth=2)
    plt.plot(numpy.arange(num_steps), out_data, "-")
    plt.ylim(-max_amplitude, max_amplitude)
    plt.xlim(0, max_t_disp)
#    ax.axis('off')

if plot_outlier_amplification_metric:
    num_samples = 500 #10000   1000 for L, 148 for Q at 6 signals
    seed = 15
    numpy.random.seed(seed)
    max_amplitude = 1.5 #2.0
    noise_expo = 2.5
    d_max_disp = 5.2

    t = numpy.linspace(0, numpy.pi, num_samples)
    c1 = 2**0.5 * numpy.cos(t)
    numpy.random.shuffle(c1)
  
    X = numpy.random.normal(size=(num_samples,2))  
    Y = numpy.random.normal(size=(num_samples,2)) 
    Y_noise = 0.7*sgn_expo(numpy.random.normal(size=(num_samples,2)), noise_expo)
    Y = Y + Y_noise
    
    n1 = mdp.nodes.WhiteningNode()
    n1.train(X)
    WX = n1.execute(X)
    print WX.shape
    
    n2 = mdp.nodes.WhiteningNode()
    n2.train(Y)
    WY = n2.execute(Y)
    
    A_WX = metric_magnitude(WX)
    A_WY = metric_magnitude(WY)
    
    indices_A_WX = numpy.argsort(A_WX)   
    indices_A_WY = numpy.argsort(A_WY)   
     
    O_WX = WX[indices_A_WX[num_samples*95/100:],:]
    O_WY = WY[indices_A_WY[num_samples*95/100:],:]
    
    r_WX = A_WX[indices_A_WX[num_samples*95/100]]
    r_WY = A_WY[indices_A_WY[num_samples*95/100]]
    
    f0 = plt.figure()
    plt.suptitle("Outlier Amplification Metric")
    
    ax =plt.subplot(1,2,1)
    c = mpatches.Circle((0, 0), r_WX, fc="g", ec="r", lw=3)
    ax.add_patch(c)
    plt.plot(WX[:,0], WX[:,1], "b.")
    plt.plot(O_WX[:,0], O_WX[:,1], "ro")
    plt.xlim(-d_max_disp, d_max_disp)
    plt.ylim(-3, 4)

    ax =plt.subplot(1,2,2)
    c = mpatches.Circle((0, 0), r_WY, fc="g", ec="r", lw=3)
    ax.add_patch(c)
    plt.plot(WY[:,0], WY[:,1], "b.")
    plt.plot(O_WY[:,0], O_WY[:,1], "ro")
    plt.xlim(-d_max_disp, d_max_disp)
    plt.ylim(-3, 4)




#    plt.ylim(-max_amplitude, max_amplitude)
#    plt.xlim(0, max_t_disp)
#    ax.axis('off')


print "Displaying..."
plt.show()

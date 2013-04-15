import numpy
import mdp
from classifiers_regressions import Simple_2_Stage_Classifier
import svm as libsvm

def distance_squared_Euclidean(v1, v2):
   dif = v1 - v2    
   if len(dif.shape) > 1:
       dif = dif.flatten()
   return numpy.dot(dif, dif)

# from svm.py
#    default_parameters = {
#    'svm_type' : C_SVC,
#    'kernel_type' : RBF,
#    'degree' : 3,
#    'gamma' : 0,        # 1/num_features
#    'coef0' : 0,
#    'nu' : 0.5,
#    'cache_size' : 100,
#    'C' : 1,
#    'eps' : 1e-3,
#    'p' : 0.1,
#    'shrinking' : 1,
#    'nr_weight' : 0,
#    'weight_label' : [],
#    'weight' : [],
#    'probability' : 0
#    }

print "Experiment that shows how a SVM learns to decode slow signals"
factor = numpy.sqrt(2.0)
num_steps = 1000
num_slow_signals = 5
num_blocks = 5
block_size = num_steps / num_blocks
std_noise = numpy.linspace(0.15, 0.45, num_slow_signals)

k = numpy.arange(num_steps) / block_size
t = numpy.linspace(0, numpy.pi, num_steps)

#Best choice seems to be: C_SVC, RBF, C=1, gamma=1/num_classes

#svm_types =[libsvm.C_SVC, libsvm.NU_SVC, libsvm.ONE_CLASS, libsvm.EPSILON_SVR, libsvm.NU_SVR]
svm_types =[libsvm.NU_SVR]
#kernel_types = [libsvm.LINEAR, libsvm.POLY, libsvm.RBF, libsvm.SIGMOID, libsvm.PRECOMPUTED]
kernel_types = [libsvm.RBF]
#[2**-18, 2**-8, 2**-4]
#default_C = 1.0
#C_vals = [2**-18, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**5, 2**7, 2**9]
C_vals = [2**2]
#C_vals = [2**0]
# default_gamma*2**-4, 2**-12, 2**-9, 2**-5, 2**-4, 
default_gamma = 1.0/(num_slow_signals)
gamma_vals = [default_gamma*1.5, default_gamma, default_gamma / 1.5, 1.0/num_blocks, 0.5/num_blocks, 0.5/num_blocks]
#gamma_vals = [0.125/num_blocks]
gamma_vals = [1.0/num_blocks*1.8, 1.0/num_blocks*1.9 , 1.0/num_blocks*1.95 ]
#    'eps' : 1e-3,
epsilons = [0.0001]
epsilons = [0.0001]
ps = [0.0005, 0.00005]

#probabilities = [True, False]
#nus = [0.5, 0.6, 0.7]
#nus = [1.0]
nus = [0.6]
expos = [1.6]
#expos = [2.0]

xx = numpy.zeros((num_steps, num_slow_signals))
for i in range(num_slow_signals):
    xx[:,i] = factor * numpy.cos((i+1)*t) + std_noise[i] * numpy.random.normal(size=(num_steps))

xx_test = numpy.zeros((num_steps, num_slow_signals))
for i in range(num_slow_signals):
    xx_test[:,i] = factor * numpy.cos((i+1)*t) + std_noise[i] * numpy.random.normal(size=(num_steps))
    
subset = numpy.arange(20,120,5)
cl = {}
val = {}
pr = {}
MSE = {}
PMSE = {}
PMSE2 = {}
PMSE3 = {}
R_MSE = {}
for svm_type in svm_types:
    for kernel_type in kernel_types:
        for gamma in gamma_vals:
            for C in C_vals:
                for eps in epsilons:
                    for p in ps:
                        for nu in nus:
                            if svm_type in (libsvm.EPSILON_SVR, libsvm.NU_SVR):
                                print "svm_type= ", svm_type, ", kernel_type= ", kernel_type, "gamma= ", gamma, ", C= ", C, 
                                print "eps=",eps, "nu=", nu
                                svm_node = mdp.contrib.LibSVMNode(probability=True) #What is the meaning of probability?
                                svm_node.train(xx, k)
                                svm_node.stop_training(svm_type=svm_type, kernel_type=kernel_type, C=C, gamma=gamma, eps=eps, nu=nu, p=p)
            
                                for ee, expo in enumerate(expos):                                
                                    PMSE3[(C, gamma, eps, nu, expo, p)] = distance_squared_Euclidean(svm_node.regression2(xx_test), k)/len(k) 
                                print "svm_node.regression2(xx_test[0]) =", svm_node.regression2(xx_test[0])
                                print "svm_node.regression2(xx_test[1]) =", svm_node.regression2(xx_test[1])
                                print "svm_node.regression2(xx_test[num_steps-2]) =", svm_node.regression2(xx_test[num_steps-2])
                                print "svm_node.regression2(xx_test[num_steps-1]) =", svm_node.regression2(xx_test[num_steps-1])
                            else:
                                print "svm_type= ", svm_type, ", kernel_type= ", kernel_type, "gamma= ", gamma, ", C= ", C, 
                                print "eps=",eps, "nu=", nu
                                svm_node = mdp.contrib.LibSVMNode(probability=True)
                                svm_node.train(xx, k)
                                svm_node.stop_training(svm_type=svm_type, kernel_type=kernel_type, C=C, gamma=gamma, eps=eps, nu=nu, p=p)
            
                                classes = svm_node.classify(xx_test)
                                classes = numpy.array(svm_node.label_of_class(classes))
                                
                                print "true labels=", k[subset]
                                print "output classes=    ", classes[subset] 
                                MSE[(C,gamma, eps, nu)] = distance_squared_Euclidean(classes, k)/len(k)
    
                                values = []
                                for ii in range(len(xx_test[subset])):
                                    values.append(svm_node.model.predict_values(xx_test[subset][ii]))
                                print ""
                                print "values=     ", values
                
                                pred = svm_node.probability(xx_test)
                                print "pred=", pred
#                                for i, est in enumerate(pred):
#                                    print "pred[%d]="%i, est
#                                quit()
                                print "pred[%d]="%0, pred[0]
                                print "svm_node.label_of_class(%d)="%0, svm_node.label_of_class(0)     
                                
                                estimate = numpy.zeros((len(classes)))
                                estimate2 = numpy.zeros((len(classes), len(expos)))
                                for i, est in enumerate(pred):
                                    estimate[i] = 0.0
                                    prob0 = numpy.zeros(num_blocks)
                                    for l in est[1].keys():
                                        estimate[i] = estimate[i] + svm_node.label_of_class(l) * est[1][l]
                                        prob0[l] = est[1][l]
    
                                    for ee, expo in enumerate(expos):
                                        prob2 = prob0 ** expo                                
                                        prob2 = prob2/prob2.sum()
                                        for l in range(num_blocks):
                                            estimate2[i, ee] = estimate2[i, ee] + svm_node.label_of_class(l) * prob2[l]                             
    
                                for ee, expo in enumerate(expos):                                
                                    print "estimate2[0, expo=%.3f]="%expo, estimate2[0]
                                    PMSE2[(C, gamma, eps, nu, expo)] = distance_squared_Euclidean(estimate2[:,ee], k)/len(k)
#                                    PMSE3[(C, gamma, eps, nu, expo)] = distance_squared_Euclidean(svm_node.regression(xx_test, expo), k)/len(k) 
    
                                print "estimate[0]=", estimate[0]
                                PMSE[(C,gamma, eps, nu)] = distance_squared_Euclidean(estimate, k)/len(k)
             
                                print "pred=", pred
#                                R_MSE[(C,gamma)] = distance_squared_Euclidean(val[(C,gamma)], k[20:120:5])/len(k[20:120:5])
                        
print "expos=", expos
for svm_type in svm_types:
    for kernel_type in kernel_types:
        for gamma in gamma_vals:
            print "gamma=", gamma
            for C in C_vals:
                print "C=", C
                for eps in epsilons:
                    print "eps=", eps
                    for p in ps:
                        print "p=", p
                        for nu in nus:
                            print "nu=", nu
                            for expo in expos:
                                if svm_type in (libsvm.EPSILON_SVR, libsvm.NU_SVR):                            
                                    print "PMSE3(%.3f) ="%expo, PMSE3[(C,gamma, eps, nu, expo, p)],
                                else:
#                            print "C= ", C, ", gamma= ", gamma, ", class xx_test[20:120:5]=", cl[(C,gamma)]
                                    print "MSE =", MSE[(C,gamma, eps, nu)], "PMSE =", PMSE[(C,gamma, eps, nu)],

                                    print "PMSE2(%.3f) ="%expo, PMSE2[(C, gamma, eps, nu, expo)],
#            print "" 
#                print "RMSE =", R_MSE[(C,gamma)]
#                print "val=", val[(C,gamma)]
#                print "MSE =", MSE[(C,gamma)], "PMSE =", PMSE[(C,gamma)]

print "Comparing with Gaussian Classifier/Regression"
S2SC = Simple_2_Stage_Classifier()
S2SC.train(data=xx, labels=k, block_size=block_size,spacing=None)
c2, l2 = S2SC.classifyGaussian(xx_test)
v2 = S2SC.GaussianRegression(xx_test)

MSE_GClass = distance_squared_Euclidean(l2, k)/len(k)
PMSE_GReg = distance_squared_Euclidean(v2, k)/len(k)
print "MSE_GClass= ", MSE_GClass, "PMSE_GReg= ", PMSE_GReg 

    
#probs = svm_node.probability(xx[0])
#print "probs[0]=", probs

#print "xx.var is", xx.var(axis=0)
#xx = xx / numpy.sqrt(xx.var(axis=0))
#delta = comp_delta(xx)
#print "delta is", delta
#print "var is", xx.var(axis=0)

#best for NU_SVC is nu=0.6, gamma = 0.005, C=256, expo=1.6
#best for C_SVC (25 blocks) is nu=1.0, C=512, gamma=0.125/num_blocks, expo=1.5
#best for EPSILON_SVM 5 blocks is C=2**2, gamma = 1.0/num_blocks, epsilons = 0.01, nus = *

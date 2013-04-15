#Basic Classes useful for classifiying SFA output signals
#By Alberto Escalante, strongly based on/reusing code by Niko Wilbert
# Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 3 August 2009
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott

import numpy
from  classifiers_regressions import *
from  eban_SFA_libs import *

print "****************************************************************************"
print "*****       TESTING CLASSIFICATION / REGRESSION FUNCS          *************"
print "****************************************************************************"


#t = numpy.arange(0.0, 3.1416, 0.1)
t = numpy.arange(0.0, 3.000, 0.1)
input_dim = 3
x = numpy.zeros((input_dim, len(t))) 
x[0] = numpy.cos(t)
x[1] = numpy.cos(2*t)
x[2] = numpy.cos(3*t)

print "x=", x
x = x.T

CVC_L0 = ClosestDistanceClassifier()
means, blocks, b_labels = blocker_computer(x, labels=t, block_size=5, spacing=None)
print "means=", means
CVC_L0.train(means, labels=None)

CVC_nodes_L1 = range(means.shape[0])
for i in range(means.shape[0]):
    CVC_nodes_L1[i] = ClosestDistanceClassifier()
    CVC_nodes_L1[i].train(blocks[i], labels = b_labels[i])

class_x = CVC_L0.classify(x)
print "x classified=", class_x

print "blocks[0] is", blocks[0]
print "labels[0:5] is", b_labels[0][0:5]


for i in range(x.shape[0]):
    xx = x[i].reshape((1,input_dim))
    cx = CVC_L0.classify(xx)
    c_L1= CVC_nodes_L1[cx].classify(xx)
    print xx, " classified to ", cx, " subclass ", c_L1, " with label: ", CVC_nodes_L1[cx].labels[c_L1]
    


t = numpy.arange(0.0, 3.00, 0.001)
input_dim = 2
x = numpy.zeros((input_dim, len(t))) 
for i in range(input_dim):
    x[i] = numpy.cos((i+1) * t)
x = x.T

xn = x + numpy.random.normal(scale=0.15, size = (len(t), input_dim))

print "********************************************"
print "Testing Gaussian Classifier"
print "********************************************"
print "x.shape=", x.shape

#NOTE!!! GAUSSIAN CLASSIFIER MISSERABLY FAILS FOR SMALL BLOCK SIZES!!!!
#IT ALSO FAILS IF A VARIABLE IS STUCK AT ZERO!!!!
GC = mdp.nodes.GaussianClassifierNode()
classes = numpy.arange(x.shape[0]).astype('int')/20
GC.train(xn, classes)
GC.stop_training()

cl = GC.classify(x)
probabilities = GC.class_probabilities(x)
print "cl[100:200]= ", cl[100:200]
print "len(cl)=", len(cl)
print "probs[0]=", probabilities[0]



print "********************************************"
print "Testing Two Stage Classifier                "
print "********************************************"


S2SC = Simple_2_Stage_Classifier()
S2SC.train(data=xn, labels=t, block_size=200,spacing=None)
#print "Simple_2_Stage_Classifier, with %d Classes in L0"%(S2SC.CDC_L0.n_classes)

x2 = numpy.zeros((input_dim, len(t))) 
for i in range(input_dim):
    x2[i] = numpy.cos((i+1) * t)
x2 = x2.T + numpy.random.normal(scale=0.001, size = (len(t), input_dim))


print "t= ", t
c, l = S2SC.classify(x2)
MSD = distance_squared_Euclidean(t, l)/len(t)
print "c= ", c
print "l[10:20]= ", l[40:80]
print "MSD= ", MSD

c2, l2 = S2SC.classifyGaussian(x2)
v2 = S2SC.GaussianRegression(x2)
MSD2 = distance_squared_Euclidean(t, l2)/len(t)
print "l2.shape", l2.shape
print "v2.shape", v2.shape
MSD3 = distance_squared_Euclidean(t, v2)/len(t)
print "c2= ", c2
print "lGaussian[10:20]= ", l2[40:80]
print "vGaussian[10:20]= ", v2[40:80]
print "MSD2= ", MSD2
print "MSD3= ", MSD3

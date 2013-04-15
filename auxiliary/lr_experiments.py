import numpy
import mdp
from classifiers_regressions import Simple_2_Stage_Classifier
import svm as libsvm

def distance_squared_Euclidean(v1, v2):
   dif = v1 - v2    
   if len(dif.shape) > 1:
       dif = dif.flatten()
   return numpy.dot(dif, dif)


print "Experiment that shows how a LR learns to decode slow signals"
factor = numpy.sqrt(2.0)
num_steps = 8500
num_slow_signals = 9000
num_blocks = 25
block_size = num_steps / num_blocks
std_noise = 0.15

k = numpy.arange(num_steps) / block_size + 10
t = numpy.linspace(0, numpy.pi, num_steps)

xx = numpy.zeros((num_steps, num_slow_signals))
for i in range(num_slow_signals):
    xx[:,i] = factor * numpy.cos((i+1)*t) + std_noise * numpy.random.normal(size=(num_steps))

xx_test = numpy.zeros((num_steps, num_slow_signals))
for i in range(num_slow_signals):
    xx_test[:,i] = factor * numpy.cos((i+1)*t) + std_noise * numpy.random.normal(size=(num_steps))

lr_node = mdp.nodes.LinearRegressionNode(with_bias=True, use_pinv=False)
lr_node.train(xx, k.reshape((num_steps,1)))
lr_node.stop_training()
#Best choice seems to be: C_SVC, RBF, C=1, gamma=1/num_classes

y_train = lr_node.execute(xx)
y_test = lr_node.execute(xx_test)

MSE_train = distance_squared_Euclidean(k, y_train.reshape(num_steps))/num_steps
MSE_test = distance_squared_Euclidean(k, y_test.reshape(num_steps))/num_steps

print "MSE_train =", MSE_train
print "MSE_test  =", MSE_test

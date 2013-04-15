import numpy
import mdp
import math


#x0 = numpy.array([[1.0,1], [1,-1],[-1,-1], [-1, 1.0], 
#                  [1.0,1], [1,-1],[-1,-1], [-1, 1.0], 
#                  [1.0,1], [1,-1],[-1,-1], [-1, 1.0], 
#                  [1.0,1], [1,-1],[-1,-1], [-1, 1.0]])
#pca_node= mdp.nodes.PCANode(input_dim=2, output_dim=2)
#pca_node.train(x0)
#pca_node.stop_training()
#print pca_node.explained_variance
#y0 = pca_node.execute(x0)
#print y0

print "testing..."

x0 = numpy.array([[1.0,1], [1,1],[1,-3], [-3, 1.0]])
x0 = numpy.append(x0,x0, axis=0)
x0 = numpy.append(x0,x0, axis=0)
x0 = numpy.append(x0,x0, axis=0)
x0 = numpy.append(x0,x0, axis=0)
x0 = numpy.append(x0,x0, axis=0)
x0 = numpy.append(x0,x0, axis=0)

pca_node= mdp.nodes.WhiteningNode(input_dim=2, output_dim=2)
pca_node.train(x0)
pca_node.stop_training()
print pca_node.explained_variance
y0 = pca_node.execute(x0)
print y0

x1 = numpy.array([[1,-3], [-3,1],[1,1], [1, 1.0]])/ math.sqrt(3.0)
pca_node= mdp.nodes.WhiteningNode(input_dim=2, output_dim=2)
pca_node.train(x1)
pca_node.stop_training()
print pca_node.explained_variance
y1 = pca_node.execute(x1)
print y1

x3 = numpy.array([[ 1, 1,   1, -3], 
                  [ 1, 1,  -3,  1], 
                  [ 1,-3,   1,  1], 
                  [-3, 1.0, 1,  1]])
x3 = numpy.append(x3,x3, axis=0)
x3 = numpy.append(x3,x3, axis=0)
x3 = numpy.append(x3,x3, axis=0)
x3 = numpy.append(x3,x3, axis=0)
x3 = numpy.append(x3,x3, axis=0)
x3 = numpy.append(x3,x3, axis=0)

pca_node= mdp.nodes.WhiteningNode(input_dim=4, output_dim=3)
pca_node.train(x3)
pca_node.stop_training()
print pca_node.explained_variance
y3 = pca_node.execute(x3)
print y3


#expansion()






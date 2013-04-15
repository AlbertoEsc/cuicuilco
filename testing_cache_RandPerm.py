#This code tests for a bug in object_cache involving the RandomPermutation Node
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 15 Jan 2010
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott
import object_cache
import more_nodes

n1 = more_nodes.RandomPermutationNode()
n2 = more_nodes.RandomPermutationNode()

h1 = object_cache.hash_object(n1, m=None, recursion=True, verbose=False)
h2 = object_cache.hash_object(n2, m=None, recursion=True, verbose=False)

print "h1=", h1.hexdigest()
print "h2=", h2.hexdigest()





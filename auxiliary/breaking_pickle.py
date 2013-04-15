import pickle
import sys
sys.path.append("/home/escalafl/workspace/test_SFA2/src")
sys.path.append("/home/escalafl/workspace/hiphi/src/hiphi/utils")

import numpy
#import mdp

import misc
import numpy

#17000 17000
a = numpy.random.random((4000,3000))
misc.pickle_to_disk(a, "/local2/tmp/escalafl/test_pickle12.pckl")

print "Finished Properly!!!"
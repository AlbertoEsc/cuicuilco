import numpy
import time

def multiply(x1, x2):
    return x1 * x2

def pairwise_expansion(x, func, reflexive=True):
    """Computes func(xi, xj) over all possible indices i and j.
    if reflexive==False, only pairs with i!=j are considered
    """
    x_height, x_width = x.shape
    if reflexive==True:
        k=0
        out = numpy.zeros((x_height, x_width*(x_width+1)/2))
    else:
        k=1
        out = numpy.zeros((x_height, x_width*(x_width-1)/2))    
    mask = numpy.triu(numpy.ones((x_width,x_width)), k) > 0.5

    for i in range(0, x_height):
        y1 = x[i].reshape(x_width, 1)
        y2 = x[i].reshape(1, x_width)
        yexp = func(y1, y2)
#        print "yexp=", yexp
        out[i] = yexp[mask]
    return out 

def pairwise_expansion2(x, func, reflexive=True):
    """Computes func(xi, xj) over all possible indices i and j.
    if reflexive==False, only pairs with i!=j are considered
    """
    x_height, x_width = x.shape
    if reflexive==True:
        k=0
        out = numpy.zeros((x_height, x_width*(x_width+1)/2))
    else:
        k=1
        out = numpy.zeros((x_height, x_width*(x_width-1)/2))    
    mask = numpy.triu(numpy.ones((x_width,x_width)), k) > 0.5

    y1 = x.reshape(x_height, x_width, 1)
    y2 = x.reshape(x_height, 1, x_width)
    yexp = func(y1, y2)
#        print "yexp=", yexp
    out = yexp[mask]
    return out 


def pairwise_adjacent_expansion(x, adj, func, reflexive=True):
    """Computes func(xi, xj) over a subset of all possible indices i and j
    in which abs(j-i) <= adj
    if reflexive==False, only pairs with i!=j are considered
    """
    x_height, x_width = x.shape
    if reflexive is True:
        k=0
    else:
        k=1
#   number of variables, to which the first variable is paired/connected
    mix = adj-k
    out = numpy.zeros((x_height, (x_width-adj+1)*mix))
#
    vars = numpy.array(range(x_width))
    v1 = numpy.zeros(mix * (x_width-adj+1), dtype='int')
    for i in range(x_width-adj+1):
        v1[i*mix:(i+1)*mix] = i
#
    v2 = numpy.zeros(mix * (x_width-adj+1), dtype='int')
    for i in range(x_width-adj+1):
        v2[i*mix:(i+1)*mix] = range(i+k,i+adj)
#    
    for i in range(x_height):
        out[i] = map(func, x[i][v1], x[i][v2])
#        print "yexp=", yexp
#        out[i] = yexp[mask]
    return out  


def pairwise_adjacent_expansion2(x, adj, func, reflexive=True):
    """Computes func(xi, xj) over a subset of all possible indices i and j
    in which abs(j-i) <= adj
    if reflexive==False, only pairs with i!=j are considered
    """
    x_height, x_width = x.shape
    if reflexive is True:
        k=0
    else:
        k=1
#   number of variables, to which the first variable is paired/connected
    mix = adj-k
    out = numpy.zeros((x_height, (x_width-adj+1)*mix))
#
    vars = numpy.array(range(x_width))
    v1 = numpy.zeros(mix * (x_width-adj+1), dtype='int')
    for i in range(x_width-adj+1):
        v1[i*mix:(i+1)*mix] = i
#
    v2 = numpy.zeros(mix * (x_width-adj+1), dtype='int')
    for i in range(x_width-adj+1):
        v2[i*mix:(i+1)*mix] = range(i+k,i+adj)
#   
#    print "v1=", v1 
#    print "v2=", v2 
#    print "x.shape is", x.shape
#    print "x[:,v1].shape=", x[:,v1].shape
#    print "x[:,v2].shape=", x[:,v2].shape

    out = func(x[:,v1], x[:,v2])
#        print "yexp=", yexp
#        out[i] = yexp[mask]
    return out  

#x = numpy.arange(0,300000).reshape(10000,30) / 10.0
#t0 = time.time()
#y1 = pairwise_expansion(x, multiply, reflexive=True)
##print "y1 = ", y1
#t1 = time.time()
#y2 = pairwise_expansion2(x, multiply, reflexive=True)
##print "y2 = ", y2
#t2 = time.time()
#
#print 'Original: %0.3f s' % ((t1-t0))
#print 'Improved: %0.3f s' % ((t2-t1))


x = numpy.arange(0,300000).reshape(10000,30) / 10.0
t0 = time.time()
y1 = pairwise_adjacent_expansion(x, 5, multiply, reflexive=True)

print "y1 = ", y1
t1 = time.time()
y2 = pairwise_adjacent_expansion2(x, 5, multiply, reflexive=True)
print "y2 = ", y2
t2 = time.time()

print 'Original: %0.3f s' % ((t1-t0))
print 'Improved: %0.3f s' % ((t2-t1))


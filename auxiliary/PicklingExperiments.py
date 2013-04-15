import cPickle
import numpy

import sys

sys.path.append("/home/escalafl/workspace/hiphi/src/hiphi/utils")
import misc

x = numpy.arange(80)
x = x.reshape((8,10))

obj = x

class chunk_iterator(object):
    def __init__(self, x, chunk_size):
        self.x = x
        self.chunk_size = chunk_size
        self.index = 0
        self.len_input = x.shape[0]
        if x.shape[0]%chunk_size != 0:
            Ex = Exception("Incorrect Chunk size %d should be a divisor of %d"%(chunk_size, x.shape[0]))
            raise Ex
    def __iter__(self):
        self.index = 0
        return self
    def __getitem__(self, item):
        if item  >= self.len_input:
            raise StopIteration
        ret = self.x[item*self.chunk_size:(item+1)*self.chunk_size, :]
        print "get item"
        return ret        
    def reset(self):
        self.index = 0
    def next(self):
#        print "i=", self.index
        if self.index >= self.len_input:
            raise StopIteration
        try:
            ret = self.x[self.index:self.index + self.chunk_size, :]
            self.index = self.index + self.chunk_size
        except:
            print "oooppssssss!!!"
            raise StopIteration 
        return ret
    
w = chunk_iterator(x, 2)

#print "block:", w.next()
#print "block:", w.next()
#print "block:", w.next()
#print "block:", w.next()
#print "block:", w.next()
#print "block:", w.next()
#print "block:", w.next()

for xx in w:
    print "blockA:", xx
    
for xx in w:
    print "blockB:", xx

print "Saving to file..."
misc.save_iterable(w, path="/home/escalafl/tmp")
    
y = misc.UnpickleLoader(path="/home/escalafl/tmp", recursion=False, verbose=False, pickle_ext=misc.PICKLE_EXT)

print "Retrieving from file..."   
for xx in y:
    print "block:", xx


def double(x):
    return 2*x

def exec_iterator_gen(func, iter):
    for x in iter:
        yield func(x)


class exec_iterator(object):
    def __init__(self, func, input_iter):
        self.input_iter = input_iter
        self.func = func
        print "created"
    def __getitem__(self, item):
        ret = self.func(self.input_iter[item])
        print "get item"
        return ret
    def reset(self):
        self.input_iter.reset()
    def __iter__(self):
        print "iter"
        self.input_iter.reset()
        return self
    def next(self):
        print "next"
        x = self.input_iter.next()
#        print "x=",x
        ret = self.func(x)
        return ret

#self.input_iter.next()
#exec_iter = exec_iterator_gen(double, w)
#
#for xx in exec_iter:
#    print "block0:", xx
#
#for xx in exec_iter:
#    print "block1:", xx

w.reset()
exec_iter2 = exec_iterator(double, w)

for xx in exec_iter2:
    print "block2:", xx

for xx in exec_iter2:
    print "block3:", xx


    
#! /usr/bin/env python

#Test file for the cache and hash functions
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 7 Dec 2009
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott
import numpy
import sys

sys.path.append("/home/escalafl/workspace/hiphi/src/hiphi/utils")
import misc

cache = misc.Cache("/home/scratch_sb/escalafl/tmp_signals", "")

x = numpy.random.normal(size=(10,5))
y = numpy.random.normal(size=(15,5)) 
z = (x, y)

hz = cache.update_cache(z, base_filename="testz", verbose=True)
hx = cache.update_cache(x, base_filename="testx", verbose=True)
hy = cache.update_cache(y, base_filename="testy", verbose=True)
 
print "splitted testq?", cache.is_splitted_file_in_filesystem(base_filename="testq")
print "splitted testz?", cache.is_splitted_file_in_filesystem(base_filename="testz")
print "splitted testx?", cache.is_splitted_file_in_filesystem(base_filename="testx")
print "splitted testy?", cache.is_splitted_file_in_filesystem(base_filename="testy")
print "Expected: False, (True), True, True"
print
print "testq?", cache.is_file_in_filesystem(base_filename="testq")
print "testz?", cache.is_file_in_filesystem(base_filename="testz"+"_%s"%hz)
print "testx?", cache.is_file_in_filesystem(base_filename="testx"+"_%s"%hx)
print "testy?", cache.is_file_in_filesystem(base_filename="testy"+"_%s"%hy)
print "Expected: False, True, False, False"

zz = cache.load_obj_from_cache(hash_value=hz, base_filename="testz", verbose=True)
zx = cache.load_array_from_cache(hash_value=hx, base_filename="testx", verbose=True)
zy = cache.load_array_from_cache(hash_value=hy, base_filename="testy", verbose=True)

print "z =", z
print "zz=",zz

if (x == zx).all():
    print "test for x passed"
else:
    print "test for x failed!!!!!"

if (y == zy).all():
    print "test for y passed"
else:
    print "test for y failed!!!!!"

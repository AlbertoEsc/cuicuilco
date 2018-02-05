#####################################################################################################################
# mnist: This module loads the MNIST dataset                                                                        #
#        It is included in the Cuicuilco framework                                                                  #
#                                                                                                                   #
# Hint: execute this file directly (as main) for an example                                                         #
#                                                                                                                   #
# Code slightly modified by Alberto Escalante. Alberto.Escalante@ini.rub.de                                         #
# Ruhr-University-Bochum, Institute for Neural Computation, Group of Prof. Dr. Wiskott                              #
# See the docstrings for the original authors                                                                       #
#####################################################################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros


def read(digits, dataset="training", path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    Downloaded from: http://g.sweyla.com/blog/2012/mnist-numpy/
    """
    print("READING ONE OF THE MNIST DATABASES")
    
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    print("MNIST, native image size: %d x %d pixels" % (rows, cols))
    labels = zeros(N, dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i]*rows*cols:(ind[i]+1)*rows*cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    images = array(images)
    return images, labels

if __name__ == "__main__":
    from pylab import *
    from numpy import *
    import scipy
    import scipy.misc
    import numpy
    import Image

    images, labels = read([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'training', '/home/escalafl/Databases/MNIST')

    for i in range(10):
        print("#labes equal to %d is %d" % (i, (labels == i).sum()))
       
    out_width = 6
    out_height = 6
    
    im_small_arr = numpy.zeros((len(labels), out_width, out_height))
    
    crop_size = (out_width, out_height)
    rotation_crop_x0 = 0
    rotation_crop_y0 = 0
    rotation_crop_x1 = 28
    rotation_crop_y1 = 28              
    
    for i, im_arr in enumerate(images):
        im = scipy.misc.toimage(im_arr, mode='L')                  
        
        # Here crop size should not loose any pixel from the rotation window
        crop_coordinates = (rotation_crop_x0, rotation_crop_y0, rotation_crop_x1, rotation_crop_y1)
        im_small = im.transform(crop_size, Image.EXTENT, crop_coordinates, Image.BICUBIC)
        im_small_arr[i] = numpy.asarray(im_small)

    print(labels)
    
    imshow(im_small_arr.mean(axis=0), cmap=cm.gray)
    show()

#Functions for generating several types of non-linear expansions
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 10 Dec 2009
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott

import numpy
import numexpr

nan = numpy.nan


#BASIC ELEMENTWISE TRANSFORMATION FUNCTIONS
#functions suitable for any n-dimensional arrays
#this function is so important that we give it a name although it does nothing
def identity(x):
    return x

# xi <- xi / (k+|xi|^d)
def poly_asymmetric_normalize(x, k=1.0, d=0.6):
    return x / (k+numpy.abs(x)**d)

# x <- x / (k+||x||^d) 
def norm2_normalize(x, k=1.0, d=0.6):
    norm = norm2(x)
    return x / (k+norm**d)

# x <- x / (k+e^(d*||x||)) 
def exponential_normalize(x, k=0.0, d=1.0):
    norm = norm2(x)
    return x / (k+numpy.exp(norm*d))

# xi <- xi / (k+e^(d*||xi||))
def expo_asymmetric_normalize(x, k=0.0, d=1.0):
    factor = k+numpy.exp(numpy.abs(x)*d)
    return x / factor

def neg_expo(x, expo):
    signs = numpy.sign(x)
    y = numpy.abs(x)**expo * signs
    return y

def sgn_expo(x, expo):
    s = numpy.sign(x)
    y = s * numpy.abs(x)**expo
    return y

signed_expo = neg_expo

def new_signed_nl_func(data, expo1=2, expo2=0.5):
    mask = numpy.abs(data) < 1
    res = numpy.zeros(data.shape)

    res[mask] = (signed_expo(data, expo1))[mask]
    res[mask^True] = (signed_expo(data, expo2))[mask^True]   
    return res

#HELPER FUNCTIONS FOR COMBINING 2 ARRAYS
def abs_dif(x1, x2):
    return numpy.abs(x1 - x2)

def abs_sum(x1, x2):
    return numpy.abs(x1 + x2)

def multiply(x1, x2):
    return x1 * x2

def signed_sqrt_multiply(x1, x2):
    z = x1*x2
    return signed_sqrt(z)

def unsigned_sqrt_multiply(x1, x2):
    z = x1*x2
    return unsigned_sqrt(z)

def sqrt_abs_sum(x1, x2):
    return numpy.sqrt(numpy.abs(x1+x2))

def sqrt_abs_dif(x1, x2):
    return numpy.sqrt(numpy.abs(x1-x2))

def multiply_sigmoid(x1, x2, expo=0.4):
    return neg_expo(x1 * x2, expo)

def multiply_sigmoid_02(x1, x2):
    expo=0.2
    return neg_expo(x1 * x2, expo)

def multiply_sigmoid_03(x1, x2):
    expo=0.3
    return neg_expo(x1 * x2, expo)

def multiply_sigmoid_04(x1, x2):
    expo=0.4
    return neg_expo(x1 * x2, expo)

def multiply_sigmoid_04_02(x1, x2):
    return new_signed_nl_func(x1 * x2, 0.4, 0.2)

def multiply_sigmoid_06(x1, x2):
    expo=0.6
    return neg_expo(x1 * x2, expo)

def multiply_sigmoid_08(x1, x2):
    expo=0.8
    return neg_expo(x1 * x2, expo)


def compute_quadratic_indices(dim):
    indices1 = numpy.zeros(dim*(dim+1)/2, dtype=int)
    indices2 = indices1.copy()

    current_pos = 0
    for x in range(dim):
        indices1[current_pos:current_pos+(dim-x)] = x
        current_pos += (dim-x)

    all_vars = numpy.arange(dim)
    current_pos = 0
    for x in range(dim):
        indices2[current_pos:current_pos+(dim-x)] = all_vars[x:]
        current_pos += (dim-x)
        
    return indices1, indices2 

quadratic_indices_dict = {}
def products_2_fast(x):
    dim = x.shape[1]
    if dim not in quadratic_indices_dict:
        indices1, indices2 = compute_quadratic_indices(dim)
        quadratic_indices_dict[dim] = (indices1, indices2)
    else:
        indices1, indices2 = quadratic_indices_dict[dim]
    return x[:, indices1] * x[:, indices2]

########## FUNCTIONS APPLIED TO PAIRS OF ELEMENTS, POSSIBLY REPEATEDLY EX: f(f(x_i,x_j),x_k) for various i, j, k. ###########
def pairwise_expansion(x, func, reflexive=True):
    """Computes func(xi, xj) over all possible indices i and j.
    if reflexive==False, only pairs with i!=j are considered
    """
    x_height, x_width = x.shape
    if reflexive==True:
        k=0
        ### out = numpy.zeros((x_height, x_width*(x_width+1)/2))
    else:
        k=1
        ### out = numpy.zeros((x_height, x_width*(x_width-1)/2))    
    mask = numpy.triu(numpy.ones((x_width,x_width)), k) > 0.5
#    mask = mask.reshape((1,x_width,x_width))
    y1 = x.reshape(x_height, x_width, 1)
    y2 = x.reshape(x_height, 1, x_width)
    yexp = func(y1, y2)
    
#    print "yexp.shape=", yexp.shape
#    print "mask.shape=", mask.shape
    out = yexp[:, mask]  
#    print "out.shape=", out.shape
    #yexp.reshape((x_height, N*N))
    return out 


#FUNCTIONS THAT COMBINE TWO OR MORE ELEMENTS
#default k=0 (all pairs), k=1 omits pairs (x_j, x_j)
def products_2(x, func, k=0):
    x_height, x_width = x.shape

    mask = numpy.triu(numpy.ones((x_width,x_width)), k) > 0.5

    z1 = x.reshape(x_height, x_width, 1)
    z2 = x.reshape(x_height, 1, x_width)
    yexp = func(z1, z2) # twice computation, but performance gain due to lack of loops

    out = yexp[:, mask]
    return out 


#Computes: func(func(xi,xj),xk) for all i <= j <= k 
def products_3(x, func):
    _, x_width = x.shape

    res = []
    for i in range(x_width):
        for j in range(i, x_width):
            prod1 = func(x[:,i:i+1], x[:,j:j+1])
            for k in range(j, x_width):
                prod2 = func(x[:,k:k+1], prod1)
                res.append(prod2)
    out = numpy.concatenate(res,axis=1)
    return out


#Computes: func(func(func(xi,xj),xk),xl) for all i <= j <= k <= l
def products_4(x, func):
    _, x_width = x.shape

    res = []
    for i in range(x_width):
        for j in range(i, x_width):
            prod1 = func(x[:,i:i+1], x[:,j:j+1])
            for k in range(j, x_width):
                prod2 = func(x[:,k:k+1], prod1)
                for l in range(k, x_width):
                    prod3 = func(x[:,l:l+1], prod2)
                    res.append(prod3)
    out = numpy.concatenate(res,axis=1)
    return out


#Computes: func(func(func(func(xi,xj),xk),xl),xm) for all i <= j <= k <= l <= m
def products_5(x, func):
    _, x_width = x.shape

    res = []
    for i in range(x_width):
        for j in range(i, x_width):
            prod1 = func(x[:,i:i+1], x[:,j:j+1])
            for k in range(j, x_width):
                prod2 = func(x[:,k:k+1], prod1)
                for l in range(k, x_width):
                    prod3 = func(x[:,l:l+1], prod2)
                    for m in range(l, x_width):
                        prod4 = func(x[:,m:m+1], prod3)
                        res.append(prod4)
    out = numpy.concatenate(res,axis=1)
    return out


#Computes: func(func(func(func(func(xi,xj),xk),xl),xm),xn) for all i <= j <= k <= l <= m <= n
def products_6(x, func):
    _, x_width = x.shape

    res = []
    for i in range(x_width):
        for j in range(i, x_width):
            prod1 = func(x[:,i:i+1], x[:,j:j+1])
            for k in range(j, x_width):
                prod2 = func(x[:,k:k+1], prod1)
                for l in range(k, x_width):
                    prod3 = func(x[:,l:l+1], prod2)
                    for m in range(l, x_width):
                        prod4 = func(x[:,m:m+1], prod3)
                        for n in range(m, x_width):
                            prod5 = func(x[:,n:n+1], prod4)
                            res.append(prod5)
    out = numpy.concatenate(res,axis=1)
    return out


#Computes: func(func(func(func(func(func(xi,xj),xk),xl),xm),xn),x0) for all i <= j <= k <= l <= m <= n <= o
def products_7(x, func):
    _, x_width = x.shape

    res = []
    for i in range(x_width):
        for j in range(i, x_width):
            prod1 = func(x[:,i:i+1], x[:,j:j+1])
            for k in range(j, x_width):
                prod2 = func(x[:,k:k+1], prod1)
                for l in range(k, x_width):
                    prod3 = func(x[:,l:l+1], prod2)
                    for m in range(l, x_width):
                        prod4 = func(x[:,m:m+1], prod3)
                        for n in range(m, x_width):
                            prod5 = func(x[:,n:n+1], prod4)
                            for o in range(n, x_width):
                                prod6 = func(x[:,o:o+1], prod5)
                                res.append(prod6)
    out = numpy.concatenate(res,axis=1)
    return out


#Computes: func(func(func(func(func(func(func(xi1,xi2),xi3),xi4),xi5),xi6),xi7, xi8) for all i1 <= i2 <= i3 <= i4 <= i5 <= i6 <= i7 <= i8
def products_8(x, func):
    _, x_width = x.shape

    res = []
    for i1 in range(x_width):
        for i2 in range(i1, x_width):
            prod1 = func(x[:,i1:i1+1], x[:,i2:i2+1])
            for i3 in range(i2, x_width):
                prod2 = func(x[:,i3:i3+1], prod1)
                for i4 in range(i3, x_width):
                    prod3 = func(x[:,i4:i4+1], prod2)
                    for i5 in range(i4, x_width):
                        prod4 = func(x[:,i5:i5+1], prod3)
                        for i6 in range(i5, x_width):
                            prod5 = func(x[:,i6:i6+1], prod4)
                            for i7 in range(i6, x_width):
                                prod6 = func(x[:,i7:i7+1], prod5)
                                for i8 in range(i7, x_width):
                                    prod7 = func(x[:,i8:i8+1], prod6)
                                    res.append(prod7)
    out = numpy.concatenate(res,axis=1)
    return out


#Computes: func(func(func(func(func(func(func(func(xi1,xi2),xi3),xi4),xi5),xi6),xi7),xi8),xi9) for all i1<=i2<=i3<=i4<=i5<=i6<=i7<=i8<=i9
def products_9(x, func):
    _, x_width = x.shape

    res = []
    for i1 in range(x_width):
        for i2 in range(i1, x_width):
            prod1 = func(x[:,i1:i1+1], x[:,i2:i2+1])
            for i3 in range(i2, x_width):
                prod2 = func(x[:,i3:i3+1], prod1)
                for i4 in range(i3, x_width):
                    prod3 = func(x[:,i4:i4+1], prod2)
                    for i5 in range(i4, x_width):
                        prod4 = func(x[:,i5:i5+1], prod3)
                        for i6 in range(i5, x_width):
                            prod5 = func(x[:,i6:i6+1], prod4)
                            for i7 in range(i6, x_width):
                                prod6 = func(x[:,i7:i7+1], prod5)
                                for i8 in range(i7, x_width):
                                    prod7 = func(x[:,i8:i8+1], prod6)
                                    for i9 in range(i8, x_width):
                                        prod8 = func(x[:,i9:i9+1], prod7)
                                        res.append(prod8)
    out = numpy.concatenate(res,axis=1)
    return out


def Q_func(x, func):
    return products_2(x,func)

def C_func(x, func):
    return products_3(x,func)

def P4_func(x, func):
    return products_4(x,func)

def P5_func(x, func):
    return products_5(x,func)

def P6_func(x, func):
    return products_6(x,func)

def P7_func(x, func):
    return products_7(x,func)

def P8_func(x, func):
    return products_8(x,func)

def P9_func(x, func):
    return products_9(x,func)
    
# CONCRETE QUADRATIC AND CUBIC MONOMIAL GENERATION
def QT(x):
    return Q_func(x,multiply)

def CT(x):
    return C_func(x,multiply)

def P4(x):
    return P4_func(x,multiply)

def P5(x):
    return P5_func(x,multiply)

def P6(x):
    return P6_func(x,multiply)

def P7(x):
    return P7_func(x,multiply)

def P8(x):
    return P8_func(x,multiply)

def P9(x):
    return P9_func(x,multiply)

#AN=Asymmetric Normalize
def Q_AN(x,k=1.0,d=0.6):
    xx = poly_asymmetric_normalize(x, k, d)
    return QT(xx)

def C_AN(x,k=1.0,d=0.73):
    xx = poly_asymmetric_normalize(x, k, d)
    return CT(xx)

#N=(Symmetric) Normalize
def Q_N(x,k=1.0,d=0.6):
#    xx = norm2_normalize(x, k, d)
    y = QT(x)/(k+norm2(x)**d)
    #print "Q_N expanded:", y, y.shape
    return y

def C_N(x,k=1.0,d=0.73):
#    xx = norm2_normalize(x, k, d)
    y = CT(x)/(k+norm2(x)**d)
    #print "CT_N expanded:", y, y.shape
    return y

#E=(Symmetric) Exponential Normalize
def Q_E(x,k=1.0,d=0.6):
    norm = norm2(x)
    return QT(x) / (k+numpy.exp(norm*d))

def C_E(x,k=1.0,d=0.73):
    norm = norm2(x)
    return CT(x)/ (k+numpy.exp(norm*d))

#AE=(Asymmetric) Exponential Normalize
def Q_AE(x,k=1.0,d=0.6):
    xx = expo_asymmetric_normalize(x, k, d)
#    print "xx.shape=", xx.shape
    return QT(xx)

def C_AE(x,k=1.0,d=0.73):
    xx = expo_asymmetric_normalize(x, k, d)
    return CT(xx)

#Note: don't know why I called it asymmetric, this transformation seems symmetric to me
#AE=(Asymmetric) Polynomial Normalize: x-> x**d (signed exponentiation)
def QT_AP(x, d=0.4):
    xx = sgn_expo(x, d)
    return QT(xx)

def CT_AP(x, d=0.3):
    xx = sgn_expo(x, d)
    return CT(xx)



#Expansion with terms: f(x_i,x_i), f(x_i,x_i), ... f(x_i,x_i+k), f(x_i+1,x_i+1), ... f(x_n-k,x_n)
#If reflexive=True include terms f(x_j, x_j)
##This (twice commented) version is slower, so now it was replaced by the function bellow
##see experiments_general_expansion for a benchmark
##def pairwise_adjacent_expansion(x, adj, func, reflexive=True):
##    """Computes func(xi, xj) over a subset of all possible indices i and j
##    in which abs(j-i) <= adj
##    if reflexive==False, only pairs with i!=j are considered
##    """
##    x_height, x_width = x.shape
##    if reflexive is True:
##        k=0
##    else:
##        k=1
###   number of variables, to which the first variable is paired/connected
##    mix = adj-k
##    out = numpy.zeros((x_height, (x_width-adj+1)*mix))
###
##    vars = numpy.array(range(x_width))
##    v1 = numpy.zeros(mix * (x_width-adj+1), dtype='int')
##    for i in range(x_width-adj+1):
##        v1[i*mix:(i+1)*mix] = i
###
##    v2 = numpy.zeros(mix * (x_width-adj+1), dtype='int')
##    for i in range(x_width-adj+1):
##        v2[i*mix:(i+1)*mix] = range(i+k,i+adj)
###    
##    for i in range(x_height):
##        out[i] = map(func, x[i][v1], x[i][v2])
###        print "yexp=", yexp
###        out[i] = yexp[mask]
##    return out    
def pairwise_adjacent_expansion_subset(x, adj, func, reflexive=True, k=10):
    return pairwise_adjacent_expansion(x[:,0:k], adj, func, reflexive)

def pairwise_adjacent_expansion(x, adj, func, reflexive=True):
    """Computes func(xi, xj) over a subset of all possible indices i and j
    in which abs(j-i) <= mix, mix=adj-k
    if reflexive==False, only pairs with i!=j are considered
    """
    _, x_width = x.shape
    if reflexive is True:
        k=0
    else:
        k=1
#   number of variables, to which the first variable is paired/connected
    mix = adj-k
    ###out = numpy.zeros((x_height, (x_width-adj+1)*mix))
#
    ###vars = numpy.array(range(x_width))
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
#    print "x[:,v1].shape=", x[:,v1].shape
#    print "x[:,v2].shape=", x[:,v2].shape

    out = func(x[:,v1], x[:,v2])
#        print "yexp=", yexp
#        out[i] = yexp[mask]
    return out  

#Two-Halbs mixed product expansion
def halbs_product_expansion(x, func):
    """Computes func(xi, xj) over a subset of all possible indices i and j
    in which 0<=i<N and N<=j<2N
    where 2N is the dimension size
    """
    x_height, x_width = x.shape
    if x_width%2 != 0:
        ex = "input dimension must be of even!!!"
        raise ex
    N = x_width/2

    y1 = x[:,:N].reshape(x_height, N, 1)
    y2 = x[:,N:].reshape(x_height, 1, N)
    print "y1.shape=", y1.shape
    print "y2.shape=", y2.shape
    yexp = func(y1, y2)
    print "yexp.shape=", yexp.shape
    
    return yexp.reshape((x_height, N*N))

def halbs_multiply_ex(x):
    return halbs_product_expansion(x, multiply)
#
#xx = numpy.arange(20).reshape((5,4))
#print "xx=", xx
#yy = halbs_multiply_ex(xx)
#print "yy=", yy


def div2_sel65_unsigned_08expo_orig(x):
    num_samples, dim = x.shape
    if dim % 2:
        ex = "Input dimensionality is odd, input array cannot be splitted."
        raise Exception(ex)
    split_size = min(dim/2, 65)
    y = numpy.zeros((num_samples, split_size*2))
    y[:, 0:split_size] = unsigned_08expo(x[:,0:split_size])
    y[:, split_size:] = unsigned_08expo(x[:,dim/2:dim/2+split_size])
    return y

def div2_sel80_unsigned_08expo(x):
    return divN_selK_unsigned_08expo(x, 2, 80)

def div2_sel75_unsigned_08expo(x):
    return divN_selK_unsigned_08expo(x, 2, 75)

def div2_sel70_unsigned_08expo(x):
    return divN_selK_unsigned_08expo(x, 2, 70)

def div2_sel65_unsigned_08expo(x):
    return divN_selK_unsigned_08expo(x, 2, 65)

def div2_sel60_unsigned_08expo(x):
    return divN_selK_unsigned_08expo(x, 2, 60)


def divN_selK_unsigned_08expo(x, num_parts, max_feats_per_part):
    num_samples, dim = x.shape
    if dim % num_parts:
        ex = "Input dimensionality is not a multiple of num_parts, input array cannot be splitted."
        raise Exception(ex)
    split_size = min(dim/num_parts, max_feats_per_part)
    orig_part_size =  dim/num_parts
    y = numpy.zeros((num_samples, split_size*num_parts))
    for part in range(num_parts):
        y[:, split_size*part:split_size*(part+1)] = unsigned_08expo(x[:, orig_part_size*part:orig_part_size*part+split_size])
    return y
    
def sel14_QT(x):
    return QT(x[:,0:14])

def sel18_QT(x):
    return QT(x[:,0:18])

def sel20_QT(x):
    return QT(x[:,0:20])

def sel25_QT(x):
    return QT(x[:,0:25])

def sel30_QT(x):
    return QT(x[:,0:30])

def sel35_QT(x):
    return QT(x[:,0:35])

def sel40_QT(x):
    return QT(x[:,0:40])

def sel50_QT(x):
    return QT(x[:,0:50])

def sel60_QT(x):
    return QT(x[:,0:60])

def sel70_QT(x):
    return QT(x[:,0:70])

def sel80_QT(x):
    return QT(x[:,0:80])

def sel8_04QT(x):
    return neg_expo(QT(x[:,0:8]), 0.4)

def sel10_04QT(x):
    return neg_expo(QT(x[:,0:10]), 0.4)

def sel10_045QT(x):
    return neg_expo(QT(x[:,0:10]), 0.45)

def sel14_04QT(x):
    return neg_expo(QT(x[:,0:14]), 0.4)

def sel14_045QT(x):
    return neg_expo(QT(x[:,0:14]), 0.45)

def sel18_04QT(x):
    return neg_expo(QT(x[:,0:18]), 0.4)

def sel25_CT(x):
    return CT(x[:,0:25])

def sel30_CT(x):
    return CT(x[:,0:30])

def sel35_CT(x):
    return CT(x[:,0:35])


def sel90_unsigned_08expo(x):
    return unsigned_08expo(x[:,0:90])

def sel70_unsigned_08expo(x):
    return unsigned_08expo(x[:,0:70])

def sel65_unsigned_08expo(x):
    return unsigned_08expo(x[:,0:65])

def sel60_unsigned_08expo(x):
    return unsigned_08expo(x[:,0:60])

def sel50_unsigned_08expo(x):
    return unsigned_08expo(x[:,0:50])

def sel40_unsigned_08expo(x):
    return unsigned_08expo(x[:,0:40])

def sel30_unsigned_08expo(x):
    return unsigned_08expo(x[:,0:30])

#This function does not pickle!!!!
def sel_exp(num_comp, func):
    return lambda x: func(x[:, 0:num_comp])

#Euclidean Magnitude (Norm2)
#x is an mdp array, and norm2 is computed row-wise
def norm2(x):
    num_samples, dim = x.shape
    return (((x**2).sum(axis=1) / dim )**0.5).reshape((num_samples, 1))

def dist_to_closest_neighbour_old(x):
    num_samples, dim = x.shape
    zero = numpy.zeros((1, dim))*numpy.nan
    distances = numpy.zeros(num_samples)

    for i in range(num_samples):
        x1 = x[i]
        diffA = x[0:i, :]-x1
        diffB = x[i+1:, :]-x1
        diff = numpy.concatenate((diffA, zero, diffB))
        sqr_dist = (diff*diff).sum(axis=1)
        w = numpy.nanmin(sqr_dist)
        distances[i] = w**0.5
    return distances

def dist_to_closest_neighbour(x, y=None):
    if y is None:
        y = x
    num_samples, _ = x.shape
#    zero = numpy.zeros((1, dim))*numpy.nan
    distances = numpy.zeros(num_samples)
#    indices = numpy.zeros(num_samples)

    for i in range(num_samples):
        x1 = x[i]
        diff = y[:, :]-x1
        sqr_dist = (diff*diff).sum(axis=1)
        sqr_dist[sqr_dist == 0.0] = numpy.inf
        index = numpy.argmin(sqr_dist)
        distances[i] = sqr_dist[index]**0.5
#        indices[i] = index
    return distances #, indices

def fix_mean_var(x):
    x = (x-x.mean(axis=0))/x.std(axis=0)
    return x

def FuncListFromExpansion(Exp, k, d):
    f = lambda x: Exp(x, k=k, d=d)
    #WARNING, next line breaks pickling!!!!
    #f.__name__ = "L:"+Exp.__name__ +" k=%f d=%f"%(k,d)
    f.myname = "L:"+Exp.__name__ +" k=%f d=%f"%(k,d)
    return [f,]

















def unsigned_11expo(x):
    return numpy.abs(x) ** 1.1

def signed_11expo(x):
    return neg_expo(x, 1.1)


def unsigned_15expo(x):
    return numpy.abs(x) ** 1.5

def signed_09expo(x):
    return neg_expo(x, 0.9)

def signed_15expo(x):
    return neg_expo(x, 1.5)

def tanh_025_signed_15expo(x):
    return numpy.tanh(0.25 * neg_expo(x, 1.5)) / 0.25

def tanh_05_signed_15expo(x):
    return numpy.tanh(0.50 * neg_expo(x, 1.5)) / 0.5

def tanh_0125_signed_15expo(x):
    return numpy.tanh(0.125 * neg_expo(x, 1.5)) / 0.125


def unsigned_2_08expo(x, expo1=2, expo2=0.8):
    x_abs = numpy.abs(x)
    mask = x_abs < 1
    res = numpy.zeros(x.shape)

    res[mask] = (x_abs ** expo1)[mask]
    res[mask^True] = (x_abs ** expo2)[mask^True]   
    return res

def unsigned_08expo(x):
    return numexpr.evaluate("abs(x)**0.8")

#def unsigned_08expo_slow(x):
#    return numpy.abs(x) ** 0.8

def unsigned_08expo_m1(x):
    return numpy.abs(x-1) ** 0.8


def unsigned_08expo_p1(x):
    return numpy.abs(x+1) ** 0.8

def signed_06expo(x):
    return neg_expo(x, 0.6)

def unsigned_06expo(x):
    return numpy.abs(x) ** 0.6

def signed_08expo(x):
    return neg_expo(x, 0.8)

def signed_sqrt(x):
    return neg_expo(x, 0.5)

def unsigned_sqrt(x):
    return numpy.abs(x) ** 0.5

def signed_sqr(x):
    return neg_expo(x, 2.0)

def e_neg_sqr(x):
    return numpy.exp(-x **2)

#Weird sigmoid
def weird_sig(x):
    x1 = numpy.exp(-x **2)
    x1[x<0] = 2 -x1 [x<0]
    return x1

def weird_sig2(x):
    x1 = numpy.exp(- (x/2) **2 )
    x1[x<0] = 2 -x1 [x<0]
    return x1

def weird_sig_prod(x1, x2):
    z = x1*x2
    k1 = numpy.exp(- numpy.abs(z) **2)
    k1[z<0] = 2 - k1[z<0]
    return k1

def pair_abs_dif_ex(x):
    return pairwise_expansion(x, abs_dif, reflexive=False)

def pair_abs_sum_ex(x):
    return pairwise_expansion(x, abs_sum)

def pair_prod_ex(x):
    return pairwise_expansion(x, multiply)

def pair_sqrt_abs_sum_ex(x):
    return pairwise_expansion(x, sqrt_abs_sum)

def pair_sqrt_abs_dif_ex(x):
    return pairwise_expansion(x, sqrt_abs_dif, reflexive=False)

#Only product of strictly consecutive variables
def pair_prod_mix1_ex(x):
    return pairwise_adjacent_expansion(x, adj=2, func=multiply, reflexive=False)

def pair_prod_mix2_ex(x):
    return pairwise_adjacent_expansion(x, adj=3, func=multiply, reflexive=False)

def pair_prod_mix3_ex(x):
    return pairwise_adjacent_expansion(x, adj=4, func=multiply, reflexive=False)

#Only sqares of input variables
def pair_prod_adj1_ex(x):
    """returns x_i ^ 2 """
    return pairwise_adjacent_expansion(x, adj=1, func=multiply, reflexive=True)


def pair_prodsigmoid_mix1_ex(x):
    """returns x_i * x_i+1"""
    return pairwise_adjacent_expansion(x, adj=2, func=multiply_sigmoid, reflexive=False)

def pair_prodsigmoid_03_mix1_ex(x):
    """returns x_i * x_i+1"""
    return pairwise_adjacent_expansion(x, adj=2, func=multiply_sigmoid_03, reflexive=False)

def pair_prodsigmoid_04_mix1_ex(x):
    """returns x_i * x_i+1"""
    return pairwise_adjacent_expansion(x, adj=2, func=multiply_sigmoid_04, reflexive=False)

def pair_prodsigmoid_04_adj2_ex(x):
    """returns x_i * x_i, x_i * x_i+1"""
    return pairwise_adjacent_expansion(x, adj=2, func=multiply_sigmoid_04, reflexive=True)

def pair_prodsigmoid_04_adj3_ex(x):
    """returns x_i * x_i, x_i * x_i+1"""
    return pairwise_adjacent_expansion(x, adj=3, func=multiply_sigmoid_04, reflexive=True)

def pair_prodsigmoid_04_adj4_ex(x):
    """returns x_i * x_i, x_i * x_i+1"""
    return pairwise_adjacent_expansion(x, adj=4, func=multiply_sigmoid_04, reflexive=True)

def pair_prodsigmoid_04_02_mix1_ex(x):
    """returns x_i * x_i+1"""
    return pairwise_adjacent_expansion(x, adj=2, func=multiply_sigmoid_04_02, reflexive=False)

def pair_prodsigmoid_06_mix1_ex(x):
    """returns x_i * x_i+1"""
    return pairwise_adjacent_expansion(x, adj=2, func=multiply_sigmoid_06, reflexive=False)

def pair_prodsigmoid_08_mix1_ex(x):
    """returns x_i * x_i+1"""
    return pairwise_adjacent_expansion(x, adj=2, func=multiply_sigmoid_08, reflexive=False)

#Squares and product of adjacent variables
def pair_prod_adj2_ex(x):
    """returns x_i ^ 2 and x_i * x_i+1"""
    return pairwise_adjacent_expansion(x, adj=2, func=multiply, reflexive=True)

def pair_prod_adj3_ex(x):
    return pairwise_adjacent_expansion(x, adj=3, func=multiply, reflexive=True)

def pair_prod_adj4_ex(x):
    return pairwise_adjacent_expansion(x, adj=4, func=multiply, reflexive=True)

def pair_prod_adj5_ex(x):
    return pairwise_adjacent_expansion(x, adj=5, func=multiply, reflexive=True)

def pair_prod_adj6_ex(x):
    return pairwise_adjacent_expansion(x, adj=6, func=multiply, reflexive=True)

def pair_sqrt_abs_dif_adj2_ex(x):
    """returns sqrt(abs(x_i - x_i+1)) """
    return pairwise_adjacent_expansion(x, adj=2, func=sqrt_abs_dif, reflexive=False)

def pair_sqrt_abs_dif_adj3_ex(x):
    return pairwise_adjacent_expansion(x, adj=3, func=sqrt_abs_dif, reflexive=False)

def pair_sqrt_abs_dif_adj4_ex(x):
    return pairwise_adjacent_expansion(x, adj=4, func=sqrt_abs_dif, reflexive=False)

def pair_sqrt_abs_dif_adj5_ex(x):
    return pairwise_adjacent_expansion(x, adj=5, func=sqrt_abs_dif, reflexive=False)

def pair_sqrt_abs_dif_adj6_ex(x):
    return pairwise_adjacent_expansion(x, adj=6, func=sqrt_abs_dif, reflexive=False)


#Normalized product of adjacent variables (or squares)
def signed_sqrt_pair_prod_adj2_ex(x):
    """returns f(x_i ^ 2) and f(x_i * x_i+1)"""
    return pairwise_adjacent_expansion(x, adj=2, func=signed_sqrt_multiply, reflexive=True)

def signed_sqrt_pair_prod_adj3_ex(x):
    """returns f(x_i ^ 2), f(x_i * x_i+1) and f(x_i * x_i+2)"""
    return pairwise_adjacent_expansion(x, adj=3, func=signed_sqrt_multiply, reflexive=True)

def signed_sqrt_pair_prod_mix1_ex(x):
    """returns f(x_i * x_i+1)"""
    return pairwise_adjacent_expansion(x, adj=2, func=signed_sqrt_multiply, reflexive=False)

def signed_sqrt_pair_prod_mix2_ex(x):
    """returns f(x_i * x_i+1)"""
    return pairwise_adjacent_expansion(x, adj=3, func=signed_sqrt_multiply, reflexive=False)

def signed_sqrt_pair_prod_mix3_ex(x):
    """returns f(x_i * x_i+1)"""
    return pairwise_adjacent_expansion(x, adj=4, func=signed_sqrt_multiply, reflexive=False)

def unsigned_sqrt_pair_prod_mix1_ex(x):
    """returns f(x_i * x_i+1)"""
    return pairwise_adjacent_expansion(x, adj=2, func=unsigned_sqrt_multiply, reflexive=False)

def unsigned_sqrt_pair_prod_mix2_ex(x):
    """returns f(x_i * x_i+1)"""
    return pairwise_adjacent_expansion(x, adj=3, func=unsigned_sqrt_multiply, reflexive=False)

#CONCRETE EXPANSIONS
def I_exp(x, k=nan, d=nan):
    return x+0.0

def S_exp(x, k=nan, d=0.8):
    return numpy.concatenate((x, numpy.abs(x)**d), axis=1)

def Q_exp(x, k=nan, d=2.0):
    xx = sgn_expo(x, d/2.0)
    qt = QT(xx)
    return numpy.concatenate((xx, qt), axis=1)

def C_exp(x, k=nan, d=3.0):
    xx = sgn_expo(x, d/3.0)
    qt = QT(xx)
    ct = CT(xx)
#    print qe.shape
    return numpy.concatenate((xx, qt, ct), axis=1)
        
def Q_AN_exp(x, k=1.0, d=0.6):
    xx = poly_asymmetric_normalize(x, k, d)
    qt = QT(xx)
    return numpy.concatenate((xx, qt), axis=1)

def C_AN_exp(x, k=1.0, d=0.7333):
    xx = poly_asymmetric_normalize(x, k, d)
    qt = QT(x)
    te = CT(x)
    return numpy.concatenate((xx, qt, te), axis=1)

def Q_N_exp(x, k=1.0, d=0.6):
    lin = norm2_normalize(x+0.0, k, d)
    qe = Q_N(x+0.0, k, d)
    #WAAARNINNGGG
    #te = C_N(x, k, d)
    return numpy.concatenate((lin, qe), axis=1)

def C_N_exp(x, k=1.0, d=0.7333):
    lin = norm2_normalize(x+0.0, k, d)
    qe = Q_N(x+0.0, k, d)
    te = C_N(x+0.0, k, d)
    return numpy.concatenate((lin, qe, te), axis=1)
#    return lin

def Q_E_exp(x, k=1.0, d=1.0):
    lin = exponential_normalize(x, k, d)
    qe = Q_E(x, k, d)
    return numpy.concatenate((lin, qe), axis=1)

def C_E_exp(x, k=1.0, d=1.0):
    lin = exponential_normalize(x, k, d)
    qe = Q_E(x, k, d)
    te = C_E(x, k, d)
    return numpy.concatenate((lin, qe, te), axis=1)

def Q_AE_exp(x, k=1.0, d=1.0):
    xx =  expo_asymmetric_normalize(x, k, d)
    qe = QT(xx)
    return numpy.concatenate((xx, qe), axis=1)

def C_AE_exp(x, k=1.0, d=1.0):
    xx = expo_asymmetric_normalize(x, k, d)
    qe = QT(xx)
    te = CT(xx)
    return numpy.concatenate((xx, qe, te), axis=1)

def Q_AP_exp(x, k=nan, d=1.0):
    xx = sgn_expo(x, d)
    qe = QT(xx)
    return numpy.concatenate((xx, qe), axis=1)

#Warning, identical to C_exp for expo = 3*d
def C_AP_exp(x, k=nan, d=1.0):
    xx = sgn_expo(x, d)
    qe = QT(xx)
    te = CT(xx)
    return numpy.concatenate((xx, qe, te), axis=1)

#LIST VERSIONS OF THE EXPANSIONS
def I_L(k=nan, d=nan):
    return FuncListFromExpansion(I_exp, k=k, d=d)

def S_L(k=nan, d=0.8):
    return FuncListFromExpansion(S_exp, k=k, d=d)

def Q_L(k=nan, d=2.0):
    return FuncListFromExpansion(Q_exp, k=k, d=d)

def C_L(k=nan, d=3.0):
    return FuncListFromExpansion(C_exp, k=k, d=d)
     
def Q_AN_L(k=1.0, d=0.6):
    return FuncListFromExpansion(Q_AN_exp, k=k, d=d)

def C_AN_L(k=1.0, d=0.7333):
    return FuncListFromExpansion(C_AN_exp, k=k, d=d)

def Q_N_L(k=1.0, d=0.6):
    return FuncListFromExpansion(Q_N_exp, k=k, d=d)

def C_N_L(k=1.0, d=0.7333):
    return FuncListFromExpansion(C_N_exp, k=k, d=d)

def Q_E_L(k=1.0, d=1.0):
    return FuncListFromExpansion(Q_E_exp, k=k, d=d)

def C_E_L(k=1.0, d=1.0):
    return FuncListFromExpansion(C_E_exp, k=k, d=d)

def Q_AE_L(k=1.0, d=1.0):
    return FuncListFromExpansion(Q_AE_exp, k=k, d=d)

def C_AE_L(k=1.0, d=1.0):
    return FuncListFromExpansion(C_AE_exp, k=k, d=d)

def Q_AP_L(k=nan, d=1.0):
    return FuncListFromExpansion(Q_AP_exp, k=k, d=d)

def C_AP_L(k=nan, d=1.0):
    return FuncListFromExpansion(C_AP_exp, k=k, d=d)



def MaxE(x):
    return products_2(x,numpy.maximum, k=1)

def sel14_MaxE(x):
    return MaxE(x[:,0:14])

def sel10_MaxE(x):
    return MaxE(x[:,0:10])

def pair_max_mix1_ex(x):
    return pairwise_adjacent_expansion(x, adj=2, func=numpy.maximum, reflexive=False)

def pair_smax_mix1_ex(x):
    return numpy.sign(pairwise_adjacent_expansion(x, adj=2, func=numpy.maximum, reflexive=False))

def pair_smax50_mix1_ex(x):
    return numpy.sign(pairwise_adjacent_expansion(x[:,0:50], adj=2, func=numpy.maximum, reflexive=False))

def pair_smax25_mix1_ex(x):
    return numpy.sign(pairwise_adjacent_expansion(x[:,0:25], adj=2, func=numpy.maximum, reflexive=False))

def pair_max30_mix1_ex(x):
    return pairwise_adjacent_expansion(x[:,0:31], adj=2, func=numpy.maximum, reflexive=False)

def pair_max50_mix1_ex(x):
    return pairwise_adjacent_expansion(x[:,0:51], adj=2, func=numpy.maximum, reflexive=False)

def pair_max70_mix1_ex(x):
    return pairwise_adjacent_expansion(x[:,0:71], adj=2, func=numpy.maximum, reflexive=False)

def pair_max90_mix1_ex(x):
    return pairwise_adjacent_expansion(x[:,0:91], adj=2, func=numpy.maximum, reflexive=False)


def modulation50_adj1_08_ex(x):
    x1 = numpy.sign(x[:, 0:49])
    return x1 * neg_expo(x[:, 1:50], 0.8)

def modulation50_adj1_02_07_ex(x):
    x1 = x[:, 0:50]
    x2 = x[:, 1:51]
    
    return neg_expo(x1, 0.2) * neg_expo(x2, 0.7)


def modulation90_adj1_02_07_ex(x):
    x1 = x[:, 0:90]
    x2 = x[:, 1:91]

    return neg_expo(x1, 0.2) * neg_expo(x2, 0.7)


def media50_adj2_ex(x):
    x1 = x[:, 0:50]
    x2 = x[:, 1:51]
    x3 = x[:, 2:52]

    minimum = numpy.minimum(numpy.minimum(x1,x2),x3)
    maximum = numpy.maximum(numpy.maximum(x1,x2),x3)
    
    return (x1+x2+x3) - (minimum+maximum)

def maximum_mix1_ex(x):
    dim = x.shape[1]
    x1 = x[:, 0:dim-1]
    x2 = x[:, 1:dim]

    return numpy.maximum(x1,x2)

def maximum_50mix3_ex(x):
    x1 = x[:, 0:50]
    x2 = x[:, 1:51]
    x3 = x[:, 2:52]

    maximum = numpy.maximum(numpy.maximum(x1,x2),x3)

    return maximum

def maxpmin_Fmix3_ex(x, F):
    x1 = x[:, 0:F]
    x2 = x[:, 1:F+1]
    x3 = x[:, 2:F+2]

    maximum = numpy.maximum(numpy.maximum(x1,x2),x3)
    minimum = numpy.minimum(numpy.minimum(x1,x2),x3)

    return 0.5*(maximum+minimum)

def maxpmin_25mix3_ex(x):
    return maxpmin_Fmix3_ex(x, 25)

def maxpmin_50mix3_ex(x):
    return maxpmin_Fmix3_ex(x, 50)

def maxpmin_75mix3_ex(x):
    return maxpmin_Fmix3_ex(x, 75)

def maximum_Fmix2_ex(x,F):
    x1 = x[:, 0:F]
    x2 = x[:, 1:F+1]

    maximum = numpy.maximum(x1,x2)

    return maximum

def maximum_25mix2_ex(x):
    return maximum_Fmix2_ex(x,25)

def maximum_50mix2_ex(x):
    return maximum_Fmix2_ex(x,50)

def maximum_75mix2_ex(x):
    return maximum_Fmix2_ex(x,75)

def maximum_99mix2_ex(x):
    return maximum_Fmix2_ex(x,99)


########## Functions that compute 08expo terms with selection ######################
def Fu08_ex(x,F):
    return unsigned_08expo(x[:,0:F])

def s10u08ex(x):
    return Fu08_ex(x, 10)

def s11u08ex(x):
    return Fu08_ex(x, 11)

def s12u08ex(x):
    return Fu08_ex(x, 12)

def s13u08ex(x):
    return Fu08_ex(x, 13)

def s14u08ex(x):
    return Fu08_ex(x, 14)

def s15u08ex(x):
    return Fu08_ex(x, 15)

def s16u08ex(x):
    return Fu08_ex(x, 16)

def s17u08ex(x):
    return Fu08_ex(x, 17)

def s18u08ex(x):
    return Fu08_ex(x, 18)

def s19u08ex(x):
    return Fu08_ex(x, 19)

def s20u08ex(x):
    return Fu08_ex(x, 20)

def s21u08ex(x):
    return Fu08_ex(x, 21)

def s22u08ex(x):
    return Fu08_ex(x, 22)

def s23u08ex(x):
    return Fu08_ex(x, 23)

def s24u08ex(x):
    return Fu08_ex(x, 24)

def s25u08ex(x):
    return Fu08_ex(x, 25)

def s30u08(x):
    return Fu08_ex(x, 30)

def s31u08(x):
    return Fu08_ex(x, 31)

def s32u08(x):
    return Fu08_ex(x, 32)

def s33u08(x):
    return Fu08_ex(x, 33)

def s34u08(x):
    return Fu08_ex(x, 34)

def s35u08(x):
    return Fu08_ex(x, 35)

def s36u08(x):
    return Fu08_ex(x, 36)

def s38u08(x):
    return Fu08_ex(x, 38)

def s40u08ex(x):
    return Fu08_ex(x, 40)

def s50u08ex(x):
    return Fu08_ex(x, 50)

def s60u08ex(x):
    return Fu08_ex(x, 60)

def s70u08ex(x):
    return Fu08_ex(x, 70)

def s75u08ex(x):
    return Fu08_ex(x, 75)

def s100u08ex(x):
    return Fu08_ex(x, 100)


########## Functions that compute 08expo terms with selection and offset ######################
def oO_sS_u08(x, off, S):
    return unsigned_08expo(x[:,off:off+S])

def o4s30u08(x):
    return oO_sS_u08(x, 4, 30)

def o5_s30_u08(x):
    return oO_sS_u08(x, 5, 30)

def o6s30u08(x):
    return oO_sS_u08(x, 6, 30)


################################### Functions that compute maximum of consequtive features with u08 post-processing ######################
def maximum_Fmix2_sE_ex(x,F,expo):
    x1 = x[:, 0:F]
    x2 = x[:, 1:F+1]

    maximum = numpy.maximum(x1,x2)

    return neg_expo(maximum, expo)

def maximum_25mix2_s08_ex(x):
    return maximum_Fmix2_sE_ex(x,25,0.8)

def maximum_50mix2_s08_ex(x):
    return maximum_Fmix2_sE_ex(x,50,0.8)

def maximum_75mix2_s08_ex(x):
    return maximum_Fmix2_sE_ex(x,75,0.8)

def maximum_99mix2_s08_ex(x):
    return maximum_Fmix2_sE_ex(x,99,0.8)


################################### Functions that compute quadratic terms with selection, and 3 channels ######################
def ch3_sF_QT(x, F):
    ch = 3
    s = F
    ns, dim = x.shape
    xs = numpy.zeros((ns, ch * s))
    xs[:,0:s] = x[:,0:s]
    xs[:,s:2*s] = x[:, dim/3:dim/3+s]
    xs[:,2*s:] = x[:, 2*dim/3:2*dim/3+s]
    return QT(xs)

def ch3_s3_QT(x):
    return ch3_sF_QT(x, 3)

def ch3_s4_QT(x):
    return ch3_sF_QT(x, 4)

def ch3_s5_QT(x):
    return ch3_sF_QT(x, 5)

def ch3_s10_QT(x):
    return ch3_sF_QT(x, 10)

################################### Functions that compute quadratic terms with offset, selection, and 3 channels ######################
def ch3_Offset_sF_QT(x, Off, F):
    ch = 3
    s = F
    ns, dim = x.shape
    if (Off+F)*ch > dim:
        er = "Incorrect parameters for ch3_Offset_sF_QT: Off=%d, F=%d, ch=%d, but data dim=%d"%(Off, F, ch, dim)
        raise Exception(er)
    else:
        xs = numpy.zeros((ns, ch * s))
        xs[:,0:s] = x[:,Off:Off+s]
        xs[:,s:2*s] = x[:, dim/3+Off:dim/3+Off+s]
        xs[:,2*s:] = x[:, 2*dim/3+Off:2*dim/3+Off+s]
        return QT(xs)

def ch3o0s2QT(x):
    return ch3_Offset_sF_QT(x, Off=0, F=2)

def ch3o2s2QT(x):
    return ch3_Offset_sF_QT(x, Off=2, F=2)

def ch3o3s2QT(x):
    return ch3_Offset_sF_QT(x, Off=3, F=2)

def ch3o4s2QT(x):
    return ch3_Offset_sF_QT(x, Off=4, F=2)

def ch3o0s3QT(x):
    return ch3_Offset_sF_QT(x, Off=0, F=3)

def ch3o1s3QT(x):
    return ch3_Offset_sF_QT(x, Off=1, F=3)

def ch3o2s3QT(x):
    return ch3_Offset_sF_QT(x, Off=2, F=3)

def ch3o3s3QT(x):
    return ch3_Offset_sF_QT(x, Off=3, F=3)

def ch3o4s3QT(x):
    return ch3_Offset_sF_QT(x, Off=4, F=3)

def ch3o5s3QT(x):
    return ch3_Offset_sF_QT(x, Off=5, F=3)

def ch3o6s3QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=3)

def ch3o7s3QT(x):
    return ch3_Offset_sF_QT(x, Off=7, F=3)

def ch3o8s3QT(x):
    return ch3_Offset_sF_QT(x, Off=8, F=3)

def ch3o9s3QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=3)

def ch3o0s4QT(x):
    return ch3_Offset_sF_QT(x, Off=0, F=4)

def ch3o1s4QT(x):
    return ch3_Offset_sF_QT(x, Off=1, F=4)

def ch3o2s4QT(x):
    return ch3_Offset_sF_QT(x, Off=2, F=4)

def ch3o3s4QT(x):
    return ch3_Offset_sF_QT(x, Off=3, F=4)

def ch3o4s4QT(x):
    return ch3_Offset_sF_QT(x, Off=4, F=4)

def ch3o5s4QT(x):
    return ch3_Offset_sF_QT(x, Off=5, F=4)

def ch3o6s4QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=4)

def ch3o7s4QT(x):
    return ch3_Offset_sF_QT(x, Off=7, F=4)

def ch3o8s4QT(x):
    return ch3_Offset_sF_QT(x, Off=8, F=4)

def ch3o9s4QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=4)

def ch3o9s5QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=5)

def ch3o9s6QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=6)

def ch3o9s7QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=7)

def ch3o6s2QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=2)

def ch3o6s3QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=3)

def ch3o6s4QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=4)

def ch3o7s4QT(x):
    return ch3_Offset_sF_QT(x, Off=7, F=4)

def ch3o0s5QT(x):
    return ch3_Offset_sF_QT(x, Off=0, F=5)

def ch3o1s5QT(x):
    return ch3_Offset_sF_QT(x, Off=1, F=5)

def ch3o2s5QT(x):
    return ch3_Offset_sF_QT(x, Off=2, F=5)

def ch3o3s5QT(x):
    return ch3_Offset_sF_QT(x, Off=3, F=5)

def ch3o4s5QT(x):
    return ch3_Offset_sF_QT(x, Off=4, F=5)

def ch3o5s5QT(x):
    return ch3_Offset_sF_QT(x, Off=5, F=5)

def ch3o6s5QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=5)

def ch3o7s5QT(x):
    return ch3_Offset_sF_QT(x, Off=7, F=5)

def ch3o1s6QT(x):
    return ch3_Offset_sF_QT(x, Off=1, F=6)

def ch3o2s6QT(x):
    return ch3_Offset_sF_QT(x, Off=2, F=6)

def ch3o3s6QT(x):
    return ch3_Offset_sF_QT(x, Off=3, F=6)

def ch3o4s6QT(x):
    return ch3_Offset_sF_QT(x, Off=4, F=6)

def ch3o5s6QT(x):
    return ch3_Offset_sF_QT(x, Off=5, F=6)

def ch3o6s6QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=6)

def ch3o1s7QT(x):
    return ch3_Offset_sF_QT(x, Off=1, F=7)

def ch3o2s7QT(x):
    return ch3_Offset_sF_QT(x, Off=2, F=7)

def ch3o3s7QT(x):
    return ch3_Offset_sF_QT(x, Off=3, F=7)

def ch3o4s7QT(x):
    return ch3_Offset_sF_QT(x, Off=4, F=7)

def ch3o5s7QT(x):
    return ch3_Offset_sF_QT(x, Off=5, F=7)

def ch3o6s7QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=7)

def ch3o2s8QT(x):
    return ch3_Offset_sF_QT(x, Off=2, F=8)

def ch3o3s8QT(x):
    return ch3_Offset_sF_QT(x, Off=3, F=8)

def ch3o4s8QT(x):
    return ch3_Offset_sF_QT(x, Off=4, F=8)

def ch3o5s8QT(x):
    return ch3_Offset_sF_QT(x, Off=5, F=8)

def ch3o6s8QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=8)

def ch3o7s8QT(x):
    return ch3_Offset_sF_QT(x, Off=7, F=8)

def ch3o8s8QT(x):
    return ch3_Offset_sF_QT(x, Off=8, F=8)

def ch3o9s8QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=8)

def ch3o0s9QT(x):
    return ch3_Offset_sF_QT(x, Off=0, F=9)

def ch3o4s9QT(x):
    return ch3_Offset_sF_QT(x, Off=4, F=9)

def ch3o6s9QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=9)

def ch3o7s9QT(x):
    return ch3_Offset_sF_QT(x, Off=7, F=9)

def ch3o8s9QT(x):
    return ch3_Offset_sF_QT(x, Off=8, F=9)

def ch3o9s9QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=9)

def ch3o0s10QT(x):
    return ch3_Offset_sF_QT(x, Off=0, F=10)

def ch3o4s10QT(x):
    return ch3_Offset_sF_QT(x, Off=4, F=10)

def ch3o6s10QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=10)

def ch3o9s10QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=10)

def ch3o6s11QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=11)

def ch3o9s11QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=11)

def ch3o4s12QT(x):
    return ch3_Offset_sF_QT(x, Off=4, F=12)

def ch3o6s12QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=12)

def ch3o9s12QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=12)

def ch3o6s13QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=13)

def ch3o9s13QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=13)

def ch3o4s14QT(x):
    return ch3_Offset_sF_QT(x, Off=4, F=14)

def ch3o6s14QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=14)

def ch3o9s14QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=14)

def ch3o6s15QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=15)

def ch3o9s15QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=15)

def ch3o0s16QT(x):
    return ch3_Offset_sF_QT(x, Off=0, F=16)

def ch3o4s16QT(x):
    return ch3_Offset_sF_QT(x, Off=4, F=16)

def ch3o6s16QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=16)

def ch3o9s16QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=16)

def ch3o9s17QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=17)

def ch3o6s18QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=18)

def ch3o9s18QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=18)

def ch3o9s19QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=19)

def ch3o0s20QT(x):
    return ch3_Offset_sF_QT(x, Off=0, F=20)

def ch3o6s20QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=20)

def ch3o9s20QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=20)

def ch3o9s21QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=21)

def ch3o6s22QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=22)

def ch3o9s22QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=22)

def ch3o9s23QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=23)

def ch3o0s24QT(x):
    return ch3_Offset_sF_QT(x, Off=0, F=24)

def ch3o6s24QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=24)

def ch3o9s24QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=24)

def ch3o9s25QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=25)

def ch3o6s26QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=26)

def ch3o9s26QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=26)

def ch3o9s27QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=27)

def ch3o6s28QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=28)

def ch3o9s28QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=28)

def ch3o9s29QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=29)

def ch3o6s30QT(x):
    return ch3_Offset_sF_QT(x, Off=6, F=30)

def ch3o9s30QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=30)

def ch3o9s31QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=31)

def ch3o9s32QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=32)

def ch3o9s33QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=33)

def ch3o9s34QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=34)

def ch3o9s35QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=35)

def ch3o9s36QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=36)

def ch3o9s37QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=37)

def ch3o9s38QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=38)

def ch3o9s39QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=39)

def ch3o9s40QT(x):
    return ch3_Offset_sF_QT(x, Off=9, F=40)



################################### Functions that compute quadratic terms with selection ######################
def s2QT(x):
    return QT(x[:,0:2])

def s3QT(x):
    return QT(x[:,0:3])

def s4QT(x):
    return QT(x[:,0:4])

def s5QT(x):
    return QT(x[:,0:5])

def s6QT(x):
    return QT(x[:,0:6])

def s7QT(x):
    return QT(x[:,0:7])

def s8QT(x):
    return QT(x[:,0:8])

def s9QT(x):
    return QT(x[:,0:9])

def s10QT(x):
    return QT(x[:,0:10])

def s11QT(x):
    return QT(x[:,0:11])
    
def s12QT(x):
    return QT(x[:,0:12])

def s13QT(x):
    return QT(x[:,0:13])

def s14QT(x):
    return QT(x[:,0:14])

def s15QT(x):
    return QT(x[:,0:15])

def s16QT(x):
    return QT(x[:,0:16])

def s17QT(x):
    return QT(x[:,0:17])

def s18QT(x):
    return QT(x[:,0:18])

def s20QT(x):
    return QT(x[:,0:20])

def s25QT(x):
    return QT(x[:,0:25])

def s30QT(x):
    return QT(x[:,0:30])

def s35QT(x):
    return QT(x[:,0:35])

def s40QT(x):
    return QT(x[:,0:40])

def s45QT(x):
    return QT(x[:,0:45])

def s50QT(x):
    return QT(x[:,0:50])

def s55QT(x):
    return QT(x[:,0:55])

def s60QT(x):
    return QT(x[:,0:60])

def s65QT(x):
    return QT(x[:,0:65])

def s70QT(x):
    return QT(x[:,0:70])

def s75QT(x):
    return QT(x[:,0:75])


################################### Functions that compute cubic terms with selection ######################
def s2CT(x):
    return CT(x[:,0:2])

def s3CT(x):
    return CT(x[:,0:3])

def s4CT(x):
    return CT(x[:,0:4])

def s5CT(x):
    return CT(x[:,0:5])

def s6CT(x):
    return CT(x[:,0:6])

def s7CT(x):
    return CT(x[:,0:7])

def s8CT(x):
    return CT(x[:,0:8])

def s9CT(x):
    return CT(x[:,0:9])

def s10CT(x):
    return CT(x[:,0:10])

def s11CT(x):
    return CT(x[:,0:11])
    
def s12CT(x):
    return CT(x[:,0:12])

def s13CT(x):
    return CT(x[:,0:13])

def s14CT(x):
    return CT(x[:,0:14])

def s15CT(x):
    return CT(x[:,0:15])

def s16CT(x):
    return CT(x[:,0:16])

def s17CT(x):
    return CT(x[:,0:17])

def s18CT(x):
    return CT(x[:,0:18])

def s20CT(x):
    return CT(x[:,0:20])

def s25CT(x):
    return CT(x[:,0:25])

def s30CT(x):
    return CT(x[:,0:30])

def s35CT(x):
    return CT(x[:,0:35])

def s40CT(x):
    return CT(x[:,0:40])


################################### Functions that compute quadratic terms with offset and selection ######################
def Offset_sF_QT(x, Off, F):
    s = F
    dim = x.shape[1]
    if (Off+F) >= dim:
        s = dim - Off    
    xs = x[:,Off:Off+s]
    return QT(xs)
    
def o4s12QT(x):
    return Offset_sF_QT(x, 4, 12)

def o4s13QT(x):
    return Offset_sF_QT(x, 4, 13)

def o4s15QT(x):
    return Offset_sF_QT(x, 4, 15)

def o5s15QT(x):
    return Offset_sF_QT(x, 5, 15)

def o6s15QT(x):
    return Offset_sF_QT(x, 6, 15)

def o7s15QT(x):
    return Offset_sF_QT(x, 7, 15)

def o4s16QT(x):
    return Offset_sF_QT(x, 4, 16)

def o5s16QT(x):
    return Offset_sF_QT(x, 5, 16)

def o6s16QT(x):
    return Offset_sF_QT(x, 6, 16)

def o4s17QT(x):
    return Offset_sF_QT(x, 4, 17)

def o4s18QT(x):
    return Offset_sF_QT(x, 4, 18)

def o6s18QT(x):
    return Offset_sF_QT(x, 6, 18)

def o4s21QT(x):
    return Offset_sF_QT(x, 4, 21)

def o4s24QT(x):
    return Offset_sF_QT(x, 4, 24)

################################### Functions that compute maximum over consequtive pairs of features, exactly F-1 terms ######################
def Fmaximum_mix1_ex(x, F):
    x1 = x[:, 0:F-1]
    x2 = x[:, 1:F]
    return numpy.maximum(x1,x2)

def s10Max(x):
    return Fmaximum_mix1_ex(x, 10)

def s11Max(x):
    return Fmaximum_mix1_ex(x, 11)

def s12Max(x):
    return Fmaximum_mix1_ex(x, 12)

def s13Max(x):
    return Fmaximum_mix1_ex(x, 13)

def s14Max(x):
    return Fmaximum_mix1_ex(x, 14)

def s15Max(x):
    return Fmaximum_mix1_ex(x, 15)

def s16Max(x):
    return Fmaximum_mix1_ex(x, 16)

def s17Max(x):
    return Fmaximum_mix1_ex(x, 17)

def s18Max(x):
    return Fmaximum_mix1_ex(x, 18)

def s19Max(x):
    return Fmaximum_mix1_ex(x, 19)

def s20Max(x):
    return Fmaximum_mix1_ex(x, 20)

def s25Max(x):
    return Fmaximum_mix1_ex(x, 25)


################################### Functions that compute the 08expo terms of the first k features of each channel, 3 channels assumed ######################
def ch3_sF_u08(x, F):
    ch = 3
    ns, dim = x.shape
    if 3*F >= dim:
        return unsigned_08expo(x)
    else:    
        xs = numpy.zeros((ns, ch * F))
        xs[:,0:F] = x[:,0:F]
        xs[:,F:2*F] = x[:, dim/3:dim/3+F]
        xs[:,2*F:] = x[:, 2*dim/3:2*dim/3+F]
        return unsigned_08expo(xs)

def ch3_oO_sF_u08(x, off, F):
    ch = 3
    ns, dim = x.shape
    if (off+F)*ch >= dim:
        return unsigned_08expo(x)
    else:
        xs = numpy.zeros((ns, ch * F))
        xs[:,:F] = x[:,off:F+off]
        xs[:,F:2*F] = x[:, dim/3+off:dim/3+F+off]
        xs[:,2*F:] = x[:, 2*dim/3+off:2*dim/3+F+off]
        return unsigned_08expo(xs)

def ch3s8u08(x):
    return ch3_sF_u08(x, 8)

def ch3s9u08(x):
    return ch3_sF_u08(x, 9)

def ch3s10u08(x):
    return ch3_sF_u08(x, 10)

def ch3o4s10u08(x):
    return ch3_oO_sF_u08(x, 4, 10)

def ch3s11u08(x):
    return ch3_sF_u08(x, 11)

def ch3s12u08(x):
    return ch3_sF_u08(x, 12)

def ch3s13u08(x):
    return ch3_sF_u08(x, 13)

def ch3s14u08(x):
    return ch3_sF_u08(x, 14)

def ch3s15u08(x):
    return ch3_sF_u08(x, 15)

def ch3o5s15u08(x):
    return ch3_oO_sF_u08(x, 5, 15)

def ch3s16u08(x):
    return ch3_sF_u08(x, 16)

def ch3s17u08(x):
    return ch3_sF_u08(x, 17)

def ch3s18u08(x):
    return ch3_sF_u08(x, 18)

def ch3s19u08(x):
    return ch3_sF_u08(x, 19)

def ch3s20u08(x):
    return ch3_sF_u08(x, 20)

def ch3o4s20u08(x):
    return ch3_oO_sF_u08(x, 4, 20)

def ch3o5s20u08(x):
    return ch3_oO_sF_u08(x, 5, 20)

def ch3o6s20u08(x):
    return ch3_oO_sF_u08(x, 6, 20)

def ch3s21u08(x):
    return ch3_sF_u08(x, 21)

def ch3s22u08(x):
    return ch3_sF_u08(x, 22)

def ch3o3s22u08(x):
    return ch3_oO_sF_u08(x, 3, 22)

def ch3s23u08(x):
    return ch3_sF_u08(x, 23)

def ch3s24u08(x):
    return ch3_sF_u08(x, 24)

def ch3s25u08(x):
    return ch3_sF_u08(x, 25)

def ch3o5s25u08(x):
    return ch3_oO_sF_u08(x, 5, 25)

def ch3s26u08(x):
    return ch3_sF_u08(x, 26)

def ch3o3s26u08(x):
    return ch3_oO_sF_u08(x, 3, 26)

def ch3s27u08(x):
    return ch3_sF_u08(x, 27)

def ch3s28u08(x):
    return ch3_sF_u08(x, 28)

def ch3s29u08(x):
    return ch3_sF_u08(x, 29)

def ch3s30u08(x):
    return ch3_sF_u08(x, 30)

def ch3o3s30u08(x):
    return ch3_oO_sF_u08(x, 3, 30)

def ch3o4s30u08(x):
    return ch3_oO_sF_u08(x, 4, 30)

def ch3o5s30u08(x):
    return ch3_oO_sF_u08(x, 5, 30)

def ch3s32u08(x):
    return ch3_sF_u08(x, 32)

def ch3s33u08(x):
    return ch3_sF_u08(x, 33)

def ch3s34u08(x):
    return ch3_sF_u08(x, 34)

def ch3s35u08(x):
    return ch3_sF_u08(x, 35)

def ch3s36u08(x):
    return ch3_sF_u08(x, 36)

def ch3s37u08(x):
    return ch3_sF_u08(x, 37)

def ch3s38u08(x):
    return ch3_sF_u08(x, 38)

def ch3s39u08(x):
    return ch3_sF_u08(x, 39)

def ch3s40u08(x):
    return ch3_sF_u08(x, 40)

def ch3o4s40u08(x):
    return ch3_oO_sF_u08(x, 4, 40)

def ch3s43u08(x):
    return ch3_sF_u08(x, 43)

def ch3s45u08(x):
    return ch3_sF_u08(x, 45)

def ch3s46u08(x):
    return ch3_sF_u08(x, 46)

def ch3s47u08(x):
    return ch3_sF_u08(x, 47)

def ch3s48u08(x):
    return ch3_sF_u08(x, 48)

def ch3s49u08(x):
    return ch3_sF_u08(x, 49)

def ch3o4s49u08(x):
    return ch3_oO_sF_u08(x, 4, 49)

def ch3s50u08(x):
    return ch3_sF_u08(x, 50)

def ch3s52u08(x):
    return ch3_sF_u08(x, 52)

def ch3s54u08(x):
    return ch3_sF_u08(x, 54)

def ch3s55u08(x):
    return ch3_sF_u08(x, 55)

def ch3s57u08(x):
    return ch3_sF_u08(x, 57)

def ch3s58u08(x):
    return ch3_sF_u08(x, 58)

def ch3s60u08(x):
    return ch3_sF_u08(x, 60)

def ch3s64u08(x):
    return ch3_sF_u08(x, 64)

def ch3s65u08(x):
    return ch3_sF_u08(x, 65)

def ch3s70u08(x):
    return ch3_sF_u08(x, 70)

def ch3s72u08(x):
    return ch3_sF_u08(x, 72)

def ch3s74u08(x):
    return ch3_sF_u08(x, 74)

def ch3s76u08(x):
    return ch3_sF_u08(x, 76)

def ch3s78u08(x):
    return ch3_sF_u08(x, 78)

def ch3s80u08(x):
    return ch3_sF_u08(x, 80)

def ch3s82u08(x):
    return ch3_sF_u08(x, 82)

def ch3s84u08(x):
    return ch3_sF_u08(x, 84)

def ch3s86u08(x):
    return ch3_sF_u08(x, 86)

################################### Functions that take the maximum over pais of features among the first k features of each channel, 3 channels assumed ######################
def ch3_sF_max(x, F):
    _, dim = x.shape
    if 3*F >= dim:
        return maximum_mix1_ex(x)
    else:
        x1 = x[:,0:F]
        x2 = x[:, dim/3:dim/3+F]
        x3 = x[:, 2*dim/3:2*dim/3+F]
        return numpy.concatenate((maximum_mix1_ex(x1), maximum_mix1_ex(x2), maximum_mix1_ex(x3)), axis=1)

def ch3_oO_sF_max(x, off, F):
    ch = 3
    _, dim = x.shape

    if (off+F)*ch >= dim:
        return maximum_mix1_ex(x)
    else:
        x1 = x[:, off:F+off]
        x2 = x[:, dim/3+off:dim/3+F+off]
        x3 = x[:, 2*dim/3+off:2*dim/3+F+off]
        return numpy.concatenate((maximum_mix1_ex(x1), maximum_mix1_ex(x2), maximum_mix1_ex(x3)), axis=1)
        
def ch3s5max(x):
    return ch3_sF_max(x, 5)

def ch3o3s5max(x):
    return ch3_oO_sF_max(x, 3, 5)

def ch3o4s5max(x):
    return ch3_oO_sF_max(x, 4, 5)

def ch3s6max(x):
    return ch3_sF_max(x, 6)

def ch3s8max(x):
    return ch3_sF_max(x, 8)

def ch3s9max(x):
    return ch3_sF_max(x, 9)

def ch3s10max(x):
    return ch3_sF_max(x, 10)

def ch3o3s10max(x):
    return ch3_oO_sF_max(x, 3, 10)

def ch3s11max(x):
    return ch3_sF_max(x, 11)

def ch3o4s11max(x):
    return ch3_oO_sF_max(x, 4, 11)

def ch3s12max(x):
    return ch3_sF_max(x, 12)

def ch3o3s12max(x):
    return ch3_oO_sF_max(x, 3, 12)

def ch3s13max(x):
    return ch3_sF_max(x, 13)

def ch3s14max(x):
    return ch3_sF_max(x, 14)

def ch3s15max(x):
    return ch3_sF_max(x, 15)

def ch3o5s15max(x):
    return ch3_oO_sF_max(x, 5, 15)

def ch3s16max(x):
    return ch3_sF_max(x, 16)

def ch3o3s16max(x):
    return ch3_oO_sF_max(x, 3, 16)

def ch3s17max(x):
    return ch3_sF_max(x, 17)

def ch3s18max(x):
    return ch3_sF_max(x, 18)

def ch3s19max(x):
    return ch3_sF_max(x, 19)

def ch3s20max(x):
    return ch3_sF_max(x, 20)

def ch3s21max(x):
    return ch3_sF_max(x, 21)

def ch3s22max(x):
    return ch3_sF_max(x, 22)

def ch3s23max(x):
    return ch3_sF_max(x, 23)

def ch3s25max(x):
    return ch3_sF_max(x, 25)

def ch3s27max(x):
    return ch3_sF_max(x, 27)


################################### Functions that select the first k features ######################
def sF_head(x,F):
    return x[:,0:F]

def s10(x):
    return sF_head(x,10)

def s12(x):
    return sF_head(x,12)

def s14(x):
    return sF_head(x,14)

def s15(x):
    return sF_head(x,15)

def s16(x):
    return sF_head(x,16)

def s17(x):
    return sF_head(x,17)

def s18(x):
    return sF_head(x,18)

def s20(x):
    return sF_head(x,20)

def s25(x):
    return sF_head(x,25)

def s30(x):
    return sF_head(x,30)

def s50(x):
    return sF_head(x,50)

def s55(x):
    return sF_head(x,55)

def s60(x):
    return sF_head(x,60)

def s62(x):
    return sF_head(x,62)

def s64(x):
    return sF_head(x,64)

def s65(x):
    return sF_head(x,65)

def s66(x):
    return sF_head(x,66)

def s68(x):
    return sF_head(x,68)

def s70(x):
    return sF_head(x,70)

def s72(x):
    return sF_head(x,72)

def s75(x):
    return sF_head(x,75)

def s80(x):
    return sF_head(x,80)

def s85(x):
    return sF_head(x,85)


################################### Functions that select the first k features of each channel, 3 channels assumed ######################
def ch3_sF_head(x, F):
    _, dim = x.shape
    if 3*F >= dim:
        return x 
    x1 = x[:,0:F]
    x2 = x[:, dim/3:dim/3+F]
    x3 = x[:, 2*dim/3:2*dim/3+F]
    return numpy.concatenate((x1,x2,x3), axis=1)

def ch3s6(x):
    return ch3_sF_head(x,6)

def ch3s7(x):
    return ch3_sF_head(x,7)

def ch3s8(x):
    return ch3_sF_head(x,8)

def ch3s9(x):
    return ch3_sF_head(x,9)

def ch3s10(x):
    return ch3_sF_head(x,10)

def ch3s11(x):
    return ch3_sF_head(x,11)

def ch3s12(x):
    return ch3_sF_head(x,12)

def ch3s14(x):
    return ch3_sF_head(x,14)

def ch3s16(x):
    return ch3_sF_head(x,16)

def ch3s17(x):
    return ch3_sF_head(x,17)

def ch3s18(x):
    return ch3_sF_head(x,18)

def ch3s19(x):
    return ch3_sF_head(x,19)

def ch3s20(x):
    return ch3_sF_head(x,20)

def ch3s23(x):
    return ch3_sF_head(x,23)

def ch3s24(x):
    return ch3_sF_head(x,24)

def ch3s25(x):
    return ch3_sF_head(x,25)

def ch3s26(x):
    return ch3_sF_head(x,26)

def ch3s27(x):
    return ch3_sF_head(x,27)

def ch3s28(x):
    return ch3_sF_head(x,28)

def ch3s29(x):
    return ch3_sF_head(x,29)

def ch3s30(x):
    return ch3_sF_head(x,30)

def ch3s33(x):
    return ch3_sF_head(x,33)

def ch3s34(x):
    return ch3_sF_head(x,34)

def ch3s35(x):
    return ch3_sF_head(x,35)

def ch3s36(x):
    return ch3_sF_head(x,36)

def ch3s37(x):
    return ch3_sF_head(x,37)

def ch3s38(x):
    return ch3_sF_head(x,38)

def ch3s39(x):
    return ch3_sF_head(x,39)

def ch3s40(x):
    return ch3_sF_head(x,40)

def ch3s43(x):
    return ch3_sF_head(x,43)

def ch3s45(x):
    return ch3_sF_head(x,45)

def ch3s46(x):
    return ch3_sF_head(x,46)

def ch3s49(x):
    return ch3_sF_head(x,49)

def ch3s50(x):
    return ch3_sF_head(x,50)

def ch3s51(x):
    return ch3_sF_head(x,51)

def ch3s52(x):
    return ch3_sF_head(x,52)

def ch3s53(x):
    return ch3_sF_head(x,53)

def ch3s54(x):
    return ch3_sF_head(x,54)

def ch3s55(x):
    return ch3_sF_head(x,55)

def ch3s56(x):
    return ch3_sF_head(x,56)

def ch3s57(x):
    return ch3_sF_head(x,57)

def ch3s58(x):
    return ch3_sF_head(x,58)

def ch3s59(x):
    return ch3_sF_head(x,59)

def ch3s60(x):
    return ch3_sF_head(x,60)

def ch3s61(x):
    return ch3_sF_head(x,61)

def ch3s62(x):
    return ch3_sF_head(x,62)

def ch3s63(x):
    return ch3_sF_head(x,63)

def ch3s64(x):
    return ch3_sF_head(x,64)

def ch3s65(x):
    return ch3_sF_head(x,65)

def ch3s70(x):
    return ch3_sF_head(x,70)

def ch3s72(x):
    return ch3_sF_head(x,72)

def ch3s74(x):
    return ch3_sF_head(x,74)

def ch3s75(x):
    return ch3_sF_head(x,75)

def ch3s77(x):
    return ch3_sF_head(x,77)

def ch3s80(x):
    return ch3_sF_head(x,80)

def ch3s82(x):
    return ch3_sF_head(x,82)

def ch3s85(x):
    return ch3_sF_head(x,85)

def ch3s87(x):
    return ch3_sF_head(x,87)

def ch3s89(x):
    return ch3_sF_head(x,89)

def ch3s90(x):
    return ch3_sF_head(x,90)

def ch3s95(x):
    return ch3_sF_head(x,95)

def ch3s105(x):
    return ch3_sF_head(x,105)

def ch3s115(x):
    return ch3_sF_head(x,115)


############### Functions that involve offsets and a three sources of the same size ##############
def ch3_Offset_sF_dD_Q_N(x, Off, F, d):
    ch = 3
    s = F
    ns, dim = x.shape

    xs = numpy.zeros((ns, ch * s))
    xs[:,0:s] = x[:,Off:Off+s]
    xs[:,s:2*s] = x[:, dim/3+Off:dim/3+Off+s]
    xs[:,2*s:] = x[:, 2*dim/3+Off:2*dim/3+Off+s]  
    return Q_N(xs, k=1.0, d=d) # xi xj / (1 + ||x||^2*d)

def ch3_o0_s2_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=0, F=2, d=1)

def ch3_o0_s3_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=0, F=3, d=1)

def ch3_o1_s3_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=1, F=3, d=1)

def ch3_o2_s3_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=2, F=3, d=1)

def ch3_o3_s3_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=3, F=3, d=1)

def ch3_o4_s3_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=4, F=3, d=1)

def ch3_o5_s3_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=5, F=3, d=1)

def ch3_o6_s3_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=6, F=3, d=1)

def ch3_o0_s4_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=0, F=4, d=1)

def ch3_o1_s4_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=1, F=4, d=1)

def ch3_o2_s4_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=2, F=4, d=1)

def ch3_o3_s4_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=3, F=4, d=1)

def ch3_o4_s4_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=4, F=4, d=1)

def ch3_o5_s4_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=5, F=4, d=1)

def ch3_o6_s4_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=6, F=4, d=1)

def ch3_o0_s5_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=0, F=5, d=1)

def ch3_o1_s5_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=1, F=5, d=1)

def ch3_o2_s5_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=2, F=5, d=1)

def ch3_o3_s5_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=3, F=5, d=1)

def ch3_o4_s5_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=4, F=5, d=1)

def ch3_o5_s5_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=5, F=5, d=1)

def ch3_o6_s5_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=6, F=5, d=1)

def ch3_o0_s6_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=0, F=6, d=1)

def ch3_o1_s6_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=1, F=6, d=1)

def ch3_o2_s6_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=2, F=6, d=1)

def ch3_o3_s6_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=3, F=6, d=1)

def ch3_o4_s6_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=4, F=6, d=1)

def ch3_o5_s6_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=5, F=6, d=1)

def ch3_o6_s6_d1_Q_N(x):
    return ch3_Offset_sF_dD_Q_N(x, Off=6, F=6, d=1)


############### Functions that involve offsets and a single source ##############
def Offset_sF_dD_Q_N(x, Off, F, d):
    return Q_N(x[:,Off:Off+F], k=1.0, d=d)

def s9_d1_Q_N(x):
    return Offset_sF_dD_Q_N(x, Off=0, F=9, d=1)

def s10_d1_Q_N(x):
    return Offset_sF_dD_Q_N(x, Off=0, F=10, d=1)

def s11_d1_Q_N(x):
    return Offset_sF_dD_Q_N(x, Off=0, F=11, d=1)

def o4_s13_d1_Q_N(x):
    return Offset_sF_dD_Q_N(x, Off=4, F=17, d=1)

def o4_s15_d1_Q_N(x):
    return Offset_sF_dD_Q_N(x, Off=4, F=17, d=1)

def o4_s17_d1_Q_N(x):
    return Offset_sF_dD_Q_N(x, Off=4, F=17, d=1)

def o5_s13_d1_Q_N(x):
    return Offset_sF_dD_Q_N(x, Off=5, F=17, d=1)

def o5_s15_d1_Q_N(x):
    return Offset_sF_dD_Q_N(x, Off=5, F=17, d=1)

def o5_s17_d1_Q_N(x):
    return Offset_sF_dD_Q_N(x, Off=5, F=17, d=1)

def o6_s13_d1_Q_N(x):
    return Offset_sF_dD_Q_N(x, Off=6, F=17, d=1)

def o6_s15_d1_Q_N(x):
    return Offset_sF_dD_Q_N(x, Off=6, F=17, d=1)

def o6_s17_d1_Q_N(x):
    return Offset_sF_dD_Q_N(x, Off=6, F=17, d=1)



################################ Functions that implement a control mechanism that simulates top-down connections ##################
def two_set_products(c, x):
    """ Computes the product of all components in c and all components in x, 
    where c and x are sets of features that should involve the same number of samples. """
    num_samples, c_dim = c.shape
    num_samples2, x_dim = x.shape
    if num_samples != num_samples2:
        er = "incorrect number of samples in control and data signals: %d vs. %d"%(num_samples,num_samples2)
        raise Exception(er) 
    result = numpy.zeros((num_samples, c_dim * x_dim))
    for i in range(c_dim):
        result[:, i*x_dim:(i+1)*x_dim] = x * c[:, i][:, numpy.newaxis]
    return result

def control4_linear(x):
    return numpy.concatenate((x[:,4:], two_set_products(x[:,0:4], x[:,4:])), axis=1)

def controlC_QTF(x, C, F):
    cc = x[:,0:C] #Control signal   
    xx = numpy.concatenate((x[:,C:], QT(x[:,C:F+C])), axis=1) #Data signal
    return numpy.concatenate((xx, two_set_products(cc, xx)), axis=1)

def control4_QT20(x):
    return controlC_QTF(x, 4, 20)

def control4_QT30(x):
    return controlC_QTF(x, 4, 30)

def control4_QT40(x):
    return controlC_QTF(x, 4, 40)

def control4_QT50(x):
    return controlC_QTF(x, 4, 50)

def control4_QT60(x):
    return controlC_QTF(x, 4, 60)

def control5_QT60(x):
    return controlC_QTF(x, 5, 60)

def control6_QT60(x):
    return controlC_QTF(x, 6, 60)

def control8_QT60(x):
    return controlC_QTF(x, 8, 60)

def control9_QT60(x):
    return controlC_QTF(x, 9, 60)



def control1_QT40(x):
    return controlC_QTF(x, 1, 40)

def control2_QT40(x):
    return controlC_QTF(x, 2, 40)

def control3_QT40(x):
    return controlC_QTF(x, 3, 40)

def control5_QT40(x):
    return controlC_QTF(x, 5, 40)

def control6_QT40(x):
    return controlC_QTF(x, 6, 40)

def control6_QT50(x):
    return controlC_QTF(x, 6, 50)


def control2_QT60(x):
    return controlC_QTF(x, 2, 60)

def controlC_QTF_CTG(x, C, F, G):
    cc = x[:,0:C] # Control signal
    xx = numpy.concatenate((x[:,C:], QT(x[:,C:F+C]), CT(x[:,C:G+C])), axis=1) #Data signal
    return numpy.concatenate((xx, two_set_products(cc, xx)), axis=1)

def control2_QT40_CT10(x):
    return controlC_QTF_CTG(x, 2, 40, 10)

def control2_QT40_CT15(x):
    return controlC_QTF_CTG(x, 2, 40, 15)

def control2_QT40_CT20(x):
    return controlC_QTF_CTG(x, 2, 40, 20)

def control2_QT40_CT25(x):
    return controlC_QTF_CTG(x, 2, 40, 25)

def control3_QT40_CT25(x):
    return controlC_QTF_CTG(x, 3, 40, 25)

def control3_QT50_CT25(x):
    return controlC_QTF_CTG(x, 3, 50, 25)


def control2_QT40_CT30(x):
    return controlC_QTF_CTG(x, 2, 40, 30)

def control2_QT40_CT35(x):
    return controlC_QTF_CTG(x, 2, 40, 35)

def control2_QT50_CT25(x):
    return controlC_QTF_CTG(x, 2, 50, 25)

def control2_QT60_CT25(x):
    return controlC_QTF_CTG(x, 2, 60, 25)

def control2_QT70_CT25(x):
    return controlC_QTF_CTG(x, 2, 70, 25)

def control2_QT75_CT25(x):
    return controlC_QTF_CTG(x, 2, 75, 25)

def control6_QT50_CT10(x):
    return controlC_QTF_CTG(x, 6, 50, 10)

def control6_QT50_CT15(x):
    return controlC_QTF_CTG(x, 6, 50, 15)

def control6_QT50_CT20(x):
    return controlC_QTF_CTG(x, 6, 50, 20)

def QTA_CTB_controlC_QTF_CTG(x, A, B, C, F, G):
    px = numpy.concatenate((x[:,C:], QT(x[:,C:A+C]), CT(x[:,C:B+C])), axis=1)
    xx = numpy.concatenate((x[:,C:], QT(x[:,C:F+C]), CT(x[:,C:G+C])), axis=1)
    cc = x[:,0:C]
    return numpy.concatenate((px, two_set_products(cc, xx)), axis=1)

def QT60_CT30_control6_QT40_CT10(x):
    return QTA_CTB_controlC_QTF_CTG(x, 60, 30, 6, 40, 10)

def QT60_CT35_control6_QT40_CT10(x):
    return QTA_CTB_controlC_QTF_CTG(x, 60, 35, 6, 40, 10)

def QT60_CT25_control3_QT40_CT10(x):
    return QTA_CTB_controlC_QTF_CTG(x, 60, 25, 3, 40, 10)

def QT50_CT20_control9_QT30_CT5(x):
    return QTA_CTB_controlC_QTF_CTG(x, 50, 20, 9, 30, 5)

def QT50_CT20_control5_QT35_CT5(x):
    return QTA_CTB_controlC_QTF_CTG(x, 50, 20, 5, 35, 5)

def QT50_CT20_control5_QT40_CT5(x):
    return QTA_CTB_controlC_QTF_CTG(x, 50, 20, 5, 40, 5)

def QT50_CT20_control5_QT45_CT5(x):
    return QTA_CTB_controlC_QTF_CTG(x, 50, 20, 5, 45, 5)

def QT50_CT20_control5_QT50_CT5(x):
    return QTA_CTB_controlC_QTF_CTG(x, 50, 20, 5, 50, 5)

def QT50_CT20_control5_QT50_CT10(x):
    return QTA_CTB_controlC_QTF_CTG(x, 50, 20, 5, 50, 10)

def QT50_CT20_control5_QT50_CT15(x):
    return QTA_CTB_controlC_QTF_CTG(x, 50, 20, 5, 50, 15)


def QT40_CT30_control5_QT30_CT5(x):
    return QTA_CTB_controlC_QTF_CTG(x, 40, 30, 5, 30, 5)

def QT40_CT25_control5_QT30_CT5(x):
    return QTA_CTB_controlC_QTF_CTG(x, 40, 25, 5, 30, 5)

def QT40_CT35_control5_QT30_CT5(x):
    return QTA_CTB_controlC_QTF_CTG(x, 40, 35, 5, 30, 5)


def QT50_CT20_control9_QT40_CT5(x):
    return QTA_CTB_controlC_QTF_CTG(x, 50, 20, 9, 40, 5)

def QT60_CT30_control9_QT30_CT5(x):
    return QTA_CTB_controlC_QTF_CTG(x, 60, 30, 9, 30, 5)

def QT50_CT20_control9_QT25_CT5(x):
    return QTA_CTB_controlC_QTF_CTG(x, 50, 20, 9, 25, 5)

def QT50_CT20_control9_QT35_CT5(x):
    return QTA_CTB_controlC_QTF_CTG(x, 50, 20, 9, 35, 5)

def QT50_CT20_control9_QT45_CT5(x):
    return QTA_CTB_controlC_QTF_CTG(x, 50, 20, 9, 45, 5)

def QT60_CT35_control4_QT40_CT5(x):
    return QTA_CTB_controlC_QTF_CTG(x, 60, 35, 4, 40, 5)


################################### Functions that compute cubic terms with offset, selection, and 3 channels ######################
def ch3_Offset_sF_CT(x, Off, F):
    ch = 3
    s = F
    ns, dim = x.shape
    if (Off+F)*ch > dim:
        er = "Incorrect parameters for ch3_Offset_sF_CT: Off=%d, F=%d, ch=%d, but data dim=%d"%(Off, F, ch, dim)
        raise Exception(er)
    else:
        xs = numpy.zeros((ns, ch * s))
        xs[:,0:s] = x[:,Off:Off+s]
        xs[:,s:2*s] = x[:, dim/3+Off:dim/3+Off+s]
        xs[:,2*s:] = x[:, 2*dim/3+Off:2*dim/3+Off+s]
        return CT(xs)

################################### Functions that compute forth-degree terms with offset, selection, and 3 channels ######################
def ch3_Offset_sF_P4(x, Off, F):
    ch = 3
    s = F
    ns, dim = x.shape
    if (Off+F)*ch > dim:
        er = "Incorrect parameters for ch3_Offset_sF_P4: Off=%d, F=%d, ch=%d, but data dim=%d"%(Off, F, ch, dim)
        raise Exception(er)
    else:
        xs = numpy.zeros((ns, ch * s))
        xs[:,0:s] = x[:,Off:Off+s]
        xs[:,s:2*s] = x[:, dim/3+Off:dim/3+Off+s]
        xs[:,2*s:] = x[:, 2*dim/3+Off:2*dim/3+Off+s]
        return P4(xs)

####################### FUNCTIONS THAT INVOLVE SIGMOIDS ###############
def extract_sigmoid_features(x, c1, l1):
    if x.shape[1] != c1.shape[0] or c1.shape[1] != len(l1):
        er = "Array dimensions mismatch: x.shape =" + str(x.shape) + ", c1.shape =" + str(c1.shape) + ", l1.shape=" + str(l1.shape)
        print er
        raise Exception(er)   
    s = numpy.dot(x,c1)+l1
    f = numpy.tanh(s)
    return f

####################### Specific function definitions for ch3_Offset_sF_CT ################
def ch3o0s2CT(x):
    return ch3_Offset_sF_CT(x, Off=0, F=2)

def ch3o0s3CT(x):
    return ch3_Offset_sF_CT(x, Off=0, F=3)

def ch3o0s4CT(x):
    return ch3_Offset_sF_CT(x, Off=0, F=4)

def ch3o0s5CT(x):
    return ch3_Offset_sF_CT(x, Off=0, F=5)

def ch3o9s2CT(x):
    return ch3_Offset_sF_CT(x, Off=9, F=2)

def ch3o9s3CT(x):
    return ch3_Offset_sF_CT(x, Off=9, F=3)

def ch3o9s4CT(x):
    return ch3_Offset_sF_CT(x, Off=9, F=4)

def ch3o9s5CT(x):
    return ch3_Offset_sF_CT(x, Off=9, F=5)

def ch3o9s6CT(x):
    return ch3_Offset_sF_CT(x, Off=9, F=6)

def ch3o9s7CT(x):
    return ch3_Offset_sF_CT(x, Off=9, F=7)

def ch3o9s8CT(x):
    return ch3_Offset_sF_CT(x, Off=9, F=8)

def ch3o0s9CT(x):
    return ch3_Offset_sF_CT(x, Off=0, F=9)

def ch3o9s9CT(x):
    return ch3_Offset_sF_CT(x, Off=9, F=9)

def ch3o9s10CT(x):
    return ch3_Offset_sF_CT(x, Off=9, F=10)

def ch3o9s11CT(x):
    return ch3_Offset_sF_CT(x, Off=9, F=11)

def ch3o9s12CT(x):
    return ch3_Offset_sF_CT(x, Off=9, F=12)

def ch3o9s13CT(x):
    return ch3_Offset_sF_CT(x, Off=9, F=13)

def ch3o9s14CT(x):
    return ch3_Offset_sF_CT(x, Off=9, F=14)

def ch3o9s15CT(x):
    return ch3_Offset_sF_CT(x, Off=9, F=15)

def ch3o9s16CT(x):
    return ch3_Offset_sF_CT(x, Off=9, F=16)

def ch3o9s17CT(x):
    return ch3_Offset_sF_CT(x, Off=9, F=17)

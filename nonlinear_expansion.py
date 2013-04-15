#Functions for generating several types of non-linear expansions
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 10 Dec 2009
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott

import numpy
import scipy
import scipy.optimize
import sfa_libs

nan = numpy.nan

def sel_exp(num_comp, func):
    return lambda x: func(x[:, 0:num_comp])

def first_k_signals(x, k=10):
    return x[:,0:k]

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
    num_samples, dim = x.shape
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

#BASIC ELEMENTWISE TRANSFORMATION FUNCTIONS
#functions suitable for any n-dimensional arrays
#this function is so important that we give it a name although it does almost nothing
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
 
#This should be done faster.... iterating over rows is too slow!!!!!
#Expansion with terms: f(x1,x1), f(x1,x2), ... f(x1,xn), f(x2,x2), ... f(xn,xn)
#If reflexive=True include terms f(xj, xj)
##This (twice commented) version is slower, so now it was replaced by the function bellow
##see experiments_general_expansion for a benchmark
##def pairwise_expansion(x, func, reflexive=True):
##    """Computes func(xi, xj) over all possible indices i and j.
##    if reflexive==False, only pairs with i!=j are considered
##    """
##    x_height, x_width = x.shape
##    if reflexive==True:
##        k=0
##        out = numpy.zeros((x_height, x_width*(x_width+1)/2))
##    else:
##        k=1
##        out = numpy.zeros((x_height, x_width*(x_width-1)/2))    
##    mask = numpy.triu(numpy.ones((x_width,x_width)), k) > 0.5
##    for i in range(0, x_height):
##        y1 = x[i].reshape(x_width, 1)
##        y2 = x[i].reshape(1, x_width)
##        yexp = func(y1, y2)
###        print "yexp=", yexp
##        out[i] = yexp[mask]
##    return out    

#FUNCTIONS THAT COMBINE TWO OR MORE ELEMENTS
#Computes: func(func(xi,xj),xk) for all i <= j <= k 
#TODO: Improve logic to avoid repeated factors, ok
def products_3(x, func):
    x_height, x_width = x.shape

    res = []
    for i in range(x_width):
        for j in range(i, x_width):
            prod1 = func(x[:,i:i+1], x[:,j:j+1])
            for k in range(j, x_width):
                prod2 = func(x[:,k:k+1], prod1)
                res.append(prod2)
    out = numpy.concatenate(res,axis=1)
    return out

def products_2(x, func):
    x_height, x_width = x.shape

    k=0
    mask = numpy.triu(numpy.ones((x_width,x_width)), k) > 0.5

    z1 = x.reshape(x_height, x_width, 1)
    z2 = x.reshape(x_height, 1, x_width)
    yexp = func(z1, z2) # twice computation, but performance gain due to lack of loops

    out = yexp[:, mask]
    return out 

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

def Q_func(x, func):
    return products_2(x,func)

def T_func(x, func):
    return products_3(x,func)

# CONCRETE QUADRATIC AND CUBIC MONOMIAL GENERATION
def QE(x):
    return Q_func(x,multiply)

def TE(x):
    return T_func(x,multiply)

#AN=Asymmetric Normalize
def Q_AN(x,k=1.0,d=0.6):
    xx = poly_asymmetric_normalize(x, k, d)
    return QE(xx)

def T_AN(x,k=1.0,d=0.73):
    xx = poly_asymmetric_normalize(x, k, d)
    return TE(xx)

#N=(Symmetric) Normalize
def Q_N(x,k=1.0,d=0.6):
#    xx = norm2_normalize(x, k, d)
    y = QE(x)/(k+norm2(x)**d)
    #print "QE_N expanded:", y, y.shape
    return y

def T_N(x,k=1.0,d=0.73):
#    xx = norm2_normalize(x, k, d)
    y = TE(x)/(k+norm2(x)**d)
    #print "TE_N expanded:", y, y.shape
    return y

#E=(Symmetric) Exponential Normalize
def Q_E(x,k=1.0,d=0.6):
    norm = norm2(x)
    return QE(x) / (k+numpy.exp(norm*d))

def T_E(x,k=1.0,d=0.73):
    norm = norm2(x)
    return TE(x)/ (k+numpy.exp(norm*d))

#AE=(Asymmetric) Exponential Normalize
def Q_AE(x,k=1.0,d=0.6):
    xx = expo_asymmetric_normalize(x, k, d)
#    print "xx.shape=", xx.shape
    return QE(xx)

def T_AE(x,k=1.0,d=0.73):
    xx = expo_asymmetric_normalize(x, k, d)
    return TE(xx)

#AE=(Asymmetric) Polynomial Normalize: x-> x**d (signed exponentiation)
def Q_AP(x, d=0.4):
    xx = sgn_expo(x, d)
    return QE(xx)

def T_AP(x, d=0.3):
    xx = sgn_expo(x, d)
    return TE(xx)
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

def unsigned_11expo(x):
    return numpy.abs(x) ** 1.1

def signed_11expo(x):
    return neg_expo(x, 1.1)


def unsigned_15expo(x):
    return numpy.abs(x) ** 1.5

def signed_15expo(x):
    return neg_expo(x, 1.5)

def tanh_025_signed_15expo(x):
    return numpy.tanh(0.25 * neg_expo(x, 1.5)) / 0.25

def tanh_05_signed_15expo(x):
    return numpy.tanh(0.50 * neg_expo(x, 1.5)) / 0.5

def tanh_0125_signed_15expo(x):
    return numpy.tanh(0.125 * neg_expo(x, 1.5)) / 0.125


def unsigned_2_08expo(x, expo1=2, expo2=0.8):
    abs = numpy.abs(x)
    mask = abs < 1
    res = numpy.zeros(x.shape)

    res[mask] = (abs ** expo1)[mask]
    res[mask^True] = (abs ** expo2)[mask^True]   
    return res

def unsigned_08expo(x):
    return numpy.abs(x) ** 0.8

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
    qe = QE(xx)
    return numpy.concatenate((xx, qe), axis=1)

def T_exp(x, k=nan, d=3.0):
    xx = sgn_expo(x, d/3.0)
    qe = QE(xx)
    te = TE(xx)
#    print qe.shape
    return numpy.concatenate((xx, qe, te), axis=1)
        
def Q_AN_exp(x, k=1.0, d=0.6):
    xx = poly_asymmetric_normalize(x, k, d)
    qe = QE(xx)
    return numpy.concatenate((xx, qe), axis=1)

def T_AN_exp(x, k=1.0, d=0.7333):
    xx = poly_asymmetric_normalize(x, k, d)
    qe = QE(x)
    te = TE(x)
    return numpy.concatenate((xx, qe, te), axis=1)

def Q_N_exp(x, k=1.0, d=0.6):
    lin = norm2_normalize(x+0.0, k, d)
    qe = Q_N(x+0.0, k, d)
    #WAAARNINNGGG
    #te = T_N(x, k, d)
    return numpy.concatenate((lin, qe), axis=1)

def T_N_exp(x, k=1.0, d=0.7333):
    lin = norm2_normalize(x+0.0, k, d)
    qe = Q_N(x+0.0, k, d)
    te = T_N(x+0.0, k, d)
    return numpy.concatenate((lin, qe, te), axis=1)
#    return lin

def Q_E_exp(x, k=1.0, d=1.0):
    lin = exponential_normalize(x, k, d)
    qe = Q_E(x, k, d)
    return numpy.concatenate((lin, qe), axis=1)

def T_E_exp(x, k=1.0, d=1.0):
    lin = exponential_normalize(x, k, d)
    qe = Q_E(x, k, d)
    te = T_E(x, k, d)
    return numpy.concatenate((lin, qe, te), axis=1)

def Q_AE_exp(x, k=1.0, d=1.0):
    xx =  expo_asymmetric_normalize(x, k, d)
    qe = QE(xx)
    return numpy.concatenate((xx, qe), axis=1)

def T_AE_exp(x, k=1.0, d=1.0):
    xx = expo_asymmetric_normalize(x, k, d)
    qe = QE(xx)
    te = TE(xx)
    return numpy.concatenate((xx, qe, te), axis=1)

def Q_AP_exp(x, k=nan, d=1.0):
    xx = sgn_expo(x, d)
    qe = QE(xx)
    return numpy.concatenate((xx, qe), axis=1)

#Warning, identical to T_exp for expo = 3*d
def T_AP_exp(x, k=nan, d=1.0):
    xx = sgn_expo(x, d)
    qe = QE(xx)
    te = TE(xx)
    return numpy.concatenate((xx, qe, te), axis=1)

#LIST VERSIONS OF THE EXPANSIONS
def I_L(k=nan, d=nan):
    return FuncListFromExpansion(I_exp, k=k, d=d)

def S_L(k=nan, d=0.8):
    return FuncListFromExpansion(S_exp, k=k, d=d)

def Q_L(k=nan, d=2.0):
    return FuncListFromExpansion(Q_exp, k=k, d=d)

def T_L(k=nan, d=3.0):
    return FuncListFromExpansion(T_exp, k=k, d=d)
     
def Q_AN_L(k=1.0, d=0.6):
    return FuncListFromExpansion(Q_AN_exp, k=k, d=d)

def T_AN_L(k=1.0, d=0.7333):
    return FuncListFromExpansion(T_AN_exp, k=k, d=d)

def Q_N_L(k=1.0, d=0.6):
    return FuncListFromExpansion(Q_N_exp, k=k, d=d)

def T_N_L(k=1.0, d=0.7333):
    return FuncListFromExpansion(T_N_exp, k=k, d=d)

def Q_E_L(k=1.0, d=1.0):
    return FuncListFromExpansion(Q_E_exp, k=k, d=d)

def T_E_L(k=1.0, d=1.0):
    return FuncListFromExpansion(T_E_exp, k=k, d=d)

def Q_AE_L(k=1.0, d=1.0):
    return FuncListFromExpansion(Q_AE_exp, k=k, d=d)

def T_AE_L(k=1.0, d=1.0):
    return FuncListFromExpansion(T_AE_exp, k=k, d=d)

def Q_AP_L(k=nan, d=1.0):
    return FuncListFromExpansion(Q_AP_exp, k=k, d=d)

def T_AP_L(k=nan, d=1.0):
    return FuncListFromExpansion(T_AP_exp, k=k, d=d)

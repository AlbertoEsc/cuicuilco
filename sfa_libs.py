#Basic Functions related to SFA, MDP, Display, Image Processing, Matplotlib
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 19 Mai 2009
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott

import numpy

#default scales (-1,1) into (0, 255)
#Normalized to [-scale/2, scale/2]
def scale_to(val, av_in=0.0, delta_in=2.0, av_out=127.5, delta_out=255.0, scale=1.0, transf='lin'):
    normalized = scale*(val - av_in) / delta_in
    if transf == 'lin':
        return normalized * delta_out + av_out
    elif transf == 'tanh':
        return numpy.tanh(normalized)*delta_out/2.0 + av_out 
    else:
        raise Exception("Wrong transf in scale_to! (choose from 'lin', 'tanh'")


def repeat_list_elements(l, rep=2):
    reps = range(rep)
    return [item for item in l for _ in reps]

def wider(imag, scale_x=1):
    z = numpy.zeros((imag.shape[0], imag.shape[1]*scale_x))
    for i in range(imag.shape[1]):
        tmp = imag[:,i].reshape(imag.shape[0], 1)
        z[:,scale_x*i:scale_x*(i+1)] = tmp
    return z

def wider_1Darray(x, scale_x=1):
    z = numpy.zeros(x.shape[0]*scale_x)
    for i in range(x.shape[0]):
        z[scale_x*i:scale_x*(i+1)] = x[i]
    return z

# def format_coord(x, y, numcols, numrows, width_factor=1.0, height_factor=1.0):
#     col = int(x/width_factor+0.5)
#     row = int(y/height_factor+0.5)
#     if col>=0 and col<numcols and row>=0 and row<numrows:
#         z = X[row,col]
#         return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
#     else:
#         return 'x=%1.4f, y=%1.4f'%(x, y)

def extend_channel_mask_to_signal_mask(input_dim, channel_mask):
    channel_size = channel_mask.size
    rep = input_dim / channel_size
    if input_dim % channel_size != 0:
        err="incompatible channel_mask length and input_dim"
        raise Exception(err)  
    res = channel_mask.copy()
    for _ in range(rep-1):
        res = numpy.concatenate((res, channel_mask))
    return res
  
def remove_Nones(in_list):
    out_list = []
    for x in in_list:
        if x != None:
            out_list.append(x)
    return out_list

def clipping(x, max_out):
    return x.clip(-max_out, max_out)

def clipping_sigma(x, max_out):
    return max_out * numpy.tanh(x / max_out)

def inv_clipping_sigma(x, max_in):
    xx = x.clip(-0.99*max_in, 0.99*max_in)
    return (max_in * numpy.arctanh(xx / max_in)).clip(-max_in, max_in)

def row_func (func, x):
    return numpy.array(map(func, x))

#Obsolete, there is a method that does this... (with t-1 as denominator)
def comp_variance(x):
    t, num_vars = x.shape
    delta = numpy.zeros(num_vars)
    for row in x:
        delta = delta + numpy.square(row)
    delta = delta / (t)
    return delta

#Returns zero-mean unit-variance samples for data in MDP format
def zero_mean_unit_var(x):
    return (x-x.mean(axis=0))/x.std(axis=0)

#Delta = (1/(N-1)) * sum (x(i+1)-x(i))**2 
def comp_delta(x):
    xderiv = x[1:, :]-x[:-1, :]
    return (xderiv**2).mean(axis=0)

def comp_delta_normalized(x):
    xn = zero_mean_unit_var(x)
    xderiv = xn[1:, :]-xn[:-1, :]
    return (xderiv**2).mean(axis=0)
    
def comp_delta_old(x):
    t, num_vars = x.shape
    delta = numpy.zeros(num_vars)
    xderiv = x[1:, :]-x[:-1, :]
    for row in xderiv:
        delta = delta + numpy.square(row)
    delta = delta / (t-1)
    return delta

def comp_eta_from_t_and_delta(t, delta):
    return t/(2*numpy.pi) * numpy.sqrt(delta)

def comp_eta(x):
    t = x.shape[0]
    return comp_eta_from_t_and_delta(t, comp_delta(x))

#TODO: Inspect this code and clean
#For clustered mode, a sequence has length num_reps, for each block...
def comp_typical_delta_eta(x, block_size, num_reps=10, training_mode='serial'):
    t, num_vars = x.shape

    if block_size is None:
        return (None, None)
    
    #TODO: support non-homogeneous block sizes, and training modes
    if isinstance(block_size, int):
        num_blocks = t / block_size
        if block_size == 1:
            num_reps = 1
    # block_origins = numpy.arange(num_blocks) * block_size
    else:       
        num_blocks = len(block_size)
        block_origins = numpy.zeros(num_blocks)
        current = 0
        for j, bs in enumerate(block_size):
            if j+1 <num_blocks:
                current = current + bs
                block_origins[j+1] = current

    delta = None
    eta = None

    if training_mode in ['serial', "mixed"]:
        delta = numpy.zeros(num_vars)
        eta = numpy.zeros(num_vars)
        test = numpy.zeros((num_blocks, num_vars))

        if isinstance(block_size, int):
            for _ in range(num_reps):
                for j in range(num_blocks):
                    w = numpy.random.randint(block_size)
                    test[j] = x[j * block_size + w]
                delta += comp_delta(test)
                eta += comp_eta(test)
        else:
            for _ in range(num_reps):
                for j in range(num_blocks):
                    w = numpy.random.randint(block_size[j])
                    #print "block_origins[%d]="%j,block_origins[j]
                    test[j] = x[block_origins[j] + w]
                delta += comp_delta(test)
                eta += comp_eta(test)
        delta = delta / num_reps
        eta = eta / num_reps       
    elif training_mode == "clustered"  or training_mode == "compact_classes": #verify this... and make exact computation
        print "exact computation of delta value for clustered graph..."
        #print "x.std(axis=0)=", x.std(axis=0)
        if isinstance(block_size, int):
            block_sizes = [block_size] * (t / block_size)
        else:
            block_sizes = block_size
        delta = 0.0
        current_pos = 0
        for block_size in block_sizes:
            #print "delta=", delta
            #print "block_size=", block_size
            #print "current_pos=", current_pos
            x_cluster = x[current_pos:current_pos+block_size,:]
            x_cluster_avg = x_cluster.mean(axis=0)
            x_cluster_ene = (x_cluster**2).sum(axis=0)
            delta += (1.0/t)*(1.0/(block_size-1))*(2*block_size*x_cluster_ene - 2*block_size**2 * x_cluster_avg**2)
            current_pos += block_size
    elif training_mode is None or training_mode == "regular":
        delta = comp_delta(x)    
    else:
        er = "Training mode unknown:", training_mode, "still, computing delta as a sequence"
        print er
        delta = comp_delta(x)    

    if eta is None:
        eta = comp_eta_from_t_and_delta(t, delta)
    
    return (delta, eta) 

def str2(x):
    c=""
    for w in x:
        if c == "":
            c+="%.2f" % w
        else:
            c+=",%.2f" % w
    return c

def str3(x):
    c=""
    for w in x:
        if c == "":
            c+="%.3f" % w
        else:
            c+=",%.3f" % w
    return c


#WARNING!!! semantics for vectors and matrices are different!!!
def distance_Euclidean(v1, v2):
    dif = v1 - v2
#either flatten it, or try something else
#dot = numpy.dot(dif, dif)
#    return numpy.sqrt(dot)
    return numpy.linalg.norm(dif.flatten(), ord=2)

def distance_squared_Euclidean(v1, v2):
    dif = v1 - v2
    dif = dif.flatten()
    return (dif * dif).sum()

def distance_best_squared_Euclidean(v1, v2):
    dif = v1 - v2    
    if len(dif.shape) > 1:
        dif = dif.flatten()
    return numpy.dot(dif, dif)

def apply_funcs_to_signal(funcs, x):
    out = []
    for func in funcs:
        if out==[]:
            out = func(x)
        else:
            out = numpy.concatenate((out, func(x)), axis=1)
    return out


#Computing Cartesian product of tuples in args
#Original source:
#http://stackoverflow.com/questions/533905/get-the-cartesian-product-of-a-series-of-lists-in-python
def product(*args):
    pools = map(tuple, args)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)
        

def select_rows_from_matrix(M, sel, mode=2):
    if mode == 1:
        return select_rows_from_matrix1(M, sel)
    elif mode == 2:
        return select_rows_from_matrix2(M, sel)
    elif mode == 3:
        return select_rows_from_matrix3(M, sel)
    elif mode == 4:
        return select_rows_from_matrix4(M, sel)
    else:
        return M[:, sel]

def select_rows_from_matrix1(M, sel):
    z = numpy.zeros((M.shape[0], len(sel)))
    for i, r in enumerate(sel):
        z[:,i] = M[:,r]
    return z

def select_rows_from_matrix2(M, sel):
    z = M[:,sel]
    return z

def select_rows_from_matrix3(M, sel):
    Mt = M.T
    z = Mt[sel]
    zt = z.T
    return zt

def select_rows_from_matrix4(M, sel):
    zt = numpy.zeros((len(sel), M.shape[0]))
    Mt = M.T
    for i, r in enumerate(sel):
        zt[i] = Mt[r]
    z = zt.T
    return z

def ndarray_to_string(x, prefix="", col_sep=", ", row_sep="\n", out_filename = None):
    s = "" + prefix
    if not isinstance(x, numpy.ndarray):
        ex = "array is not ndarray"
        raise Exception(ex)

    if x.ndim == 1:
        s = s+"%d Values"%x.shape[0] + row_sep
        for i in range(len(x)):
            if i == 0:
                s += "%f"%x[i]
            else:
                s += col_sep + "%f"%x[i]
    elif x.ndim == 2:
        s = s + "%d  Samples, %d Features"%(x.shape[0], x.shape[1]) + row_sep
        for i, row in enumerate(x):
            if i != 0:
                s = s + row_sep
            for i in range(len(row)):
                if i == 0:
                    s += "%f"%row[i]
                else:
                    s += col_sep + "%f"%row[i]
    else:
        ex = "wrong number of dimensions, should be 1 or 2"
        raise Exception(ex)       
    # print s
    if out_filename != None:
        fileobj = open(out_filename, mode='wb')
        fileobj.write(s)
        fileobj.close()
    return s
    
# x = numpy.linspace(1.0, 10.0, 1000)
# x = x.reshape((4,250))
# ndarray_to_string(x, "/local/tmp/escalafl/test_write.txt")
 
def cutoff(x, min_val, max_val):
    print "Cutoff v 2.0"
    y1 = x
    if not (min_val is None):
        y1 = numpy.where(x >= min_val, x, min_val)
    if not (max_val is None):
        y1 = numpy.where(y1 <= max_val, y1, max_val)
    return y1 

#Experiment that shows unexpected speed problem with indexing 2d arrays by column!!!
test_select_rows_from_matrix_speed = False
if test_select_rows_from_matrix_speed:
    w =  numpy.random.random((6000,3000))
    x = numpy.random.permutation(3000)
    
    import time
    t0 = time.time()
    z1 = select_rows_from_matrix1(w, x)
    print "time used by select_rows_from_matrix1:", time.time()-t0    
    t0 = time.time()
    z2 = select_rows_from_matrix2(w, x)
    print "time used by select_rows_from_matrix2:", time.time()-t0   
    t0 = time.time()
    z3 = select_rows_from_matrix3(w, x)
    print "time used by select_rows_from_matrix3:", time.time()-t0
    t0 = time.time()
    z4 = select_rows_from_matrix4(w, x)
    print "time used by select_rows_from_matrix4:", time.time()-t0
    t0 = time.time()
    z5 = select_rows_from_matrix(w, x, mode=None)
    print "time used by select_rows_from_matrix:", time.time()-t0
    
    print ((z1-z2)**2+(z2-z3)**2+(z3-z4)**2+(z4-z5)**2).sum()

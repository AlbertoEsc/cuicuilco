#Basic Functions related to SFA, MDP, Display, Image Processing, Matplotlib
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 19 Mai 2009
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott
#This file will be replaced by the following modules:
#files, mdp patch, localized_inversion, iterators, non-linear expansion
#classifiers_regressions, image_loader, lattice, network_builder, object_cache

import numpy

#
#
##allows usage of a large array as an iterator
##chunk_size is given in number of blocks
##continuos=False:just split, True: ...
#class chunk_iterator(object):
#    def __init__(self, x, chunk_size, block_size=None, continuous=False, verbose=False):
#        if block_size is None:
#            block_size=1
#        self.block_size = block_size
#        self.x = x
#        if chunk_size < 2:
#            chunk_size = 2
#        self.chunk_size = chunk_size
#        self.index = 0
#        self.len_input = x.shape[0]
#        self.continuous = continuous
#        self.verbose = verbose
# 
#        if self.verbose:
#            print "Creating chunk_iterator with: chunk_size=%d,block_size=%d, continuous=%d, len_input=%d"%(chunk_size,block_size, continuous, self.len_input) 
#        if continuous is False:
#            if x.shape[0]% block_size != 0:
#                er = Exception("Incorrect block_size %d should be a divisor of %d"%(block_size, x.shape[0]))
#                raise Exception(er)
#            if x.shape[0]%(chunk_size * block_size)!= 0:
#                er = "Warning! Chunks of desired size %d using blocks of size %d over signal %d do not have the same size!"%(chunk_size, block_size, x.shape[0])
#                print er            
#        else:
#            if x.shape[0]%block_size != 0 or (x.shape[0]/block_size-1)%(chunk_size-1) != 0:
#                print "WARNING!!!!!!!!!!!!! Last Chunk won't have the same size (%d chunk_size, %d x.shape[0])"%(chunk_size, x.shape[0])        
#    def __iter__(self):
#        self.index = 0
#        return self
#    def next(self):
#        if self.verbose:
#            print "geting next chunk, for i=", self.index
#        if self.index >= self.len_input:
#            raise StopIteration
##        try:
#        ret = self.x[self.index:self.index + self.chunk_size * self.block_size, :]
#        if self.continuous is False:
#            self.index = self.index + self.chunk_size * self.block_size
#        else:
#            self.index = self.index + (self.chunk_size - 1) * self.block_size            
##        except:
##            print "oooppssssss!!!"
##            raise StopIteration 
#        return ret

#def exec_generator(func, iter):
#    for x in iter:
#        yield func(x)

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
    return [item for item in l for r in reps]

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

def format_coord(x, y, numcols, numrows, width_factor=1.0, height_factor=1.0):
    col = int(x/width_factor+0.5)
    row = int(y/height_factor+0.5)
    if col>=0 and col<numcols and row>=0 and row<numrows:
        z = X[row,col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f'%(x, y)


#The following class was borrowed from '/mdp/nodes/expansion_nodes.py'
#class _ExpansionNode(mdp.Node):
#    def __init__(self, input_dim = None, dtype = None):
#        super(_ExpansionNode, self).__init__(input_dim, None, dtype)
#    def expanded_dim(self, dim):
#        return dim
#    def is_trainable(self):
#        return False
#    def is_invertible(self):
#        return False
#    def _set_input_dim(self, n):
#        self._input_dim = n
#        self._output_dim = self.expanded_dim(n)
#    def _set_output_dim(self, n):
#        msg = "Output dim cannot be set explicitly!"
#        raise mdp.NodeException(msg)



    



def extend_channel_mask_to_signal_mask(input_dim, channel_mask):
    channel_size = channel_mask.size
    rep = input_dim / channel_size
    if input_dim % channel_size != 0:
        err="incompatible channel_mask length and input_dim"
        raise Exception(err)  
    res = channel_mask.copy()
    for iter in range(rep-1):
        res = numpy.concatenate((res, channel_mask))
    return res


  
    


#        out = []
#        for func in self.funcs:
#            if out==[]:
#                out = func(x)
#            else:
#                out = numpy.concatenate((out, func(x)), axis=1)
#        return out
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









#WARNING, infinite loop in definition!!!!!!!!!!!!!!!!!!!!!!!
#Computes the coordinates of the lattice points that lie within the image, now channel dimension is considered!!!
#mask is a y_in_ch * x_in_ch * in_ch_dim boolean Matrix where, each 2D [y, x] entry has size in_channel_dim
#mask_flat, and *_flat, refers to the y_in_ch * (x_in_ch * in_ch_dim) boolean Matrix
#careful, some parameters are unused
#def compute_lattice_matrix_with_input_dim(v1, v2, mask, x_in_channels, y_in_channels, in_channel_dim=1, n0_1 = 0, n0_2 = 0, wrap_x= False, wrap_y= False, input_dim = None, dtype = None, ignore_cover = True, allow_nonrectangular_lattice=False):
#    mask_flat = mask.flatten().reshape(x_in_channels * in_channel_dim, y_in_channels)
#
#    v1_flat = v1.copy()
#    v2_flat = v2.copy()
##remember, vectors have coordinates x, y 
#    v1_flat[0] = v1_flat[0] * in_channel_dim
#    v2_flat[0] = v2_flat[0] * in_channel_dim
#
#    x_in_channels_flat = x_in_channels * in_channel_dim
#    y_in_channels_flat = y_in_channels
#    in_channel_dim_flat = 1
#    
#    lat_mat_flat = compute_lattice_matrix_with_input_dim(v1_flat, v2_flat, mask_flat, x_in_channels_flat, y_in_channels_flat, in_channel_dim_flat, allow_nonrectangular_lattice=allow_nonrectangular_lattice) 
#                                          
#    return lat_mat_flat




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

#        return (numpy.nan, numpy.nan)

    delta = None
    eta = None

    if training_mode in ['serial', "mixed"]:
        delta = numpy.zeros(num_vars)
        eta = numpy.zeros(num_vars)
        test = numpy.zeros((num_blocks, num_vars))

        if isinstance(block_size, int):
            for i in range(num_reps):
                for j in range(num_blocks):
                    w = numpy.random.randint(block_size)
                    test[j] = x[j * block_size + w]
                delta += comp_delta(test)
                eta += comp_eta(test)
        else:
            for i in range(num_reps):
                for j in range(num_blocks):
                    w = numpy.random.randint(block_size[j])
    #                print "block_origins[%d]="%j,block_origins[j]
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
        delta = comp_delta(x)    

    if eta is None:
        eta = comp_eta_from_t_and_delta(t, delta)
    
    #print "(delta, eta)=", delta, eta
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


#Computing cartesian product of tuples in args
#Original source:
#http://stackoverflow.com/questions/533905/get-the-cartesian-product-of-a-series-of-lists-in-python
def product(*args):
    pools = map(tuple, args)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)
        

#rename_filenames_facegen()
#x = numpy.ones((2,5))
#pairwise_adjacent_expansion(x, 3, multiply, reflexive=False)

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

#input_dim = 10            
#permutation = numpy.random.permutation(range(input_dim))
#inv_permutation = numpy.zeros(input_dim)
#inv_permutation[permutation] = numpy.arange(input_dim)

#w =  numpy.random.random((5000,2000))
#x = numpy.random.permutation(2000)
#z = numpy.zeros(w.shape)

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
#    print ":)"
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
    
x = numpy.linspace(1.0, 10.0, 1000)
x = x.reshape((4,250))
ndarray_to_string(x, "/local/tmp/escalafl/test_write.txt")
 
#WARNING!!! CUTOFF DISABLED!!!! 
def cutoff(x, min, max):
#    return x
    print "Cutoff v 2.0"
    y1 = x
    if not (min is None):
        y1 = numpy.where(x >= min, x, min)
    if not (max is None):
        y1 = numpy.where(y1 <= max, y1, max)
    return y1 

##Experiment that shows unexpected speed problem with indexing 2d arrays by column!!!
#t0 = time.time()
#z1 = select_rows_from_matrix1(w, x)
#print  time.time()-t0
#
#t0 = time.time()
#z2 = select_rows_from_matrix2(w, x)
#print  time.time()-t0
#
#t0 = time.time()
#z3 = select_rows_from_matrix3(w, x)
#print  time.time()-t0
#
#t0 = time.time()
#z4 = select_rows_from_matrix4(w, x)
#print  time.time()-t0
#
#t0 = time.time()
#z5 = select_rows_from_matrix(w, x, mode=None)
#print  time.time()-t0
#
#print ((z1-z2)**2+(z2-z3)**2+(z3-z4)**2+(z4-z5)**2).sum()

#Produces the outputs:
#1.38612008095
#1.78197717667
#2.05780911446
#0.630693912506




##This hashes any n-dimensional array, and returns a hash object
##It flattens the array on beforehand, and is limited by the resolution of str
#def hash_arrayxxxx(x, m=None): 
#    y = x.flatten()
#    if m is None:
#        m = hashlib.md5() # or sha1 etc
#
#    for value in y: # array contains the data
#        m.update(str(value))
#    return m
#
##This hashes any list of scalar values (or values printable by str), and returns a hash object
##be aware that the uniqueness is limited by the representation capabilities of str
#def hash_listxxxx(x, m=None, recursion=False): 
#    if m is None:
#        m = hashlib.md5() # or sha1 etc
#
#    for value in x: # array contains the data
#        if recursion == False:
#            m.update(str(value))
#        elif isinstance(value, (tuple, list)):
#            m = hash_list(value, m, recursion)
#        else:
#            m.update(str(value))            
#    return m
#
#def get_data_varsxxxx(object):
#    return [var for var in dir(object) if not callable(getattr(object, var))]
#
#def remove_hidden_varxxxx(variable_list):
#    clean_list = []
#    for var in variable_list:
#        if var[0] != "_":
#            clean_list.append(var)
#    return clean_list
#
##Warning, setting recursion to False causes the hash of [x] and x to be different
##Recursion should usually be true, but avoid circular references!!!!
#def hash_objectxxxx(obj, m=None, recursion=True, verbose=True): 
#    print "Hashing ", obj
#    
#    if m is None:
#        m = hashlib.md5() # or sha1 etc
#
#    #Case 0: obj is enumerable
#    if isinstance(obj, (tuple, list)):
#        if verbose:
#            print "Hashing each list element"
#        for var in obj:
#            m = hash_object(var, m, recursion)
#        return m
#
#    #Case 1: obj is an array
#    if isinstance(obj, numpy.ndarray):
#        if verbose:
#            print "Hashing as an array:", obj
#        m = hash_array(obj, m) 
#        return m
#
#    dataList = get_data_vars(obj)
#    dataList = remove_hidden_vars(dataList)
#
#    #Case 2: obj is an scalar value, or an "empty" obj
#    if len(dataList) == 0:
#        if verbose:
#            print "Hashing as single scalar:", str(obj)
#        m.update(str(obj))
#        return m
#
#    #Case 3: obj has data attributes
#    if verbose:
#        print "Hashing as an object..."
#
#    for var in dataList:
#        if recursion == False:
#            if verbose:
#                print "Simple Hashing:", getattr(obj, var)
#            m.update(str(getattr(obj, var)))
#        else:
#            if verbose:
#                print "Considering attribute:", var
#                print "Hashing as obj:", getattr(obj, var)
##            if len(remove_hidden_vars(get_data_vars(obj))) > 0:
#            m = hash_object(getattr(object, var), m, recursion)
##            else:
##                m.update(str(getattr(object, variable)))
#    return m



        
        
        
        

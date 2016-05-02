#Classes and functions useful to hash objects/arrays, and to store them in a cache (hard drive)
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 9 Dec 2009
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott
#(Original) classes and functions for pickling/unpickling courtesy of Niko Wilbert
#see hiphi/utils/misc.py

import sys, os
import inspect
import fnmatch

#Chose either cPickle or pickle
#import cPickle as pickle
import pickle
import numpy
import hashlib
import types

PICKLE_EXT = ".pckl"  # default filename extension for pickle files
PICKLE_PROTO = -1


def pickle_to_disk(obj, filename, pickle_proto=PICKLE_PROTO, overwrite=True):
    """Pickle an object to disk.
    Requires a complete filename    
    """
    if overwrite == False:
        print "Error, overwriting is in this code always enabled"
        
    pickle_file = open(filename, "wb")
    try:
        pickle.dump(obj, pickle_file, protocol=pickle_proto)
    finally:
        pickle_file.close( )


#requires complete filename        
def unpickle_from_disk(filename):
    """Unpickle an object from disk.

    Requires a complete filename      
    Passes Exception if file could not be loaded.
    """
    # Warning: only using 'r' or 'w' can result in EOFError when unpickling! 
    pickle_file = open(filename, "rb")
    try:
        obj = pickle.load(pickle_file)
    finally:
        pickle_file.close()
    return obj


def save_iterable(iterable, path=""):
    """Take an iterable and pickle the data to disk.
    
    Output filenames are called seq_XXXX.pckl for some integer XXXX
    """
    try:
        os.makedirs(path)
    except:
        pass
    for i_chunk, chunk in enumerate(iterable):
        # print "rendered seq. %d" % i_seq
        pickle_to_disk(chunk, os.path.join(path, 
                                           "seq_%04d" % i_chunk + PICKLE_EXT))


def save_iterable2(iterable, path="", basefilename="seq"):
    """Take an iterable and pickle the data to disk.
        
    Output files are called basefilename_SXXXX.pckl, where XXXX is an integer
    """
    try:
        os.makedirs(path)
    except:
        pass
    for i_chunk, chunk in enumerate(iterable):
        # print "rendered seq. %d" % i_seq
        pickle_to_disk(chunk, os.path.join(path, basefilename + "_S%04d" % i_chunk + PICKLE_EXT))


class UnpickleLoader2(object):
    """Load and unpickle files in the given directory, implements iterable.  
    Only those files are unpickled that end with the given extension, and
    are prefixed with baseilename
    Can be used together with save_iterable2.
    """
    
    def __init__(self, path="", basefilename="seq", recursion=False, verbose=False, 
                 pickle_ext=PICKLE_EXT):
        """Make a list of all the files to unpickle.
        
        keyword arguments:
        verbose -- output which files are unpickled
        pickle_ext -- filename extension for the pickle files
        """
        self.verbose = verbose
        self.path = path
        self.filenames = []
        if verbose:
            print "Searching for prefix:", basefilename
        for root, _, files in os.walk(path):
            for filename in files:
                if os.path.splitext(filename)[1] == pickle_ext:
                    txt = os.path.splitext(filename)[0]
                    if txt[0:len(basefilename)] == basefilename: 
                        self.filenames.append(os.path.join(root, filename))
                        if verbose:
                            print "file %s taken"%filename
#                    else:
#                        if verbose:
#                            print "file %s discarded"%file
            if not recursion:
                break
        self.filenames.sort()
        
    def __iter__(self):
        """Unpickle and return the files."""
        for filename in self.filenames:
            if self.verbose:
                print "load " + filename
            yield unpickle_from_disk(filename)


def pickle_array(x, base_dir="", base_filename="array", chunk_size=5000, block_size=1, continuous=False, overwrite=False, verbose=True):
    """Stores a possibly large array x in disk, dividing it in blocks of chunk_size rows * block_size    
    """
    if overwrite == False:
        print "Error, overwriting in pickle_array is always enabled"

    array_iter = chunk_iterator(x, chunk_size=chunk_size, block_size=block_size, continuous=continuous, verbose=verbose)
    save_iterable2(array_iter, path=base_dir, basefilename=base_filename)
    del array_iter
    
    
    
def unpickle_array(base_dir="", base_filename="array", recursion=False, verbose=False, pickle_ext=PICKLE_EXT):
    """Loads a possibly large array x from disk, merging all the rows    
    """
    arr_iter = UnpickleLoader2(path=base_dir, basefilename=base_filename, verbose=True)
    arr = from_iter_to_array(arr_iter, continuous=False, block_size=1, verbose=verbose)
    del arr_iter
    return arr  


class Cache(object):
    """ Cache object for storing/retrieving objects and in particular large arrays
    """
    def __init__(self, base_dir="", base_filename="object"):
        """ Set default base directory where cache files are located, and a default base filename 
        """
        self.base_dir = base_dir
        self.base_filename = base_filename
                
        if base_dir != "":
            try:
                os.makedirs(base_dir)
            except:
                pass
            
    def update_cache(self, obj, obj_data=None, base_dir = None, base_filename=None, overwrite=True, use_hash=None, verbose=True):
        """Stores the object obj_data (or obj) in cache 
        
        It requires a base_filename without extension, and in case of arrays without sequence number
        """
        if use_hash != None:
            print "Warning, hashing was disabled by calling function in update_cache!!!"
            hashval = use_hash
        else:
            hashval = hash_object(obj).hexdigest()
        
        if base_dir == None:
            base_dir = self.base_dir
        
        if base_filename == None:
            base_filename = self.base_filename

        if obj_data == None:
            obj_data = obj

        #base_filename += "_" + hashval

        if isinstance(obj_data, numpy.ndarray):
            #Warning! later make this more flexible, think about paralellization
            pickle_array(obj_data, base_dir=base_dir, base_filename=base_filename, chunk_size=5000, block_size=1, continuous=False, overwrite = overwrite, verbose=verbose)
        else:
            pickle_to_disk(obj_data, os.path.join(base_dir, base_filename + PICKLE_EXT), overwrite=overwrite)
        return hashval
    
    def is_splitted_file_in_filesystem(self, base_dir=None, base_filename=None, recursion=None):
        if base_dir == None:
            base_dir = self.base_dir     

        complete_path = os.path.join(base_dir, base_filename)
        base_dir = os.path.dirname(complete_path)
        base_filename = os.path.basename(complete_path)
        
        print "Looking for a splitted file in directory: %s with base_filename=%s"%(base_dir, base_filename)
        verbose = True
        file_found = False
        for root, _, files in os.walk(base_dir):
            for filename in files:
                if os.path.splitext(filename)[1] == PICKLE_EXT:
                    txt = os.path.splitext(filename)[0]
                    if txt[0:len(base_filename)] == base_filename: 
                        file_found = True
                        if verbose:
                            print "file %s found"%filename
#                    else:
#                        if verbose:
#                            print "file %s discarded"%file
            if not recursion:
                break
        return file_found
    
    def is_file_in_filesystem(self, base_dir=None, base_filename=None):
        if base_dir == None:
            base_dir = self.base_dir     

        filename = os.path.join(base_dir, base_filename + PICKLE_EXT)
        print "Looking for file:", filename
        return os.path.lexists(filename)

    def load_obj_from_cache(self, hash_value=None, base_dir = None, base_filename=None, verbose=True):
        if base_dir == None:
            base_dir = self.base_dir
        
        if base_filename == None:
            base_filename = self.base_filename
        
        if hash_value != None:
            base_filename += "_" + hash_value
        filename = os.path.join(base_dir, base_filename + PICKLE_EXT)
        obj = unpickle_from_disk(filename)
        return obj

    def load_array_from_cache(self, hash_value=None, base_dir = None, base_filename=None, verbose=True):
        if base_dir == None:
            base_dir = self.base_dir        
        if base_filename == None:
            base_filename = self.base_filename
        if hash_value != None:
            base_filename = base_filename + "_" + hash_value

        complete_path = os.path.join(base_dir, base_filename)
        base_dir = os.path.dirname(complete_path)
        base_filename = os.path.basename(complete_path)
        
        print "Trying to unpickle, with basedir=", base_dir, " and base_filename=", base_filename
        x = unpickle_array(base_dir=base_dir, base_filename=base_filename, verbose=verbose)
#                 unpickle_array(base_dir="", base_filename="array", recursion=False, verbose=False, pickle_ext=PICKLE_EXT):
        return x
    
    


#WARNING, local copy of same methods in eban_SFA_libs, these are current
#Turbo fast, speed optimization, hashes only depend on a few entries
def hash_array(x, m=None, turbo_fast=True): 
    #Warning, turbo fast hashing is not reliable... it ignores too much data!
    #Warning, function str gives all elements sometimes!!!
    if m == None:
        m = hashlib.md5() # or sha1 etc
            
    if turbo_fast:
        if x.ndim == 1:
            m.update(str(x)+str(x.sum())) #+str(x.shape())
        else:
#Warning. Condensed form should be faster, per component form uses less memory?
#            m.update(str(x[0])+str(x[-1])+str(x.sum())+str(x.diagonal())) #+str(x.shape())
            m.digest()
            m.update(str(x[0]))
            m.digest()
            m.update(str(x[-1]))
            m.digest()
            m.update(str(x.sum()))
            m.digest()            
            m.update(str(x.diagonal())) #+str(x.shape())
        return m
    else:
        y = x.flatten()
    
        for value in y: # array contains the data
            m.update(str(value))
        return m

def get_data_vars(obj):
    return [var for var in dir(obj) if (not callable(getattr(obj, var))) or str(getattr(obj, var).__class__)[-6:-2] == "Node"]

def get_all_vars(obj):
    return [var for var in dir(obj)]

def remove_hidden_vars(variable_list):
    clean_list = []
    for var in variable_list:
        if var[0] != "_":
            clean_list.append(var)
    return clean_list

#Warning, setting recursion to False causes the hash of [x] and x to be different
#Recursion should usually be true, but avoid circular references!!!!
def hash_object(obj, m=None, recursion=True, verbose=False): 
    #verbose=False or True
    if verbose:
        print "hashing_object called with obj: ", obj, "of type: ", type(obj)
    #print "verbose=",verbose
    #quit()

    if m == None:
        m = hashlib.md5() # or sha1 etc

    
    #Case -3: object is a numpy scalar
    if isinstance(obj, (numpy.float, numpy.float64, numpy.int)):
        rep = str(obj)
        if verbose:
            print "Hashing as numpy_scalar with representation:", rep
        m.update(rep)
        return m

    #Case -2: object is a "type" object
    skip_types = True
    if isinstance(obj, type):
        if skip_types:
            return m
        rep = str(obj)
        if verbose:
            print "Hashing as a type:", rep
        m.update(rep)
        return m

    #Case -1: object is a numpy.dtype object, or other recursive types
    skip_dtypes = True
    if isinstance(obj, numpy.dtype):
        if skip_dtypes:
            return m
        rep = str(obj)
        if verbose:
            print "Hashing as single (recursive) scalar:", rep
        m.update(rep)
        return m
    
    #Case 0: object is enumerable.. Is the second condition necessary????
    #TODO: Check if condition is necessary, changed 2 to 1
    if isinstance(obj, (tuple, list)) and len(obj) >= 1:
        if verbose:
            print "Hashing each list element"
        for var in obj:
            m = hash_object(var, m, recursion, verbose=verbose)
        return m

    #Case 0.5: object is a dictionary...
    if isinstance(obj, dict):
        if verbose:
            print "Hashing each dictionary element and key:",  obj
#WARNING!!! REMOVING THIS LINES; HOPE THE CODE WORKS AND DICTIONARIES ARE HASHED
#        if len(object) not in [0,5]:
#            print "Ignoring length %d dictionary"%len(object)
#            return m
        for var in obj.keys():
            m.update(str(var)) # hash_object(var, m, recursion, verbose=verbose)
            m = hash_object(obj[var], m, recursion, verbose=verbose)
        return m

    #Case 1: object is an array
    if isinstance(obj, numpy.ndarray):
        if verbose:
            print "Hashing as an array:", obj
        m = hash_array(obj, m) 
        return m
    
    #Case 2: object is a string
    if isinstance(obj, str):
        if verbose:
            print "Hashing as a string:", obj
        m.update(obj) 
        return m
    

    #Case 3: object is a function
    if isinstance(obj, (types.FunctionType, types.LambdaType, types.BuiltinFunctionType, types.BuiltinMethodType)):
        if verbose:
            print "Hashing functionTypes by name", obj
        m.update(str(obj.__name__))
        return m
   
    #Case 4: object is None
    if isinstance(obj, (types.NoneType)):
        if verbose:
            print "Skipping hash of NoneType", obj
        #m.update(str(obj.__name__))
        return m

 
    #Unknown type, probably an object 
    #hash object type, and then its contents
    rep = str(type(obj))
    m.update(rep)

    dataList = get_data_vars(obj) #get_data_vars(object) #Warning... why only data vars instead of all vars???
    #Add funcList and hash contents as text (non-recursively)!!
    dataList = remove_hidden_vars(dataList)

    #Case 2: object is an scalar value, or an "empty" object
    skip_empty_object=True
    if len(dataList) == 0:
        if skip_empty_object:
            return m
        rep = str(obj)
        if verbose:
            print "Hashing as single scalar or empty object:", rep
        m.update(rep)
        return m

    #Case 3: object has data attributes
    if verbose:
        print "Hashing as an object, contains data parameters:", dataList

    for var in dataList:
        if recursion == False:
            rep = str(getattr(obj, var))
            if verbose:
                print "Non-Recursive (Incomplete) Hashing:", rep
            m.update(rep)
        else:
            if verbose:
                print "Recursive Hashing as object:",
#            if len(remove_hidden_vars(get_data_vars(object))) > 0:
            m = hash_object(getattr(obj, var), m, recursion, verbose=verbose)
#            else:
#                m.update(str(getattr(object, variable)))
    return m

#continous indicates whether the data was saved in continous mode or not
#if it continuous is True, an element is skipped to avoid repetitions
#usually continuos should be False, unless you are doing a hack
def from_iter_to_array(iterator, continuous, block_size=1, verbose=0):
    print "Creating an array from an iterator"
    result = None
    for data in iterator:
        if result is None:
            result = data   
        else:
            if continuous is True:
                result = numpy.append(result[block_size:], data, axis=0)
            else:
                result = numpy.append(result, data, axis=0)                
    return result
          
#Warning, copied from SFA_libs, should not be here
#allows usage of a large array as an iterator
#chunk_size is given in number of blocks
#continuos=False:just split, True:
class chunk_iterator(object):
    def __init__(self, x, chunk_size, block_size=None, continuous=False, verbose=False):
        if block_size is None:
            block_size=1
        self.block_size = block_size
        self.x = x
        if chunk_size < 2:
            chunk_size = 2
        self.chunk_size = chunk_size
        self.index = 0
        self.len_input = x.shape[0]
        self.continuous = continuous
        self.verbose = verbose
 
        if self.verbose:
            print "Creating chunk_iterator with: chunk_size=%d,block_size=%d, continuous=%d, len_input=%d"%(chunk_size,block_size, continuous, self.len_input) 
        if continuous is False:
            if x.shape[0]% block_size != 0:
                er = Exception("Incorrect block_size %d should be a divisor of %d"%(block_size, x.shape[0]))
                raise Exception(er)
            if x.shape[0]%(chunk_size * block_size)!= 0:
                er = "Warning! Chunks of desired size %d using blocks of size %d over signal %d do not have the same size!"%(chunk_size, block_size, x.shape[0])
                print er            
        else:
            if x.shape[0]%block_size != 0 or (x.shape[0]/block_size-1)%(chunk_size-1) != 0:
                print "WARNING!!!!!!!!!!!!! Last Chunk won't have the same size (%d chunk_size, %d x.shape[0])"%(chunk_size, x.shape[0])        
    def __iter__(self):
        self.index = 0
        return self
    def next(self):
        if self.verbose:
            print "geting next chunk, for i=", self.index
        if self.index >= self.len_input:
            raise StopIteration
#        try:
        ret = self.x[self.index:self.index + self.chunk_size * self.block_size, :]
        if self.continuous is False:
            self.index = self.index + self.chunk_size * self.block_size
        else:
            self.index = self.index + (self.chunk_size - 1) * self.block_size            
#        except:
#            print "oooppssssss!!!"
#            raise StopIteration 
        return ret


def find_filenames_beginning_with(base_dir, prefix, recursion=False, extension=".pckl"):
    filenames = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if os.path.splitext(file)[1] == extension:
                txt = os.path.splitext(file)[0]
                if txt[0:len(prefix)] == prefix: 
                    filenames.append(os.path.join(root, file))
        if not recursion:
            break
    filenames.sort()
    return filenames

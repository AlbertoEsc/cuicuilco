#Specialized functions for computing node inverses and localized inverses
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 10 Dec 2009
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott


import numpy
import scipy
import scipy.optimize
import mdp
import more_nodes
#import patch_mdp
import sfa_libs


def localized_inverse(self, x_local, y_to_invert, verbose=False):
    """Computes the localized inverse of a flow, as the composition of the
    localized inverses of each node
    """
    flow = self.flow
    num_nodes = len(flow)

    #compute all forward signals
    forward_x = [x_local, ]
    current_x = x_local
    for i in range(num_nodes):
        next_x = flow[i].execute(current_x)
        forward_x.append(next_x)
        current_x = next_x
    
    #then compute each backwards signal
    current_y =  y_to_invert
    for i in range(len(flow)-1, -1, -1):
        try:
            node = flow[i]
            #Localized inverse of linear nodes
            linear_nodes = (mdp.nodes.SFANode, mdp.nodes.PCANode, mdp.nodes.WhiteningNode)
            processed = False
            
            for node_class in linear_nodes:
                if isinstance(node, node_class) and processed == False:
                    processed = True
                    current_y = node.localized_inverse(forward_x[i], forward_x[i+1], current_y)

            #"Localized" inverse of invertible nodes
            invertible_nodes = [more_nodes.PInvSwitchboard, more_nodes.PointwiseFunctionNode, more_nodes.RandomPermutationNode]
            for node_class in invertible_nodes:
                if isinstance(node, node_class) and processed == False:
                    processed = True
                    current_y = node.inverse(current_y)

            #Localized inverse of Layers (recursion)
            if isinstance(node, mdp.hinet.Layer) and processed == False:
                processed = True
                current_y = node.localized_inverse(forward_x[i], forward_x[i+1], current_y)

            #Localized inverse of GeneralExpansionNode        
            if isinstance(node, more_nodes.GeneralExpansionNode) and processed == False:
                processed = True
                current_y = node.localized_inverse(forward_x[i], forward_x[i+1],current_y)

            if processed == False:
                txt = "I was not able to compute the localized inverse of node: %s ", str(node)
                raise Exception(txt)
        except Exception, e:
            self._propagate_exception(e, i)
        if verbose == True:
            print "Localized Node Inversion finished"

    print "Localized Flow Inversion finished"        
    return current_y


def linear_localized_inverse(self, x, y, y_prime):
    """Computes a localized inverse around the point x.
    This assumes that the node is linear and keeps the part of x that is lost
    when mapping to y and then back to x' by using the pseudoinverse
    """
    #print "LinLocInv: x.shape=", x.shape, "y.shape=", y.shape, "y_prime.shape=", y_prime.shape
    x_perp = x - self.inverse(y)
    return x_perp + self.inverse(y_prime) 


def layer_localized_inverse(self, x, y, y_prime, verbose=False):
    """Computes a localized inverse around the point x for a given layer
    as the concatenation of the localized inverses of each node contained in the layer
    """
    #slightly modified version of mdp.hinet.Layer.inverse
    in_start = 0
    in_stop = 0
    out_start = 0
    out_stop = 0
    x_prime = None

    nodes_with_localized_inverse = (mdp.nodes.SFANode, mdp.nodes.PCANode, mdp.nodes.WhiteningNode)

    for node in self.nodes:
        #print "Layer localized inversion iteration"
        # compared with execute, input and output are switched
        out_start = out_stop
        out_stop += node.input_dim
        in_start = in_stop
        in_stop += node.output_dim
       
        #print "out_start = %d, out_stop = %d, in_start = %d, in_stop = %d" %(out_start, out_stop, in_start, in_stop)
        if x_prime is None:
            if isinstance(node, nodes_with_localized_inverse):
                node_x = node.localized_inverse(x[:,out_start:out_stop], y[:,in_start:in_stop], y_prime[:,in_start:in_stop])
            else:
                if verbose:
                    print "Using inverse for node:", node
                node_x = node.inverse(y_prime[:,in_start:in_stop])
                
            x_prime = numpy.zeros([node_x.shape[0], self.input_dim], dtype=node_x.dtype)
            x_prime[:,out_start:out_stop] = node_x
        else:
            if isinstance(node, nodes_with_localized_inverse):
                x_prime[:,out_start:out_stop] = node.localized_inverse(x[:,out_start:out_stop], y[:,in_start:in_stop], y_prime[:,in_start:in_stop])
            else:
                x_prime[:,out_start:out_stop] = node.inverse(y_prime[:,in_start:in_stop])
    return x_prime


def general_expansion_node_localized_inverse(self, x, y, y_prime, max_steady_factor=None, \
                 delta_factor=None, min_delta=None):
    """The localized inverse in a general_expansion_node is just the inverse using x as hint (initial point) """
    return self.inverse(y_prime, use_hint=x, max_steady_factor=max_steady_factor, delta_factor=delta_factor, min_delta=min_delta)



#input: exp_x_noisy.shape=[1,dim_exp_x], 
#outputs: one dim vectors  
def invert_exp_funcs(exp_x_noisy, dim_x, exp_funcs, distance=sfa_libs.distance_best_squared_Euclidean, use_hint=False, max_steady_factor=5, delta_factor=0.7, min_delta=0.0001, verbose=False):
    """This function is deprecated, don't use it """
    num_samples = exp_x_noisy.shape[0]

    if use_hint:
        if verbose == True:
            print "Using lowest dim_x=%d elements of input for first approximation!"%(dim_x)
        app_x = exp_x_noisy[:,0:dim_x].copy()
    else:
        app_x = numpy.random.normal(size=(num_samples,dim_x))
    
    app_exp_x =  sfa_libs.apply_funcs_to_signal(exp_funcs, app_x)
    if verbose == True:
        print "app_exp_x", app_exp_x

    dim_exp_x = exp_x_noisy.shape[1]

    iterations = 0
    max_steady_iter = numpy.sqrt(dim_x) * max_steady_factor
    for row in range(num_samples):
        app_x_row = app_x[row].reshape(1, dim_x)
        exp_x_noisy_row = exp_x_noisy[row].reshape(1, dim_exp_x)
        app_exp_x_row = app_exp_x[row].reshape(1, dim_exp_x)
        if verbose == True:
            print "distance is: ", distance
        dist = distance(exp_x_noisy_row, app_exp_x_row)
        delta = 1.0
        while delta > min_delta:
#        print "Delta Value=", delta
            steady_iter = 0
            while steady_iter < max_steady_iter:
                iterations = iterations + 1
                i = numpy.random.randint(0, high=dim_x)
                app_x_tmp_row = app_x_row.copy()
#        print "i=", i, 
                app_x_tmp_row[0,i] = app_x_tmp_row[0,i] + numpy.random.normal(scale=delta)
                app_exp_x_tmp_row =  sfa_libs.apply_funcs_to_signal(exp_funcs, app_x_tmp_row)
                dist_tmp = distance(exp_x_noisy_row, app_exp_x_tmp_row) 
                if dist_tmp < dist:
# WARNING, was copying necessary??????
#                    app_x_row = app_x_tmp_row.copy()
#                    app_exp_x_row = app_exp_x_tmp_row.copy()
                    app_x_row = app_x_tmp_row
                    app_exp_x_row = app_exp_x_tmp_row
                    dist=dist_tmp
                    if verbose:
                        print "app_x_row =", app_x_row
                        print ", Dist=", dist
                    steady_iter = 0
                else:
                    steady_iter = steady_iter + 1    
            delta = delta * delta_factor
        app_x[row] = app_x_row[0]
        app_exp_x[row] = app_exp_x_row[0]
        if verbose:
            print "GEXP Inv: ", iterations, " iterations. ",
    return app_x, app_exp_x


#*******************************************8
#y_noisy.shape is (M), exp_funcs
#only one dimensional vector inputs
#favour solutions close to x_orig
#adding a term  [ k(x-x_orig) ] ** 2 to the minimization problem
def residuals(app_x, y_noisy, exp_funcs, x_orig, k):
    """ Computes error signals as the concatenation of the reconstruction error 
    (y_noisy - exp_funcs(app_x)) and the distance from the original (x_orig - app_x)
    using a weighting factor k
    """
    app_x = app_x.reshape((1,len(app_x)))
    app_exp_x =  sfa_libs.apply_funcs_to_signal(exp_funcs, app_x)
   
    div_y = numpy.sqrt(len(y_noisy))
    div_x = numpy.sqrt(len(x_orig))
    return numpy.append( (1-k)*(y_noisy-app_exp_x[0]) / div_y, k * (x_orig - app_x[0])/div_x )

#input: exp_x_noisy.shape=[1,dim_exp_x], 
#outputs: one dim vectors  
#Improved version
def invert_exp_funcs2(exp_x_noisy, dim_x, exp_funcs, distance=sfa_libs.distance_best_squared_Euclidean, use_hint=False, max_steady_factor=5, delta_factor=0.7, min_delta=0.0001, k=0.5, verbose=False):
    """ Function that approximates a preimage of exp_x_noisy notice 
    that distance, max_steady_factor, delta, min_delta are deprecated and useless
    """
    
    num_samples = exp_x_noisy.shape[0]

    if isinstance(use_hint, numpy.ndarray):
        if verbose == True:
            print "Using suggested approximation!"
        app_x = use_hint.copy()
    elif use_hint == True:
        if verbose == True:
            print "Using lowest dim_x=%d elements of input for first approximation!"%(dim_x)
        app_x = exp_x_noisy[:,0:dim_x].copy()
    else:
        app_x = numpy.random.normal(size=(num_samples,dim_x))
    
    
    for row in range(num_samples):
#        app_x_row = app_x[row].reshape(1, dim_x)
#        exp_x_noisy_row = exp_x_noisy[row].reshape(1, dim_exp_x)
#        app_exp_x_row = app_exp_x[row].reshape(1, dim_exp_x)
#Definition:       scipy.optimize.leastsq(func, x0, args=(), Dfun=None, full_output=0, col_deriv=0, 
#                                         ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=0, epsfcn=0.0, 
#                                         factor=100, diag=None, warning=True)
        plsq = scipy.optimize.leastsq(residuals, app_x[row], args=(exp_x_noisy[row], exp_funcs, app_x[row], k), ftol=1.49012e-06, xtol=1.49012e-06, gtol=0.0, maxfev=50*dim_x, epsfcn=0.0, factor=1.0)
        app_x[row] = plsq[0]

    app_exp_x = sfa_libs.apply_funcs_to_signal(exp_funcs, app_x)
    return app_x, app_exp_x




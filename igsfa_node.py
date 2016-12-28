import numpy
import scipy
import scipy.optimize
import mdp
from mdp.utils import pinv

import copy
import sys
from sfa_libs import select_rows_from_matrix, distance_squared_Euclidean
from inversion import invert_exp_funcs2
import more_nodes
from more_nodes import GeneralExpansionNode
from gsfa_node import GSFANode

class iGSFANode(mdp.Node):
    """This node implements "information-preserving graph-based SFA (iGSFA)", which is the main component of
    hierarchical iGSFA (HiGSFA). For further information, see:
    A. N. Escalante-B. and L. Wiskott. Improved graph-based SFA: Information preservation complements the slowness principle.
    e-print arXiv:1601.03945, 1 2016a    
    """    
    def __init__(self, input_dim = None, output_dim=None, pre_expansion_node_class = None, expansion_funcs=None, expansion_output_dim=None, expansion_starting_point=None, max_lenght_slow_part = None, max_num_samples_for_ev = None, max_test_samples_for_ev=None, offsetting_mode = "all features", max_preserved_sfa=1.9999, reconstruct_with_sfa = True,  out_sfa_filter=False, **argv ):
        super(iGSFANode, self).__init__(input_dim =input_dim, output_dim=output_dim, **argv)
        self.pre_expansion_node_class = pre_expansion_node_class #Type of node used to expand the data
        self.pre_expansion_node = None #Node that expands the data
        self.expansion_output_dim = expansion_output_dim #Expanded dimensionality
        self.expansion_starting_point = expansion_starting_point #Initial parameters of the expansion function

        #creates an expansion node
        if expansion_funcs != None:
            #print "creating node with expansion dim = ", self.expansion_output_dim,
            self.exp_node = GeneralExpansionNode(funcs=expansion_funcs, output_dim = self.expansion_output_dim, starting_point=self.expansion_starting_point)
        else:
            self.exp_node = None

        self.sfa_node = None
        self.max_lenght_slow_part = max_lenght_slow_part
        
        ###self.max_num_samples_for_ev = max_num_samples_for_ev
        ###self.max_test_samples_for_ev = max_test_samples_for_ev

        self.feature_scaling_factor = 0.5  #Factor that prevents the amplitude of the features from growing too much through the layers of the network
        self.exponent_variance = 0.5
        self.max_preserved_sfa=max_preserved_sfa
        self.reconstruct_with_sfa = reconstruct_with_sfa #Indicates whether (nonlinear) SFA components are used for reconstruction
        ###self.out_sfa_filter = out_sfa_filter
        self.compress_input_with_pca = False #True
        self.compression_out_dim = 0.99 #0.99 #0.95 #98
        self.offsetting_mode = offsetting_mode
        
    def is_trainable(self):
        return True

    def _train(self, x, block_size=None, train_mode = None, node_weights=None, edge_weights=None, scheduler = None, n_parallel=None, **argv):
        self.input_dim = x.shape[1]
       
        if self.output_dim == None:
            self.output_dim = self.input_dim


        print "Training iGSFANode..."

        #Remove mean before expansion
        self.x_mean = x.mean(axis=0) 
        x_zm=x-self.x_mean
        
        #Reorder or pre-process the data before it is expanded, but only if there is really an expansion
        if self.pre_expansion_node_class != None and self.exp_node != None:
            self.pre_expansion_node = self.pre_expansion_node_class() #GSFANode() or a WhitheningNode()
            self.pre_expansion_node.train(x_zm, block_size=block_size, train_mode = train_mode) #Some arguments might not be necessary
            self.pre_expansion_node.stop_training()
            x_pre_exp = self.pre_expansion_node.execute(x_zm)
        else:
            x_pre_exp = x_zm

        #Expand data
        if self.exp_node != None: 
            print "expanding x..."
            exp_x = self.exp_node.execute(x_pre_exp) #x_zm
        else:
            exp_x = x_pre_exp

        self.expanded_dim = exp_x.shape[1]

        if self.max_lenght_slow_part == None:
            sfa_output_dim = min(self.expanded_dim, self.output_dim)
        else:
            sfa_output_dim = min(self.max_lenght_slow_part, self.expanded_dim, self.output_dim)
                    
        #Apply SFA to expanded data
        self.sfa_node = GSFANode(output_dim=sfa_output_dim)
        self.sfa_node.train_params(exp_x, params={"block_size":block_size, "train_mode":train_mode, "node_weights":node_weights, "edge_weights":edge_weights})#, node_weights=None, edge_weights=None, scheduler = None, n_parallel=None)
        self.sfa_node.stop_training()
        print "self.sfa_node.d", self.sfa_node.d
            
        #Decide how many slow features are preserved (either use Delta_T=max_preserved_sfa when 
        #max_preserved_sfa is a float, or preserve max_preserved_sfa features when max_preserved_sfa is an integer)
        if isinstance(self.max_preserved_sfa, float):
            self.num_sfa_features_preserved = (self.sfa_node.d <= self.max_preserved_sfa).sum()
        elif isinstance(self.max_preserved_sfa, int):
            self.num_sfa_features_preserved = self.max_preserved_sfa
        else:
            ex = "Cannot handle type of self.max_preserved_sfa"
            print ex
            raise Exception(ex)

        if self.num_sfa_features_preserved > self.output_dim:
            self.num_sfa_features_preserved = self.output_dim
        
        SFANode_reduce_output_dim(self.sfa_node, self.num_sfa_features_preserved)
        print "sfa execute..."
        sfa_x = self.sfa_node.execute(exp_x)
          
        #Truncate leaving only slowest features (this might be redundant)
        sfa_x = sfa_x[:,0:self.num_sfa_features_preserved]
        
        #normalize sfa_x                    
        self.sfa_x_mean = sfa_x.mean(axis=0)
        self.sfa_x_std = sfa_x.std(axis=0)            
        print "self.sfa_x_mean=", self.sfa_x_mean
        print "self.sfa_x_std=", self.sfa_x_std
        if (self.sfa_x_std==0).any():
            er = "zero-component detected"
            raise Exception(er)
        n_sfa_x = (sfa_x - self.sfa_x_mean) / self.sfa_x_std 

        if self.reconstruct_with_sfa:      
            #Compress input
            if self.compress_input_with_pca:
                self.compress_node = mdp.nodes.PCANode(output_dim=self.compression_out_dim)
                self.compress_node.train(x_zm)
                x_pca = self.compress_node.execute(x_zm)
                print "compress: %d components out of %d sufficed for the desired compression_out_dim"%(x_pca.shape[1], x_zm.shape[1]), self.compression_out_dim
            else:
                x_pca = x_zm
    
            #approximate input linearly, done inline to preserve node for future use
            print "training linear regression..."
            self.lr_node = mdp.nodes.LinearRegressionNode()
            self.lr_node.train(n_sfa_x, x_pca) #Notice that the input "x"=n_sfa_x and the output to learn is "y" = x_pca
            self.lr_node.stop_training()  
            x_pca_app = self.lr_node.execute(n_sfa_x)
    
            if self.compress_input_with_pca:
                x_app = self.compress_node.inverse(x_pca_app)
            else:
                x_app = x_pca_app
        else:
            x_app = numpy.zeros_like(x_zm)
            
        #Remove linear approximation
        sfa_removed_x = x_zm - x_app
        
        #print "Data_variance(x_zm)=", data_variance(x_zm)
        #print "Data_variance(x_app)=", data_variance(x_app)
        #print "Data_variance(sfa_removed_x)=", data_variance(sfa_removed_x)
        #print "x_app.mean(axis=0)=", x_app
        #TODO:Compute variance removed by linear approximation
        print "ranking method..."
        if self.reconstruct_with_sfa and self.offsetting_mode == "QR_decomposition": #AKA Laurenz method for feature scaling( +rotation)
            M = self.lr_node.beta[1:,:].T # bias is used by default, we do not need to consider it
            Q,R = numpy.linalg.qr(M)
            self.Q = Q
            self.R = R
            self.Rpinv = pinv(R)
            s_n_sfa_x = numpy.dot(n_sfa_x, R.T)
        elif self.reconstruct_with_sfa and self.offsetting_mode == "sensitivity_based_pure": #AKA my method for feature scaling (no rotation)
            beta = self.lr_node.beta[1:,:] # bias is used by default, we do not need to consider it
            sens = (beta**2).sum(axis=1)
            self.magn_n_sfa_x = sens
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x ** self.exponent_variance
            print "method: sensitivity_based_pure enforced"       
        elif self.reconstruct_with_sfa and self.offsetting_mode == "sensitivity_based_normalized": #AKA alternative method for feature scaling (no rotation)
            beta = self.lr_node.beta[1:,:] # bias is used by default, we do not need to consider it
            sens = (beta**2).sum(axis=1)
            self.magn_n_sfa_x = sens * ((x_pca_app**2).sum(axis=1).mean() / sens.sum())
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x ** self.exponent_variance
            print "method: sensitivity_based_normalized enforced" 
        elif self.offsetting_mode == None:
            self.magn_n_sfa_x = 1.0
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x
            print "method: constant amplitude for all slow features"
        elif self.offsetting_mode == "data_dependent":
            self.magn_n_sfa_x = 0.01 * numpy.min(x_zm.var(axis=0)) # SFA components have a variance 1/10000 times the smallest data variance
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x ** self.exponent_variance #Scale according to ranking
            print "method: data dependent"
        else:
            er = "unknown feature scaling method"
            raise Exception(er)

        print "training PCA..."
        self.pca_node = mdp.nodes.PCANode(reduce=True) #output_dim = pca_out_dim)
        self.pca_node.train(sfa_removed_x)
        self.pca_node.stop_training()

        #TODO:check that pca_out_dim > 0
        print "executing PCA..."
 
        pca_x = self.pca_node.execute(sfa_removed_x)
        
        if self.pca_node.output_dim + self.num_sfa_features_preserved < self.output_dim:
            er = "Error, the number of features computed is SMALLER than the output dimensionality of the node: " + \
            "self.pca_node.output_dim=", self.pca_node.output_dim, "self.num_sfa_features_preserved=", self.num_sfa_features_preserved, "self.output_dim=", self.output_dim
            raise Exception(er)

        #Finally output is the concatenation of scaled slow features and remaining pca components
        sfa_pca_x = numpy.concatenate((s_n_sfa_x, pca_x), axis=1)
        
        sfa_pca_x_truncated =  sfa_pca_x[:,0:self.output_dim]
        
        #Compute explained variance from amplitudes of output compared to amplitudes of input
        #Only works because amplitudes of SFA are scaled to be equal to explained variance, and because PCA is a rotation
        #And because data has zero mean
        self.evar =  (sfa_pca_x_truncated**2).sum() / (x_zm**2).sum()
        print "Variance(output) / Variance(input) is ", self.evar
        self.stop_training()

    def _is_invertible(self):
        return True

    def _execute(self, x):
        ###num_samples = x.shape[0]

        x_zm = x - self.x_mean

        if self.pre_expansion_node != None:
            x_pre_exp = self.pre_expansion_node.execute(x_zm)
        else:
            x_pre_exp = x_zm

        if self.exp_node != None: 
            exp_x = self.exp_node.execute(x_pre_exp)
        else:
            exp_x = x_pre_exp

        sfa_x = self.sfa_node.execute(exp_x)
        sfa_x = sfa_x[:,0:self.num_sfa_features_preserved]
        
        n_sfa_x = (sfa_x - self.sfa_x_mean) / self.sfa_x_std
            
#        if self.compress_input_with_pca:
#            x_pca = self.compress_node.execute(x_zm)
#        else:
#            x_pca = x_zm
        if self.reconstruct_with_sfa:
            #approximate input linearly, done inline to preserve node
            x_pca_app = self.lr_node.execute(n_sfa_x)
    
            if self.compress_input_with_pca:
                x_app = self.compress_node.inverse(x_pca_app)
            else:
                x_app = x_pca_app
        else:
            x_app = numpy.zeros_like(x_zm)

        #Remove linear approximation
        sfa_removed_x = x_zm - x_app
        
        if self.reconstruct_with_sfa and self.offsetting_mode == "QR_decomposition": #AKA Laurenz method for feature scaling( +rotation)
            s_n_sfa_x = numpy.dot(n_sfa_x, self.R.T)
        elif self.reconstruct_with_sfa and self.offsetting_mode == "sensitivity_based_pure": #AKA my method for feature scaling (no rotation)
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x ** self.exponent_variance
        elif self.reconstruct_with_sfa and self.offsetting_mode == "sensitivity_based_normalized": #AKA alternative method for feature scaling (no rotation)
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x ** self.exponent_variance
        elif self.offsetting_mode == None:
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x ** self.exponent_variance
        elif self.offsetting_mode == "data_dependent":
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x ** self.exponent_variance #Scale according to ranking
        else:
            er = "unknown feature scaling method"
            raise Exception(er)
        
        #Apply PCA to sfa removed data             
        pca_x = self.pca_node.execute(sfa_removed_x)
        
        #Finally output is the concatenation of scaled slow features and remaining pca components
        sfa_pca_x = numpy.concatenate((s_n_sfa_x, pca_x), axis=1)
        
        sfa_pca_x_truncated =  sfa_pca_x[:,0:self.output_dim]

        return sfa_pca_x_truncated

#        verbose=False
#        if verbose:
#            print "x[0]=",x_orig[0]
#            print "x_zm[0]=", x[0]
#            print "exp_x[0]=", exp_x[0]
#            print "s_x_1[0]=", s_x_1[0]
#            print "sfa_removed_x[0]=", sfa_removed_x[0]
#            print "proj_sfa_x[0]=", proj_sfa_x[0]
#            print "pca_x[0]=", pca_x[0]
#            print "n_pca_x[0]=", n_pca_x[0]        
#            print "sfa_x[0]=", sfa_x[0] + self.sfa_x_mean
#            print "s_x_2_truncated[0]=", s_x_2_truncated[0]
#            print "sfa_filtered[0]=", sfa_filtered[0]

    def _inverse(self, y, linear_inverse=True):
        if linear_inverse:
            return self.linear_inverse(y)
        else:
            return self.non_linear_inverse(y)

    def non_linear_inverse(self, y, verbose=False):
        x_lin = self.linear_inverse(y)
        rmse_lin = ((y - self.execute(x_lin))**2).sum(axis=1).mean()**0.5 
#        scipy.optimize.leastsq(func, x0, args=(), Dfun=None, full_output=0, col_deriv=0, ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=0, epsfcn=0.0, factor=100, diag=None)
        x_nl = numpy.zeros_like(x_lin)
        y_dim = y.shape[1]
        x_dim = x_lin.shape[1]
        if y_dim < x_dim:
            num_zeros_filling = x_dim - y_dim
        else:
            num_zeros_filling = 0
        if verbose:
            print "x_dim=", x_dim, "y_dim=", y_dim, "num_zeros_filling=", num_zeros_filling
        y_long = numpy.zeros(y_dim+num_zeros_filling)

        for i, y_i in enumerate(y):
            y_long[0:y_dim]= y_i
            if verbose:
                print "x_0=", x_lin[i], 
                print "y_long=", y_long            
            plsq = scipy.optimize.leastsq(func=f_residual, x0=x_lin[i], args=(self, y_long), full_output=False)
            x_nl_i = plsq[0]
            if verbose:
                print "x_nl_i=", x_nl_i, "plsq[1]=", plsq[1]
            if plsq[1] != 2:
                print "Quitting: plsq[1]=", plsq[1]
                #quit()
            x_nl[i] = x_nl_i
            print "|E_lin(%d)|="%i, ((y_i - self.execute(x_lin[i].reshape((1,-1))))**2).sum()**0.5, 
            print "|E_nl(%d)|="%i, ((y_i - self.execute(x_nl_i.reshape((1,-1))))**2).sum()**0.5
        rmse_nl = ((y - self.execute(x_nl))**2).sum(axis=1).mean()**0.5
        print "rmse_lin(all samples)=", rmse_lin, "rmse_nl(all samples)=", rmse_nl
        return x_nl

    def linear_inverse(self, y):
        num_samples = y.shape[0]
        if y.shape[1] != self.output_dim:
            er = "Serious dimensionality inconsistency:", y.shape[0], self.output_dim
            raise Exception(er)

        sfa_pca_x_full = numpy.zeros((num_samples, self.pca_node.output_dim+self.num_sfa_features_preserved)) #self.input_dim
        #print "self.output_dim=", self.output_dim, "y.shape=", y.shape, "sfa_pca_x_full.shape=", sfa_pca_x_full.shape, "self.num_sfa_features_preserved=",self.num_sfa_features_preserved 
        sfa_pca_x_full[:,0:self.output_dim] = y
        
        s_n_sfa_x = sfa_pca_x_full[:, 0:self.num_sfa_features_preserved]
        pca_x = sfa_pca_x_full[:, self.num_sfa_features_preserved:]
        
        if pca_x.shape[1]>0:
            #print "self.pca_node.input_dim=",self.pca_node.input_dim, "self.pca_node.output_dim=",self.pca_node.output_dim
            sfa_removed_x = self.pca_node.inverse(pca_x)
        else:
            sfa_removed_x = numpy.zeros((num_samples, self.input_dim))

        #print "s_n_sfa_x.shape=", s_n_sfa_x.shape, "magn_n_sfa_x.shape=", self.magn_n_sfa_x.shape       
        if self.reconstruct_with_sfa and self.offsetting_mode == "QR_decomposition": #AKA Laurenz method for feature scaling( +rotation)          
            n_sfa_x = numpy.dot(s_n_sfa_x, self.Rpinv.T)
        else:
            n_sfa_x = s_n_sfa_x / self.magn_n_sfa_x ** self.exponent_variance

        #sfa_x = n_sfa_x * self.sfa_x_std + self.sfa_x_mean 
        if self.reconstruct_with_sfa:
            x_pca_app = self.lr_node.execute(n_sfa_x)
        
            if self.compress_input_with_pca:
                x_app = self.compress_node.inverse(x_pca_app)
            else:
                x_app = x_pca_app        
        else:
            x_app = numpy.zeros_like(sfa_removed_x)

        x_zm = sfa_removed_x + x_app
        
        x = x_zm + self.x_mean

        #print "Data_variance(x_zm)=", data_variance(x_zm)
        #print "Data_variance(x_app)=", data_variance(x_app)
        #print "Data_variance(sfa_removed_x)=", data_variance(sfa_removed_x)
        #print "x_app.mean(axis=0)=", x_app

        #verbose=False
        #if verbose:
        #    print "x[0]=",x[0]
        #    print "zm_x[0]=", zm_x[0]
        #    print "exp_x[0]=", exp_x[0]
        #    print "s_x_1[0]=", s_x_1[0]
        #    print "proj_sfa_x[0]=", proj_sfa_x[0]
        #    print "sfa_removed_x[0]=", sfa_removed_x[0]
        #    print "pca_x[0]=", pca_x[0]
        #    print "n_pca_x[0]=", n_pca_x[0]        
        #    print "sfa_x[0]=", sfa_x[0]
        
        return x
    
    
def SFANode_reduce_output_dim(sfa_node, new_output_dim, verbose=False):
    """ This function modifies an already trained SFA node (or GSFA node), 
    reducing the number of preserved SFA features to new_output_dim features.
    The modification takes place in place
    """
    if verbose:
        print "Updating the output dimensionality of SFA node"
    if new_output_dim > sfa_node.output_dim:
        er = "Can only reduce output dimensionality of SFA node, not increase it"
        raise Exception(er)
    if verbose:
        print "Before: sfa_node.d.shape=", sfa_node.d.shape, " sfa_node.sf.shape=",sfa_node.sf.shape, " sfa_node._bias.shape=",sfa_node._bias.shape
    sfa_node.d = sfa_node.d[:new_output_dim]
    sfa_node.sf = sfa_node.sf[:,:new_output_dim]
    sfa_node._bias = sfa_node._bias[:new_output_dim]
    sfa_node._output_dim = new_output_dim
    if verbose:
        print "After: sfa_node.d.shape=",sfa_node.d.shape, " sfa_node.sf.shape=",sfa_node.sf.shape, " sfa_node._bias.shape=",sfa_node._bias.shape

#Computes output errors dimension by dimension for a single sample: y - node.execute(x_app)
#The library fails when dim(x_app) > dim(y), thus filling of x_app with zeros is recommended
def f_residual(x_app_i, node, y_i):
    #print "%",
    #print "f_residual: x_appi_i=", x_app_i, "node.execute=", node.execute, "y_i=", y_i
    res_long = numpy.zeros_like(y_i)
    y_i = y_i.reshape((1,-1))
    y_i_short = y_i[:,0:node.output_dim]
#    x_app_i = x_app_i[0:node.input_dim]
    res = (y_i_short - node.execute(x_app_i.reshape((1,-1)))).flatten()
    #print "res_long=", res_long, "y_i=", y_i, "res", res
    res_long[0:len(res)]=res
#    res = (y_i - node.execute(x_app_i))
    #print "returning resudial res=", res
    return res_long

import mdp
import numpy as numx
import numpy


def learn_histogram_equalizer(x, num_pivots, linear_histogram=False, ignore_two_pivots = False):
    num_samples = x.shape[0]
    y=x.copy()
    y.sort()
    if ignore_two_pivots == False:
        indices_pivots=numpy.linspace(0, num_samples-1, num_pivots).round().astype(int)
    else:
        indices_pivots=numpy.linspace(0, num_samples-1, num_pivots+2).round().astype(int)[1:-1]
    pivots = y[indices_pivots]
    print "pivots computed=", pivots
    print "indices_pivots=", indices_pivots
    if linear_histogram:
        pivot_outputs = numpy.linspace(0, 1.0, num_pivots)
    else:
        pivot_outputs = 0.5-0.5*numpy.cos(numpy.linspace(0, 1.0, num_pivots)*numpy.pi)
    diff_pivot_outputs = pivot_outputs[1:]-pivot_outputs[:-1]
    diff_pivots = pivots[1:]-pivots[:-1]
    pivot_factor = diff_pivot_outputs/diff_pivots
    return  pivots, pivot_outputs, pivot_factor


def histogram_equalizer(y, num_pivots, pivots, pivot_outputs, pivot_factor):
    indices_sorting = numpy.searchsorted(pivots, y, side="left")
    indices_sorting[indices_sorting <= 0] = 1
    indices_sorting[indices_sorting >= num_pivots-1] = num_pivots-1#2
    indices_sorting -= 1
    hy = pivot_outputs[indices_sorting]+(y-pivots[indices_sorting]) * pivot_factor[indices_sorting]
    return hy


class HistogramEqualizationNode(mdp.Node):
    def __init__(self, num_pivots = 10, a=0.0, b=1.0, input_dim = None, output_dim = None, dtype = None):
        super(HistogramEqualizationNode, self).__init__(input_dim, output_dim, dtype)
        self.num_pivots = num_pivots
        self.a = a
        self.b = b
        self.hx = []
        print "Node creation. num_pivots=", num_pivots, "hx=", self.hx 
    def _train(self, x):
        dim = x.shape[1]
        self.hx = []
        for i in range(dim):
            self.hx.append(learn_histogram_equalizer(x[:,i], self.num_pivots))
        print "Trained: self.hx[0]=", self.hx[0]
        print "x[:,0]=", x[:,0]
    def _execute(self, x):
        dim = len(self.hx)
        y = numpy.zeros_like(x)
        for i in range(dim):
            hx = self.hx[i]
            y[:,i] = histogram_equalizer(x[:,i], num_pivots=self.num_pivots, pivots=hx[0], pivot_outputs=hx[1], pivot_factor=hx[2])      
            #yy = y[:,0].copy()
            #yy.sort()
            #print "y[:,0] sorted =", yy
        return y.clip(self.a,self.b)
    def _is_trainable(self):
        return True



def nmonomials(degree, nvariables):
    """Return the number of monomials of a given degree in a given number
    of variables."""
    return int(mdp.utils.comb(nvariables+degree-1, degree))


class CosineExpansionNode(mdp.nodes.PolynomialExpansionNode):
    def _execute(self, x):
        degree = self._degree
        dim = self.input_dim
        n = x.shape[1]
        
        # preallocate memory
        dexp = numx.zeros((self.output_dim, x.shape[0]), dtype=self.dtype)
        # copy monomials of degree 1
        dexp[0:n, :] = x.T

        k = n
        prec_end = 0
        next_lens = numx.ones((dim+1, ))
        next_lens[0] = 0
        for i in range(2, degree+1):
            prec_start = prec_end
            prec_end += nmonomials(i-1, dim)
            prec = dexp[prec_start:prec_end, :]

            lens = next_lens[:-1].cumsum(axis=0)
            next_lens = numx.zeros((dim+1, ))
            for j in range(dim):
                factor = prec[lens[j]:, :]
                len_ = factor.shape[0]
                dexp[k:k+len_, :] = x[:, j] + factor
                next_lens[j+1] = len_
                k = k+len_
        return numx.cos(dexp.T*numx.pi)


def cos_exp_dD_F(x, degree, keep_identity=False):
    num_samples, n = x.shape
    dim = n
        
    output_dim =  int(mdp.utils.comb(n+degree, degree))-1       
    # preallocate memory
    dexp = numx.zeros((output_dim, num_samples))
    # copy monomials of degree 1
    dexp[0:n, :] = x.T.copy()

    k = n
    prec_end = 0
    next_lens = numx.ones((dim+1, ))
    next_lens[0] = 0
    for i in range(2, degree+1):
        prec_start = prec_end
        prec_end += nmonomials(i-1, dim)
        prec = dexp[prec_start:prec_end, :]

        lens = next_lens[:-1].cumsum(axis=0)
        next_lens = numx.zeros((dim+1, ))
        for j in range(dim):
            factor = prec[lens[j]:, :]
            len_ = factor.shape[0]
            dexp[k:k+len_, :] = x[:, j] + factor
            next_lens[j+1] = len_
            k = k+len_

    print "keep identity is:", keep_identity
    if keep_identity:
        print "*****************"
        dexp[0:dim, :] = x[:,:].T
        dexp[dim:, :] = numpy.cos(dexp[dim:,:]*numx.pi)
        out = dexp.T
    else:
        out = numx.cos(dexp.T*numx.pi)
    return out


def cos_exp_5D_F(x):
    return cos_exp_dD_F(x, degree=5, keep_identity=False)

def cos_exp_I_5D_F(x):
    return cos_exp_dD_F(x, degree=5, keep_identity=True)

def cos_exp_I_4D_F(x):
    return cos_exp_dD_F(x, degree=4, keep_identity=True)

def cos_exp_I_3D_F(x):
    return cos_exp_dD_F(x, degree=3, keep_identity=True)

def cos_exp_2D_F(x):
    return cos_exp_dD_F(x, degree=2, keep_identity=False)

def cos_exp_I_2D_F(x):
    return cos_exp_dD_F(x, degree=2, keep_identity=True)

def cos_exp_1D_F(x):
    return cos_exp_dD_F(x, degree=1, keep_identity=False)

def cos_exp_I_1D_F(x):
    return cos_exp_dD_F(x, degree=1, keep_identity=True)

def cos_exp_I_smart2D_F(x):
    dim = x.shape[1]
    if dim > 30:
        return cos_exp_dD_F(x, degree=1, keep_identity=True) # 30 and above
    elif dim > 10:
        return cos_exp_dD_F(x, degree=2, keep_identity=True) # 11 to 30
    elif dim > 7:
        return cos_exp_dD_F(x, degree=3, keep_identity=True) # 8 to 10
    elif dim > 4:
        return cos_exp_dD_F(x, degree=4, keep_identity=True) # 5 to 7
    else:
        return cos_exp_dD_F(x, degree=5, keep_identity=True) # 1 to 4


    
class NormalizeABNode(mdp.Node):
    def __init__(self, a=0.0, b=1.0, input_dim = None, output_dim = None, dtype = None):
        super(NormalizeABNode, self).__init__(input_dim, output_dim, dtype)
        self.x_min = None
        self.x_max = None
        self.a = a
        self.b = b
    def _train(self, x):
        self.x_min = x.min(axis=0)
        self.x_max = x.max(axis=0)
        if (self.x_max == self.x_min).sum():
            er = "Error, min and max are the same for some component. min="+str(self.x_min)+" max="+str(self.x_max)
            raise Exception(er)
    def _execute(self, x):
        y = (x - self.x_min )*(self.b-self.a)/(self.x_max - self.x_min) + self.a
        return y.clip(self.a,self.b)
    def _is_trainable(self):
        return True


class NLIPCANode(mdp.Node):
    def __init__(self, exp_func = None, norm_class = None, feats_at_once = 1, factor_projection_out=1.0, factor_mode = "constant", expand_chunkwise=False, input_dim = None, output_dim=None, dtype = None):
        super(NLIPCANode, self).__init__(input_dim, output_dim, dtype)
        self.pca_nodes = []
        self.norm_class = norm_class
        self.feats_at_once = feats_at_once
        self.factor_projection_out = factor_projection_out
        self.factor_mode = factor_mode
        print "self.factor_projection_out=", self.factor_projection_out
        print "self.factor_mode=", self.factor_mode
        self.expand_chunkwise = expand_chunkwise
        self.norm_nodes = []
        self.lr_nodes = []
        self.exp_func = exp_func
        print "Initialization finished"
    def _train(self, x):
        num_samples = x.shape[0]
        residual_data = x.copy()
        y = numpy.zeros((num_samples, self.output_dim))
        normalized_y = numpy.zeros((num_samples, self.output_dim))
        
        print "Energy of input data is", (residual_data**2).sum()
        print "output_dim is", self.output_dim
        for feat_nr in range(0,self.output_dim,self.feats_at_once):
            pca_node = mdp.nodes.PCANode(output_dim=self.feats_at_once) #PCANode
            pca_node.train(residual_data)
            pca_node.stop_training()
            new_feature = pca_node.execute(residual_data)
            y[:,feat_nr:feat_nr+self.feats_at_once] = new_feature

            if self.norm_class != None:
                norm_node = self.norm_class()
                norm_node.train(new_feature)
                norm_node.stop_training()
                normalized_y[:,feat_nr:feat_nr+self.feats_at_once] = norm_node.execute(new_feature)
                self.norm_nodes.append(norm_node)
            else:
                normalized_y[:,feat_nr:feat_nr+self.feats_at_once] = y[:,feat_nr:feat_nr+self.feats_at_once]          
            
            if self.expand_chunkwise:
                if self.exp_func != None:
                    expanded_data = self.exp_func(normalized_y[:,feat_nr:feat_nr+self.feats_at_once])
                else:
                    expanded_data = normalized_y[:,feat_nr:feat_nr+self.feats_at_once]
            else:
                if self.exp_func != None:
                    expanded_data = self.exp_func(normalized_y[:,0:feat_nr+self.feats_at_once])
                else:
                    expanded_data = normalized_y[:,0:feat_nr+self.feats_at_once]
            lr_node = mdp.nodes.LinearRegressionNode(use_pinv=True)
            print "AA Energy of residual data is", (residual_data**2).sum()
            lr_node.train(expanded_data, residual_data)
            lr_node.stop_training()
            residual_data_app = lr_node.execute(expanded_data)
            print "BB Energy of residual data app is", (residual_data_app**2).sum()
            if self.factor_mode == "constant":
                effective_projection_factor = self.factor_projection_out
            elif self.factor_mode == "increasing":
                k = feat_nr * 1.0 / (self.output_dim-self.feats_at_once)
                effective_projection_factor = k*1.0 + (1.0-k) * self.factor_projection_out
            elif self.factor_mode == "decreasing":
                k = feat_nr * 1.0 / (self.output_dim-self.feats_at_once)
                effective_projection_factor = (1-k)*1.0 + k*self.factor_projection_out  
            else:
                ex = "unknown factor_mode:", self.factor_mode
                raise Exception(ex)
            residual_data = residual_data - residual_data_app * effective_projection_factor
            self.pca_nodes.append(pca_node)
            self.lr_nodes.append(lr_node)
            print "CC Energy of residual data is", (residual_data**2).sum()
    #TODO: FIX AMPLITUDE OF OUTPUT FEATURES!
    def _execute(self, x):
        num_samples = x.shape[0]
        residual_data = x.copy()
        y = numpy.zeros((num_samples, self.output_dim))
        normalized_y = numpy.zeros((num_samples, self.output_dim))
       
        for feat_nr in range(0,self.output_dim,self.feats_at_once):
            pca_node = self.pca_nodes[feat_nr/self.feats_at_once]
            new_feature = pca_node.execute(residual_data)
            y[:,feat_nr:feat_nr+self.feats_at_once] = new_feature

            if self.norm_class != None:
                norm_node = self.norm_nodes[feat_nr/self.feats_at_once]
                normalized_y[:,feat_nr:feat_nr+self.feats_at_once] = norm_node.execute(new_feature)
            else:
                normalized_y[:,feat_nr:feat_nr+self.feats_at_once] = y[:,feat_nr:feat_nr+self.feats_at_once]          
            
            if self.expand_chunkwise:
                if self.exp_func != None:
                    expanded_data = self.exp_func(normalized_y[:,feat_nr:feat_nr+self.feats_at_once])
                else:
                    expanded_data = normalized_y[:,feat_nr:feat_nr+self.feats_at_once]
            else:
                if self.exp_func != None:
                    expanded_data = self.exp_func(normalized_y[:,0:feat_nr+self.feats_at_once])
                else:
                    expanded_data = normalized_y[:,0:feat_nr+self.feats_at_once]

            lr_node = self.lr_nodes[feat_nr/self.feats_at_once]
            residual_data_app = lr_node.execute(expanded_data)
            if self.factor_mode == "constant":
                effective_projection_factor = self.factor_projection_out
            elif self.factor_mode == "increasing":
                k = feat_nr * 1.0 / (self.output_dim-self.feats_at_once)
                effective_projection_factor = k*1.0 + (1.0-k) * self.factor_projection_out
            elif self.factor_mode == "decreasing":
                k = feat_nr * 1.0 / (self.output_dim-self.feats_at_once)
                effective_projection_factor = (1-k)*1.0 + k*self.factor_projection_out  
            else:
                ex = "unknown factor_mode:", self.factor_mode
                raise Exception(ex)            
            print "effective_projection_factor=", effective_projection_factor
            residual_data = residual_data - residual_data_app * effective_projection_factor
            print "Energy of residual data is", (residual_data**2).sum()
        return y
    def _inverse(self, y):
        num_samples = y.shape[0]
        residual_data = numpy.zeros((num_samples, self.input_dim))
        normalized_y = numpy.zeros((num_samples, self.output_dim))
        if self.norm_class != None:
            for feat_nr in range(0,self.output_dim,self.feats_at_once):
                normalized_y[:,feat_nr:feat_nr+self.feats_at_once] = self.norm_nodes[feat_nr/self.feats_at_once].execute(y[:,feat_nr:feat_nr+self.feats_at_once])        
        else:
            normalized_y = y 

        for output_nr in range(self.output_dim-self.feats_at_once,-1,-1*self.feats_at_once):
            if self.expand_chunkwise:
                if self.exp_func != None:
                    expanded_data = self.exp_func(normalized_y[:,output_nr:output_nr+self.feats_at_once])
                else:
                    expanded_data = normalized_y[:,output_nr:output_nr+self.feats_at_once]
            else:
                if self.exp_func != None:
                    expanded_data = self.exp_func(normalized_y[:,0:output_nr+self.feats_at_once])
                else:
                    expanded_data = normalized_y[:,0:output_nr+self.feats_at_once]

            lr_node = self.lr_nodes[output_nr/self.feats_at_once]
            residual_data_app = lr_node.execute(expanded_data)
            if self.factor_mode == "constant":
                effective_projection_factor = self.factor_projection_out
            elif self.factor_mode == "increasing":
                k = feat_nr * 1.0 / (self.output_dim-self.feats_at_once)
                effective_projection_factor = k*1.0 + (1.0-k) * self.factor_projection_out
            elif self.factor_mode == "decreasing":
                k = feat_nr * 1.0 / (self.output_dim-self.feats_at_once)
                effective_projection_factor = (1-k)*1.0 + k*self.factor_projection_out  
            else:
                ex = "unknown factor_mode:", self.factor_mode
                raise Exception(ex)
            print "effective_projection_factor=", effective_projection_factor
            residual_data = residual_data + residual_data_app * effective_projection_factor
            print "Energy of residual data is", (residual_data**2).sum()
        return residual_data
        
    def is_trainable(self):
        return True
    def is_invertible(self):
        return True
    #TODO: code inversion function

# class NLIPCANode(mdp.Node):
#     def __init__(self, exp_func = None, norm_class = None, input_dim = None, output_dim=None, dtype = None):
#         super(NLIPCANode, self).__init__(input_dim, output_dim, dtype)
#         self.pca_nodes = []
#         self.norm_class = norm_class
#         self.norm_nodes = []
#         self.lr_nodes = []
#         self.exp_func = exp_func
#         print "Initialization finished"
#     def _train(self, x):
#         num_samples = x.shape[0]
#         residual_data = x.copy()
#         y = numpy.zeros((num_samples, self.output_dim))
#         normalized_y = numpy.zeros((num_samples, self.output_dim))
#         
#         print "Energy of input data is", (residual_data**2).sum()
#         print "output_dim is", self.output_dim
#         for feat_nr in range(self.output_dim):
#             pca_node = mdp.nodes.PCANode(output_dim=1)
#             pca_node.train(residual_data)
#             pca_node.stop_training()
#             new_feature = pca_node.execute(residual_data)
#             y[:,feat_nr] = new_feature.flatten()
# 
#             if self.norm_class != None:
#                 norm_node = self.norm_class()
#                 norm_node.train(new_feature)
#                 norm_node.stop_training()
#                 normalized_y[:,feat_nr] = norm_node.execute(new_feature).flatten()
#                 self.norm_nodes.append(norm_node)
#             else:
#                 normalized_y[:,feat_nr] = y[:,feat_nr]          
# 
#             if self.exp_func != None:
#                 expanded_data = self.exp_func(normalized_y[:,0:feat_nr+1])
#             else:
#                 expanded_data = normalized_y[:,0:feat_nr+1]
#             lr_node = mdp.nodes.LinearRegressionNode(use_pinv=True)
#             print "AA Energy of residual data is", (residual_data**2).sum()
#             lr_node.train(expanded_data, residual_data)
#             lr_node.stop_training()
#             residual_data_app = lr_node.execute(expanded_data)
#             print "BB Energy of residual data app is", (residual_data_app**2).sum()
#             residual_data = residual_data - residual_data_app
#             self.pca_nodes.append(pca_node)
#             self.lr_nodes.append(lr_node)
#             print "CC Energy of residual data is", (residual_data**2).sum()
#     #TODO: FIX AMPLITUDE OF OUTPUT FEATURES!
#     def _execute(self, x):
#         num_samples = x.shape[0]
#         residual_data = x.copy()
#         y = numpy.zeros((num_samples, self.output_dim))
#         normalized_y = numpy.zeros((num_samples, self.output_dim))
#        
#         for feat_nr in range(self.output_dim):
#             pca_node = self.pca_nodes[feat_nr]
#             new_feature = pca_node.execute(residual_data)
#             y[:,feat_nr] = new_feature.flatten()
# 
#             if self.norm_class != None:
#                 norm_node = self.norm_nodes[feat_nr]
#                 normalized_y[:,feat_nr] = norm_node.execute(new_feature).flatten()
#             else:
#                 normalized_y[:,feat_nr] = y[:,feat_nr]          
#             
#             if self.exp_func != None:
#                 expanded_data = self.exp_func(normalized_y[:,0:feat_nr+1])
#             else:
#                 expanded_data = normalized_y[:,0:feat_nr+1]
#             lr_node = self.lr_nodes[feat_nr]
#             residual_data_app = lr_node.execute(expanded_data)
#             residual_data = residual_data - residual_data_app
#             print "Energy of residual data is", (residual_data**2).sum()
#         return y
#     def _inverse(self, y):
#         num_samples = y.shape[0]
#         residual_data = numpy.zeros((num_samples, self.input_dim))
#         normalized_y = numpy.zeros((num_samples, self.output_dim))
#         if self.norm_class != None:
#             for feat_nr in range(self.output_dim):
#                 normalized_y[:,feat_nr] = self.norm_nodes[feat_nr].execute(y[:,feat_nr:feat_nr+1]).flatten()           
#         else:
#             normalized_y = y 
# 
#         for output_nr in range(self.output_dim-1,-1,-1):
#             if self.exp_func != None:
#                 expanded_data = self.exp_func(normalized_y[:,0:output_nr+1])
#             else:
#                 expanded_data = normalized_y[:,0:output_nr+1]
#             lr_node = self.lr_nodes[output_nr]
#             residual_data_app = lr_node.execute(expanded_data)
#             residual_data = residual_data + residual_data_app
#             print "Energy of residual data is", (residual_data**2).sum()
#         return residual_data
#         
#     def is_trainable(self):
#         return True
#     def is_invertible(self):
#         return True
#     #TODO: code inversion function
    

def increment(feature_vector, degree, num_vars, max_sel_vars):
    feature_vector[degree-1] += 1
    
    if feature_vector[degree-1] < num_vars: 
        #print "A"
        if len(set(feature_vector)) <= max_sel_vars: #allows increment if current_num_vars allows it
            #print "AA"
            #print "max_sel_vars", max_sel_vars
            #print "len(set(feature_vector))", len(set(feature_vector))
            return True
        else:
            #print "AB"
            pos = degree-2
            incremented = feature_vector[degree-2]
            while (feature_vector[pos] == incremented) and pos >= 0:
                pos -= 1
            
            if pos >= 0:
                for x in range(pos, degree):
                    feature_vector[x] = incremented
            else:
                for x in range(0, degree):
                    feature_vector[x] = incremented+1
            return True
    else:
        #print "B"
        if degree < 2:
            return False
        pos = degree-2
        if feature_vector[pos] == num_vars-1:
            #print "BA"
            while (feature_vector[pos] == num_vars-1) and pos >= 0:
                pos -= 1
            #print "pos2=", pos
            if pos >= 0:
                incremented = feature_vector[pos]+1
                for x in range(pos, degree):
                    feature_vector[x] = incremented
                return True
            else:
                return False
        else:
            #print "BB", "pos=", pos
            feature_vector[pos] = feature_vector[pos]+1
            feature_vector[pos+1] = feature_vector[pos]
            return True
        
def cos_exp_eE_mM_F(x, exact_degree, max_sel_vars, keep_identity=False):
    num_samples, num_vars = x.shape
    if num_vars == 0:
        return numpy.zeros((num_samples,0))
    feature_vector = numpy.zeros(exact_degree, dtype="int")
    all_feature_vectors = []
    all_feature_vectors.append(list(feature_vector))
    while (increment(feature_vector, exact_degree, num_vars, max_sel_vars)):
        #print "feature_vector=", feature_vector
        all_feature_vectors.append(list(feature_vector))
    #print "all_feature_vectors=", all_feature_vectors
    exp_dim = len(all_feature_vectors)
    print "expanded dim is", exp_dim, " ",
    dexp = numpy.zeros((num_samples, exp_dim))
    for i, feature_vector in enumerate(all_feature_vectors):
        #print "feature_vector=", feature_vector
        #print "x[feature_vector]=",x[:,feature_vector]
        dexp[:,i] = x[:,feature_vector].sum(axis=1)
    if keep_identity:
        #print "*****************"
        dexp[:, 0:num_vars] = x
        dexp[:, num_vars:] = numpy.cos(dexp[:,num_vars:]*numpy.pi)
        return dexp
    else:
        return numpy.cos(dexp*numpy.pi)

def cos_exp_mix1_F(x):
    expanded_data = []
    expanded_data.append(x)
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:30], exact_degree=2, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:25], exact_degree=3, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:20], exact_degree=4, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:15], exact_degree=5, max_sel_vars=1))
    for a in expanded_data:
        print "a.shape is", a.shape
    return numpy.hstack(expanded_data)


def cos_exp_mix2_F(x):
    expanded_data = []
    expanded_data.append(x)
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:40], exact_degree=2, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:30], exact_degree=3, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:20], exact_degree=4, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:15], exact_degree=5, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:15], exact_degree=6, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:15], exact_degree=7, max_sel_vars=1))
    for a in expanded_data:
        print " a.shape is", a.shape,
    return numpy.hstack(expanded_data)


def cos_exp_mix3_F(x):
    expanded_data = []
    expanded_data.append(x)
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:40], exact_degree=2, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:35], exact_degree=3, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:25], exact_degree=4, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:20], exact_degree=5, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:60], exact_degree=6, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:60], exact_degree=7, max_sel_vars=1))
    for a in expanded_data:
        print " a.shape is", a.shape,
    return numpy.hstack(expanded_data)

def cos_exp_mix4_F(x):
    expanded_data = []
    expanded_data.append(x)
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:40], exact_degree=2, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:30], exact_degree=3, max_sel_vars=3))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:25], exact_degree=4, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:20], exact_degree=5, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:60], exact_degree=6, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:60], exact_degree=7, max_sel_vars=1))
    for a in expanded_data:
        print " a.shape is", a.shape,
    return numpy.hstack(expanded_data)


def cos_exp_mix5_F(x):
    expanded_data = []
    expanded_data.append(x)
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:60], exact_degree=2, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:30], exact_degree=3, max_sel_vars=3))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:30], exact_degree=4, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:25], exact_degree=5, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:60], exact_degree=6, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:60], exact_degree=7, max_sel_vars=1))
    for a in expanded_data:
        print " a.shape is", a.shape,
    return numpy.hstack(expanded_data)



def cos_exp_mix6_F(x):
    expanded_data = []
    expanded_data.append(x)
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:40], exact_degree=2, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:30], exact_degree=3, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:20], exact_degree=4, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:15], exact_degree=5, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:15], exact_degree=6, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:15], exact_degree=7, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:15], exact_degree=8, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:15], exact_degree=9, max_sel_vars=1))
    for a in expanded_data:
        print " a.shape is", a.shape,
    return numpy.hstack(expanded_data)

def cos_exp_mix7_F(x):
    expanded_data = []
    expanded_data.append(x)
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:40], exact_degree=2, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,40:],  exact_degree=2, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:30], exact_degree=3, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,30:],  exact_degree=3, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:20], exact_degree=4, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,20:],  exact_degree=4, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:15], exact_degree=5, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,15:],  exact_degree=5, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:],   exact_degree=6, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:],   exact_degree=7, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:],   exact_degree=8, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:],   exact_degree=9, max_sel_vars=1))
    for a in expanded_data:
        print " a.shape is", a.shape,
    return numpy.hstack(expanded_data)

def cos_exp_mix8_F(x):
    expanded_data = []
    expanded_data.append(x)
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:60], exact_degree=2, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,60:],  exact_degree=2, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:25], exact_degree=3, max_sel_vars=3))
    expanded_data.append(cos_exp_eE_mM_F(x[:,25:],  exact_degree=3, max_sel_vars=1))
#    expanded_data.append(cos_exp_eE_mM_F(x[:,0:30], exact_degree=4, max_sel_vars=2)) #30, 2 vars
#    expanded_data.append(cos_exp_eE_mM_F(x[:,30:],  exact_degree=4, max_sel_vars=1))
#    expanded_data.append(cos_exp_eE_mM_F(x[:,0:15], exact_degree=5, max_sel_vars=2))
#    expanded_data.append(cos_exp_eE_mM_F(x[:,15:],  exact_degree=5, max_sel_vars=1))
#    expanded_data.append(cos_exp_eE_mM_F(x[:,0:],   exact_degree=6, max_sel_vars=1))
#    expanded_data.append(cos_exp_eE_mM_F(x[:,0:],   exact_degree=7, max_sel_vars=1))
#    expanded_data.append(cos_exp_eE_mM_F(x[:,0:],   exact_degree=8, max_sel_vars=1))
#    expanded_data.append(cos_exp_eE_mM_F(x[:,0:],   exact_degree=9, max_sel_vars=1))
    for a in expanded_data:
        print " a.shape is", a.shape,
    return numpy.hstack(expanded_data)


def cos_exp_mix8chunk_F(x):
    expanded_data = []
    expanded_data.append(x)
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:30], exact_degree=2, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:25], exact_degree=3, max_sel_vars=3))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:10],  exact_degree=4, max_sel_vars=4))
    for a in expanded_data:
        print " a.shape is", a.shape,
    return numpy.hstack(expanded_data)

def cos_exp_mix8block_F(x):
    expanded_data = []
    expanded_data.append(x)
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:60], exact_degree=2, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:25], exact_degree=3, max_sel_vars=3))
    expanded_data.append(cos_exp_eE_mM_F(x[:,25:50], exact_degree=3, max_sel_vars=3))
#    expanded_data.append(cos_exp_eE_mM_F(x[:,0:10],  exact_degree=4, max_sel_vars=4))
    for a in expanded_data:
        print " a.shape is", a.shape,
    return numpy.hstack(expanded_data)

def cos_exp_mix8block25n20n15c_F(x):
    expanded_data = []
    expanded_data.append(x)
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:60], exact_degree=2, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:25], exact_degree=3, max_sel_vars=3))
    expanded_data.append(cos_exp_eE_mM_F(x[:,25:45], exact_degree=3, max_sel_vars=3))
    expanded_data.append(cos_exp_eE_mM_F(x[:,45:60], exact_degree=3, max_sel_vars=3))
#    expanded_data.append(cos_exp_eE_mM_F(x[:,0:10],  exact_degree=4, max_sel_vars=4))
    for a in expanded_data:
        print " a.shape is", a.shape,
    return numpy.hstack(expanded_data)



def cos_exp_mix9_F(x):
    expanded_data = []
    expanded_data.append(x)
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:40], exact_degree=2, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,40:],  exact_degree=2, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:35], exact_degree=3, max_sel_vars=3))
    expanded_data.append(cos_exp_eE_mM_F(x[:,35:120],  exact_degree=3, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:120],  exact_degree=4, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:120],  exact_degree=5, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:120],  exact_degree=6, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:120],  exact_degree=7, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:120],  exact_degree=8, max_sel_vars=1))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:120],  exact_degree=9, max_sel_vars=1))
    for a in expanded_data:
        print " a.shape is", a.shape,
    return numpy.hstack(expanded_data)

def cos_exp_mix35_F(x):
    expanded_data = []
    expanded_data.append(x)
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:60], exact_degree=2, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:35], exact_degree=3, max_sel_vars=3))
    for a in expanded_data:
        print " a.shape is", a.shape,
    return numpy.hstack(expanded_data)

def cos_exp_mix25_F(x):
    expanded_data = []
    expanded_data.append(x)
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:25], exact_degree=2, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:25], exact_degree=3, max_sel_vars=3))
    for a in expanded_data:
        print " a.shape is", a.shape,
    return numpy.hstack(expanded_data)

def cos_exp_mix30_F(x):
    expanded_data = []
    expanded_data.append(x)
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:30], exact_degree=2, max_sel_vars=2))
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:30], exact_degree=3, max_sel_vars=3))
    for a in expanded_data:
        print " a.shape is", a.shape,
    return numpy.hstack(expanded_data)

def cos_exp_mix60q_F(x):
    expanded_data = []
    expanded_data.append(x[:,0:60])
    expanded_data.append(cos_exp_eE_mM_F(x[:,0:60], exact_degree=2, max_sel_vars=2))
#    expanded_data.append(cos_exp_eE_mM_F(x[:,0:35], exact_degree=3, max_sel_vars=3))
    for a in expanded_data:
        print " a.shape is", a.shape,
    return numpy.hstack(expanded_data)

testing = False #or True

if testing:
    print "testing histogram equalizer"
    num_samples = 1000
    x = numpy.random.normal(size = num_samples)
    eq_num_pivots = 200
    eq_pivots, eq_pivot_outputs, eq_pivot_factor = learn_histogram_equalizer(x, eq_num_pivots)
    hx = histogram_equalizer(x, eq_num_pivots, eq_pivots, eq_pivot_outputs, eq_pivot_factor)
    sorting = numpy.argsort(x)
    hx_sorted = hx[sorting]
    hx_dif = hx_sorted[1:]-hx_sorted[:-1]
    mask = hx_dif < 0
    print "mask.sum()=", mask.sum()
    hx.sort()
    print "hx.sorted=", hx
    print "pivots=", eq_pivots
    print "pivot_outputs=", eq_pivot_outputs
    print "pivot_factor=", eq_pivot_factor   
 
    num_samples_y = 50000
    y = numpy.random.normal(size = num_samples_y)
    y.sort()
    hy = histogram_equalizer(y, eq_num_pivots, eq_pivots, eq_pivot_outputs, eq_pivot_factor)
    print "hy[:20]=", hy[:20]
    print "hy[-20:]=",hy[-20:]
    #print  "hy.sorted=", hy
    hy_dif = hy[1:]-hy[:-1]
    mask = hy_dif < 0
    print "mask.sum()=", mask.sum()
    print "test finished\n"
    #quit()

if testing:
    print "Testing HistogramEqualizationNode"
    num_samples = 10000
    x = numpy.random.normal(size = (num_samples,2)) - 30
    hen = HistogramEqualizationNode(num_pivots=300)
    hen.train(x)
    hen.stop_training()
    x.sort(axis=0)    
    hx = hen.execute(x)
    print "hx (sorted) =", hx
    print "hx[0] (sorted) =",hx[:,0]
    print "eq_pivots=", hen.hx
    y = numpy.random.normal(size = (num_samples*5,2)) - 30
    y.sort(axis=0)
    hy = hen.execute(y)
    print "hy[0:100,0] (sorted) =",hy[:100,0]
    print "hy[-100:,0] (sorted) =",hy[-100:,0]


    num_samples = 1000
    x = numpy.random.normal(size = (num_samples,2)) - 30
    hen = HistogramEqualizationNode(num_pivots=20)
    hen.train(x)
    hen.stop_training()
    hx = hen.execute(x)
    hx.sort(axis=0)
    print "hx.sorted=", hx
    print "hx[0] (sorted) =",hx[:,0]
    print "eq_pivots=", hen.hx 
    print "test finished\n"
    #quit()


if testing:
    num_samples = 1000
    x = numpy.random.normal(size = num_samples)
    
    num_pivots = 50
    y=x.copy()
    y.sort()
    indices_pivots=numpy.linspace(0, num_samples-1, num_pivots).round().astype(int)
    print indices_pivots
    pivots = y[indices_pivots]
    print pivots
    indices_sorting = numpy.searchsorted(pivots, x, side="left")
    indices_sorting[indices_sorting <= 0] = 1
    indices_sorting -= 1
    print indices_sorting
    
    #print x
    #print y
    pivot_outputs = numpy.linspace(0, 1.0, num_pivots)
    print pivot_outputs
    
    diff_pivot_outputs = pivot_outputs[1:]-pivot_outputs[:-1]
    print diff_pivot_outputs
    
    diff_pivots = pivots[1:]-pivots[:-1]
    print diff_pivots
    
    hx = pivot_outputs[indices_sorting]+(x-pivots[indices_sorting]) * diff_pivot_outputs[indices_sorting]/diff_pivots[indices_sorting]
    hx.clip(pivots[0],pivots[-1])
    print hx
    
    hy = hx.copy()
    hy.sort()
    print hy
    
    xx = numpy.random.normal(size = num_samples)
    indices_sorting = numpy.searchsorted(pivots, xx, side="left")
    indices_sorting[indices_sorting <= 0] = 1
    indices_sorting[indices_sorting >= num_pivots-1] = num_pivots-2
    indices_sorting -= 1
    pivot_factor = diff_pivot_outputs/diff_pivots
    
    hxx = pivot_outputs[indices_sorting]+(xx-pivots[indices_sorting]) * diff_pivot_outputs[indices_sorting]/diff_pivots[indices_sorting]
    
    hxx = pivot_outputs[indices_sorting]+(xx-pivots[indices_sorting]) * pivot_factor[indices_sorting]
    
    #hxx = numpy.clip(hxx, 0.0,1.0)
    
    hyy = hxx.copy()
    hyy.sort()
    print hyy

print "****************************************************"
if testing: # or True:
    numpy.random.seed(1)
    num_samples = 101
    x0 = numpy.linspace(0,1,num_samples)
    x1 = numpy.random.normal(size=num_samples)
    x2 = 0.5*numpy.random.normal(size=num_samples)
    x3 = x0**2+x1+x2
    x4 = 0.5*x0**3+0.2*x0**2-x0+x1*x2-x1*x0
    x5 = 0.1 * numpy.random.normal(size=num_samples)

    y0 = numpy.linspace(0,1,num_samples)+ 0.0001 * numpy.random.normal(size=num_samples)
    y1 = numpy.random.normal(size=num_samples)
    y2 = 0.5*numpy.random.normal(size=num_samples)
    y3 = y0**2+y1+y2
    y4 = 0.5*y0**3+0.2*y0**2-y0+ y1*y2 - y1*y0
    y5 = 0.1 * numpy.random.normal(size=num_samples)
            
    x = numpy.dstack((x0,x1,x2,x3,x4,x5))[0]
    xp = numpy.dstack((y0,y1,y2,y3,y4,y5))[0]

    print "x=", x
    # constant -> 0.7 
    # increasing -> 0.35
    # decreasing -> 0.4
    nlipca_node = NLIPCANode(exp_func = cos_exp_I_5D_F, norm_class = NormalizeABNode, feats_at_once = 1, factor_projection_out=0.35, factor_mode = "increasing", input_dim = None, output_dim=4, dtype = None)
    nlipca_node.train(x)
    print "feature extraction"
    y = nlipca_node.execute(x)
    #print "y=", y
    print "executing inverse function"
    xx = nlipca_node.inverse(y)
    print "using test data"
    zp = nlipca_node.execute(xp)
    #print "y=", y
    print "executing inverse function"
    xxp = nlipca_node.inverse(zp)

    #print "xx=", xx
    
    print ""
    nlipca_node2 = NLIPCANode(exp_func = cos_exp_I_5D_F, norm_class = NormalizeABNode, feats_at_once = 4, input_dim = None, output_dim=4, dtype = None)
    print "training second node"
    nlipca_node2.train(x)
    print "feature extraction, second node"
    y2 = nlipca_node2.execute(x)
    #print "y=", y
    print "executing inverse function, second node"
    xx2 = nlipca_node2.inverse(y2)
    print "using test data"
    zp2 = nlipca_node2.execute(xp)
    #print "y=", y
    print "executing inverse function"
    xxp2 = nlipca_node2.inverse(zp2)

    
    #print "xx2=", xx2
    print "error1=", ((xx-x)**2).sum()
    print "error2=", ((xx2-x)**2).sum()
    print "test data"
    print "error1=", ((xxp-xp)**2).sum()
    print "error2=", ((xxp2-xp)**2).sum()

if testing:
    cen = CosineExpansionNode(5) #Max factor
    x = numpy.linspace(0,1,150).reshape(-1,1)
    y2 = cen(x) #Assumes elements of x in [0,1]
    print "y2=", y2
    
if testing:
    yy2 = cos_exp_5D_F(x)
    print "yy2=", yy2
    
if testing:
    n01n= NormalizeABNode(a=-1, b=1)
    n01n.train(x-0.3)
    y3=n01n.execute(x-0.3)
    print "y3=", y3


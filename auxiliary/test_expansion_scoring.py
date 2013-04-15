import numpy
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import Image
import mdp
import more_nodes
import patch_mdp
import nonlinear_expansion as nle

num_samples_x = 1000
num_samples_y = 1000
dim_x=4 #5
std_noise_first = 0.50
std_noise_last = 2 * std_noise_first
max_comp=20
output_dim=1
k=4
seed = 1
expansion = nle.S_exp #nle.S_exp, nle.Q_N_exp , nle.T_N_exp 
scores_enable = True #and False
whitening_for_scores = True and False

numpy.random.seed(seed)
t = numpy.linspace(0, numpy.pi, num_samples_x)
factor = 20*numpy.sqrt(2)
std_noise = numpy.linspace(std_noise_first, std_noise_last, dim_x)

num_slow_signals=dim_x
sl = numpy.zeros((num_samples_x, num_slow_signals))
for i in range(num_slow_signals):
    # *(numpy.random.randint(2)*2 -1)
    sl[:,i] = factor * numpy.cos((i+1)*t)  + std_noise[i] * numpy.random.normal(size=(num_samples_x)) 
    print "sl[:,i].mean()=", sl[:,i].mean(), "  sl[:,i].var()=", sl[:,i].var()
sl = nle.fix_mean_var(sl)

sl_test = numpy.zeros((num_samples_y, num_slow_signals))
for i in range(num_slow_signals):
    # *(numpy.random.randint(2)*2 -1)
    sl_test[:,i] = factor * numpy.cos((i+1)*t)  + std_noise[i] * numpy.random.normal(size=(num_samples_y)) 
    print "sl_test[:,i].mean()=", sl_test[:,i].mean(), "  sl[:,i].var()=", sl_test[:,i].var()
sl_test = nle.fix_mean_var(sl_test)



#x  = numpy.random.normal(size=(num_samples_x, dim_x))
x = sl
x_exp = expansion(x)

#TODO:REALISTIC x,y containing some structure
#y  = x + 0.01 *numpy.random.normal(size=(num_samples_y, dim_x))
y = sl_test
y_exp = expansion(y)

if scores_enable:
    if whitening_for_scores:
        wn = mdp.nodes.WhiteningNode()
        wn.train(x_exp)
        x_exp_whitened = wn.execute(x_exp)
        y_exp_whitened = wn.execute(y_exp)
    else:
        x_exp_mean = x_exp.mean(axis=0)
        x_exp_std = x_exp.std(axis=0)
        x_exp_whitened = (x_exp - x_exp_mean)/x_exp_std
        y_exp_whitened = (y_exp - x_exp_mean)/x_exp_std
    
    scores = more_nodes.rank_expanded_signals(x, x_exp_whitened, y, y_exp_whitened, max_comp=max_comp, k=k, linear=False, verbose=True)
    print "scores=", scores
    
    x_exp_norm = x_exp_whitened * scores
    y_exp_norm = y_exp_whitened * scores
else:
    x_exp_norm = x_exp 
    y_exp_norm = y_exp 

pn = mdp.nodes.WhiteningNode(output_dim=output_dim)
pn.train(x_exp_norm)
x_exp_red = pn.execute(x_exp_norm)
y_exp_red = pn.execute(y_exp_norm)

print "x_exp_red[:,0].var()=",x_exp_red[:,0].var()
print "x_exp_red[:,0].mean()=",x_exp_red[:,0].mean()
print "x_exp_red[0:3]:", x_exp_red[0:3]
print "y_exp_red[0:3]:", y_exp_red[0:3]

#feat_error = y_exp_red - x_exp_red
#feat_RMSE = (((feat_error**2).sum(axis=1))**0.5).mean()
#print "feat_RMSE", feat_RMSE

#reconstruction:
h = numpy.reshape(t, (num_samples_x,1))
h_app = more_nodes.approximate_kNN(h, x_exp_red, y_exp_red)
error = h - h_app
error_RMSE = ((error**2).mean()) ** 0.5
print "error_RMSE", error_RMSE

import system_parameters
import numpy
import scipy
import mdp
import sfa_libs 
import more_nodes
import more_nodes as he
import patch_mdp
import imageLoader
from nonlinear_expansion import *
import lattice
import PIL
from PIL import Image
import copy
import os
import string

tuning_parameter = os.getenv("CUICUILCO_TUNING_PARAMETER") #1112223339 #1112223339
if tuning_parameter==None:
    ex = "CUICUILCO_TUNING_PARAMETER unset"
    print ex
    raise Exception(ex)
print "tuning_parameter=", tuning_parameter

#WARNING: remember that initialization overwrites these values!!!!!!!!!!
#Therefore, the next section is useless unless these class variables are read!!!!!!!!!!
activate_random_permutation = False
activate_sfa_ordering = False
if activate_random_permutation:
    print "Random Permutation Activated!!!"
    system_parameters.ParamsSFALayer.ord_node_class = more_nodes.RandomPermutationNode
    system_parameters.ParamsSFALayer.ord_args = {}
    system_parameters.ParamsSFASuperNode.ord_node_class = more_nodes.RandomPermutationNode
    system_parameters.ParamsSFASuperNode.ord_args = {}
elif activate_sfa_ordering:
    system_parameters.ParamsSFALayer.ord_node_class = mdp.nodes.GSFANode
    system_parameters.ParamsSFASuperNode.ord_node_class = mdp.nodes.GSFANode
    system_parameters.ParamsSFALayer.ord_args = {}
    system_parameters.ParamsSFASuperNode.ord_args = {}
else:
    system_parameters.ParamsSFALayer.ord_node_class = None
    system_parameters.ParamsSFASuperNode.ord_node_class = None
    system_parameters.ParamsSFALayer.ord_args = {}
    system_parameters.ParamsSFASuperNode.ord_args = {}
    
print "system_parameters.ParamsSFALayer.ord_node_class is:", system_parameters.ParamsSFALayer.ord_node_class
print "system_parameters.ParamsSFALayer.ord_args is:", system_parameters.ParamsSFALayer.ord_args


def comp_layer_name(cloneLayer, exp_funcs, x_field_channels, y_field_channels, pca_out_dim, sfa_out_dim):
    name = ""
    if cloneLayer == False:
        name += "Homogeneous "
    else:
        name += "Inhomogeneous "
    if exp_funcs == "RandomSigmoids":
        name += exp_funcs
    elif exp_funcs == [identity,]:
        name += "Linear "
    else:
        name += "Non-Linear ("
        for fun in exp_funcs:
            name += fun.__name__ + ","
        name += ") "
    name += "Layer: %dx%d => %s => %s"% (y_field_channels, x_field_channels, str(pca_out_dim), str(sfa_out_dim))
    return name


def comp_supernode_name(exp_funcs, pca_out_dim, sfa_out_dim):
    name = ""
    if exp_funcs == "RandomSigmoids":
        name += exp_funcs
    elif isinstance(exp_funcs, list):
        if exp_funcs == [identity,]:
            name += "Linear "
        else:
            name += "Non-Linear ("
            for fun in exp_funcs:
                name += fun.__name__ + ","
            name += ") "
    else:
        ex = "Not recognized expansion functions:", exp_funcs
        raise Exception(ex)
    name += "SFA Super Node:  all => " + str(pca_out_dim) + " => " + str(sfa_out_dim)
    return name


def NetworkSetExpFuncs(exp_funcs, network, include_L0=True):       
    for i, layer in enumerate(network.layers):
        if i>0 or include_L0==True:
            layer.exp_funcs = exp_funcs
        else:
            layer.exp_funcs = [identity,]
    return network


def NetworkSetSFANodeClass(sfa_node_class, network):       
    for i, layer in enumerate(network.layers):
        layer.sfa_node_class = sfa_node_class
    return network


def NetworkAddSFAArgs(sfa_args, network):       
    for i, layer in enumerate(network.layers):
        for key in sfa_args.keys():
            layer.sfa_args[key] = sfa_args[key]
    return network


def NetworkSetPCASFAExpo(network, first_pca_expo=0.0, last_pca_expo=1.0, first_sfa_expo=1.2, last_sfa_expo=1.0, hard_pca_expo=False):
    num_layers = len(network.layers)
    if num_layers > 1:
        for i, layer in enumerate(network.layers):
            if hard_pca_expo == False:
                layer.sfa_args["pca_expo"] = first_pca_expo + (last_pca_expo-first_pca_expo)*i*1.0/(num_layers-1)
            else:
                if i == num_layers-1:
                    layer.sfa_args["pca_expo"] = last_pca_expo
                else:
                    layer.sfa_args["pca_expo"] = first_pca_expo                    
            layer.sfa_args["sfa_expo"] = first_sfa_expo + (last_sfa_expo-first_sfa_expo)*i*1.0/(num_layers-1)            
    return network


####################################################################
##############               Void NETWORK               ############
####################################################################  
print "*******************************************************************"
print "********    Creating Void Layer and Network      ******************"
print "*******************************************************************"
print "******** Setting Layer L0 Parameters          *********************"
layer = pVoidLayer = system_parameters.ParamsSFASuperNode()
layer.name = "Void Layer"
layer.pca_node_class = None
layer.exp_funcs = [identity,]
layer.red_node_class = None
layer.sfa_node_class = mdp.nodes.IdentityNode
layer.sfa_args = {}
layer.sfa_out_dim = None

network = voidNetwork1L = system_parameters.ParamsNetwork()
network.name = "Void 1 Layer Network"
network.L0 = pVoidLayer
network.layers = [network.L0]


network = HeadNetwork1L = copy.deepcopy(voidNetwork1L)
network.layers[0].sfa_node_class = mdp.nodes.HeadNode
network.layers[0].sfa_out_dim = 75

c35_to_800 = numpy.zeros((35,800))
for i in range(800):
    index_35_2 = numpy.random.randint(35)
    c35_to_800[index_35_2,i]= numpy.random.normal(loc=0.0, scale=2.0, size=(1,))
    index_35_2 = numpy.random.randint(35)
    c35_to_800[index_35_2,i]= numpy.random.normal(loc=0.0, scale=2.0, size=(1,))

print "c35_to_800=", c35_to_800
print "c35_to_800.shape=", c35_to_800.shape
l35_to_800 = numpy.random.normal(loc=0.0, scale=1.0, size=800)

def random_sigmoids_pairwise_35_to_800(x):
    return extract_sigmoid_features(x[:,0:35], c35_to_800, l35_to_800)

c35_to_400 = numpy.zeros((35,400))
for i in range(400):
    index_35_2 = numpy.random.randint(35)
    c35_to_400[index_35_2,i]= numpy.random.normal(loc=0.0, scale=4.0, size=(1,))
    index_35_2 = numpy.random.randint(35)
    c35_to_400[index_35_2,i]= numpy.random.normal(loc=0.0, scale=4.0, size=(1,))

print "c35_to_400=", c35_to_400
print "c35_to_400.shape=", c35_to_400.shape
l35_to_400 = numpy.random.normal(loc=0.0, scale=0.05, size=400)

def random_sigmoids_pairwise_35_to_400(x):
    return extract_sigmoid_features(x, c35_to_400, l35_to_400)


c35_to_200 = numpy.zeros((35,200))
for i in range(200):
    index_35_2 = numpy.random.randint(35)
    c35_to_200[index_35_2,i]= numpy.random.normal(loc=0.0, scale=2.0, size=(1,))
    index_35_2 = numpy.random.randint(35)
    c35_to_200[index_35_2,i]= numpy.random.normal(loc=0.0, scale=2.0, size=(1,))

print "c35_to_200=", c35_to_200
print "c35_to_200.shape=", c35_to_200.shape
l35_to_200 = numpy.random.normal(loc=0.0, scale=0.05, size=200)

def random_sigmoids_pairwise_35_to_200(x):
    return extract_sigmoid_features(x, c35_to_200, l35_to_200)

def unsigned_08expo_100(x):
    return unsigned_08expo(x[:,0:100])

def encode_signal_p9(x):
    dim = x.shape[1]
    x_out = x+0.0
    for i in range(9, dim): #9
        yy = (-1.0) ** (numpy.floor(numpy.abs(x[:, i])*2.0)%2) # if a = (+/-)**.1*** then a=> -(+/-)a, otherwise a=>a        
        print yy
        x_out[:, i] = x_out[:, i]*yy
    return x_out



print "*******************************************************************"
print "*******    MNIST DIRECT NETWORK                   *****************"
print "*******************************************************************"
layer = pSFADirectLayer = system_parameters.ParamsSFASuperNode()
layer.name = "Direct SFA Layer for MNIST"
layer.pca_node_class = mdp.nodes.PCANode #None  #None
layer.pca_args = {}
layer.pca_out_dim = 100 #40 # 35 #WARNING: 100 or None
layer.exp_funcs = [identity]
layer.sfa_node_class =  mdp.nodes.GSFANode #mdp.nodes.GSFANode #mdp.nodes.GSFANode
layer.sfa_out_dim = 35 #3 #49*2 # *3 # None

layer = pSFADirectLayer2 = system_parameters.ParamsSFASuperNode()
layer.name = "Direct SFA Layer2"
layer.pca_node_class = None  #None
layer.pca_args = {}
#layer.exp_funcs = [identity, QT]
layer.exp_funcs = [identity, QT, CT] #unsigned_08expo, signed_08expo
layer.sfa_node_class = mdp.nodes.GSFANode #mdp.nodes.GSFANode
layer.sfa_out_dim = 20 
####################################################################
#####terms_NL_expansion == 15:
#        One-Layer Linear SFA NETWORK              ############
####################################################################  
network = MNIST_8C_DirectNetwork = system_parameters.ParamsNetwork()
network.name = "SFA Direct Network for MNIST"
network.L0 = pSFADirectLayer
network.L1 = pSFADirectLayer2
network.L2 = None
network.L3 = None
network.L4 = None
network.layers = [network.L0, network.L1]


################ Simple network for face centering with direct GSFA
layer = pPCA_GSFA_DirectLayer = system_parameters.ParamsSFASuperNode()
layer.name = "Direct PCA-GSFA Layer for FaceCentering"
layer.pca_node_class = mdp.nodes.PCANode #None  #None
layer.pca_args = {}
layer.pca_out_dim = 75 #40 # 35 #WARNING: 100 or None
layer.exp_funcs = [identity, unsigned_08expo] 
layer.sfa_node_class = mdp.nodes.GSFANode
layer.sfa_out_dim = 30

#NETWORK MNIST Net0
network = PCA_GSFA_Network1L = system_parameters.ParamsNetwork()
network.name = "Direct PCA-GSFA Network for FaceCentering"
network.L0 = pPCA_GSFA_DirectLayer
network.L1 = None #pSFADirectLayer2
network.L2 = None
network.L3 = None
network.L4 = None
network.layers = [network.L0]

network = PCA_QT35_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [identity, s35QT] #unsigned_08expo

#NETWORK MNIST Net1
network = PCA_QT40_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [identity, s40QT] #unsigned_08expo

#NETWORK MNIST NetB
network = PCA_U08_QT40_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [identity, unsigned_08expo, s40QT] #unsigned_08expo


network = PCA_control4linear_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control4_linear, ] #unsigned_08expo

network = PCA_control4_QT20_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control4_QT20, ] #unsigned_08expo

network = PCA_control4_QT30_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control4_QT30, ] #unsigned_08expo

network = PCA_control4_QT40_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control4_QT40, ] #unsigned_08expo

network = PCA_control1_QT40_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control1_QT40, ] #unsigned_08expo

network = PCA_control2_QT40_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control2_QT40, ] #unsigned_08expo

network = PCA_control2_QT60_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control2_QT60, ] #unsigned_08expo

network = PCA_control3_QT40_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control3_QT40, ] #unsigned_08expo

network = PCA_control5_QT40_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control5_QT40, ] #unsigned_08expo

network = PCA_control6_QT40_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control6_QT40, ] #unsigned_08expo

network = PCA_control6_QT50_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control6_QT50, ] #unsigned_08expo


network = PCA_control4_QT50_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control4_QT50, ] #unsigned_08expo

network = PCA_control4_QT60_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control4_QT60, ] #unsigned_08expo

network = PCA_control5_QT60_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control5_QT60, ] #unsigned_08expo

network = PCA_control6_QT60_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control6_QT60, ] #unsigned_08expo

network = PCA_control8_QT60_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control8_QT60, ] #unsigned_08expo

network = PCA_control9_QT60_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control9_QT60, ] #unsigned_08expo



network = PCA_control2_QT40_CT10_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control2_QT40_CT10, ] #unsigned_08expo

network = PCA_control2_QT40_CT15_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control2_QT40_CT15, ] #unsigned_08expo

network = PCA_control2_QT40_CT20_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control2_QT40_CT20, ] #unsigned_08expo 

network = PCA_control2_QT40_CT25_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control2_QT40_CT25, ] #unsigned_08expo

network = PCA_control3_QT40_CT25_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control3_QT40_CT25, ] #unsigned_08expo

network = PCA_control2_QT40_CT30_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control2_QT40_CT30, ] #unsigned_08expo

network = PCA_control2_QT40_CT35_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control2_QT40_CT35, ] #unsigned_08expo

network = PCA_control2_QT50_CT25_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control2_QT50_CT25, ] #unsigned_08expo

network = PCA_control3_QT50_CT25_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control3_QT50_CT25, ] #unsigned_08expo

network = PCA_control6_QT50_CT10_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control6_QT50_CT10, ] #unsigned_08expo

network = PCA_control6_QT50_CT15_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control6_QT50_CT15, ] #unsigned_08expo

network = PCA_control6_QT50_CT20_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control6_QT50_CT20, ] #unsigned_08expo

network = PCA_QT60_CT30_control6_QT40_CT10_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT60_CT30_control6_QT40_CT10, ] #unsigned_08expo

network = PCA_QT60_CT35_control6_QT40_CT10_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT60_CT35_control6_QT40_CT10, ] #unsigned_08expo

network = PCA_QT60_CT25_control3_QT40_CT10_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT60_CT25_control3_QT40_CT10, ] #unsigned_08expo

network = PCA_QT50_CT20_control9_QT30_CT5_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT50_CT20_control9_QT30_CT5, ] #unsigned_08expo

network = PCA_QT50_CT20_control5_QT35_CT5_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT50_CT20_control5_QT35_CT5, ] #unsigned_08expo

network = PCA_QT50_CT20_control5_QT40_CT5_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT50_CT20_control5_QT40_CT5, ] #unsigned_08expo

network = PCA_QT50_CT20_control5_QT45_CT5_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT50_CT20_control5_QT45_CT5, ] #unsigned_08expo

network = PCA_QT50_CT20_control5_QT50_CT5_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT50_CT20_control5_QT50_CT5, ] #unsigned_08expo

network = PCA_QT50_CT20_control5_QT50_CT10_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT50_CT20_control5_QT50_CT10, ] #unsigned_08expo

network = PCA_QT50_CT20_control5_QT50_CT15_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT50_CT20_control5_QT50_CT15, ] #unsigned_08expo


network = PCA_QT40_CT25_control5_QT30_CT5_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT40_CT25_control5_QT30_CT5, ] #unsigned_08expo

network = PCA_QT40_CT35_control5_QT30_CT5_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT40_CT35_control5_QT30_CT5, ] #unsigned_08expo

network = PCA_QT40_CT30_control5_QT30_CT5_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT40_CT30_control5_QT30_CT5, ] #unsigned_08expo

network = PCA_QT50_CT20_control9_QT40_CT5_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT50_CT20_control9_QT40_CT5, ] #unsigned_08expo

network = PCA_QT60_CT30_control9_QT30_CT5_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT60_CT30_control9_QT30_CT5, ] #unsigned_08expo


network = PCA_QT50_CT20_control9_QT25_CT5_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT50_CT20_control9_QT25_CT5, ] #unsigned_08expo

network = PCA_QT50_CT20_control9_QT35_CT5_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT50_CT20_control9_QT35_CT5, ] #unsigned_08expo

network = PCA_QT50_CT20_control9_QT45_CT5_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT50_CT20_control9_QT45_CT5, ] #unsigned_08expo

network = PCA_QT60_CT35_control4_QT40_CT5_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [QT60_CT35_control4_QT40_CT5, ] #unsigned_08expo

#NETWORK MNIST NetA
network = PCA_QT50_CT25_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [identity, s50QT, s25CT, ] #unsigned_08expo


network = PCA_control2_QT60_CT25_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control2_QT60_CT25, ] #unsigned_08expo

network = PCA_control2_QT70_CT25_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control2_QT70_CT25, ] #unsigned_08expo

network = PCA_control2_QT75_CT25_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [control2_QT75_CT25, ] #unsigned_08expo



network = PCA_QT40_CT15_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [identity, s40QT, s15CT] #unsigned_08expo

network = PCA_QT40_CT20_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [identity, s40QT, s20CT] #unsigned_08expo

network = PCA_QT40_CT25_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [identity, s40QT, s25CT] #unsigned_08expo

network = PCA_QT40_CT30_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [identity, s40QT, s30CT] #unsigned_08expo

network = PCA_QT45_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [identity, s45QT] #unsigned_08expo

network = PCA_QT45_CT25_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [identity, s45QT, s25CT] #unsigned_08expo

network = PCA_QT50_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [identity, s50QT] #unsigned_08expo

network = PCA_QT50_CT25_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [identity, s50QT, s25CT] #unsigned_08expo

network = PCA_QT55_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [identity, s55QT] #unsigned_08expo

network = PCA_QT35_CT25_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [identity, s35QT, s25CT] #unsigned_08expo

network = PCA_QT30_CT25_GSFA_Network1L = copy.deepcopy(PCA_GSFA_Network1L)
network.L0.exp_funcs = [identity, s30QT, s25CT] #unsigned_08expo


#QT50_CT20_control9_QT30_CT5

print "*******************************************************************"
print "*******    Creating Direct Linear SFA Network     *****************"
print "*******************************************************************"
print "******** Setting Layer L0 Parameters          *********************"
layer = pSFADirectLayer = system_parameters.ParamsSFASuperNode()
layer.name = "Direct SFA Layer"
#layer.pca_node_class = None # mdp.nodes.GSFANode
#W
layer.pca_node_class = mdp.nodes.PCANode #None  #None
#layer.pca_args = {}
layer.pca_out_dim = 100 #35 #WARNING: 100 or None
#layer.pca_out_dim = 100#3000 # 35 #WARNING: 100 or None
#layer.exp_funcs = [identity, QT]
layer.exp_funcs = [identity,] #unsigned_08expo, signed_08expo
#layer.exp_funcs = [he.cos_exp_mix8_F]
#layer.exp_funcs = [encode_signal_p9,] #For next experiment: [encode_signal_p9,]
#layer.red_node_class = mdp.nodes.HeadNode
#layer.red_out_dim = int(tuning_parameter)
layer.sfa_node_class =  mdp.nodes.SFANode #mdp.nodes.GSFANode #mdp.nodes.GSFANode
layer.sfa_out_dim = 100 #3 #49*2 # *3 # None

#print layer.sfa_args
#quit()

layer = pSFADirectLayer2 = system_parameters.ParamsSFASuperNode()
layer.name = "Direct SFA Layer2"
#layer.pca_node_class = None # mdp.nodes.GSFANode
#W
layer.pca_node_class = mdp.nodes.GSFANode
layer.pca_args = {}
layer.pca_out_dim = 3 #WARNING: 100 or None
#layer.pca_out_dim = 35 #WARNING: 100 or None
#layer.pca_out_dim = 200 # 35 #WARNING: 100 or None
#layer.exp_funcs = [identity, QT]
layer.exp_funcs = [identity, P5, P4, CT, QT, unsigned_08expo, signed_08expo] #unsigned_08expo, signed_08expo
#layer.exp_funcs = [he.cos_exp_mix8_F]
#layer.exp_funcs = [encode_signal_p9,] #For next experiment: [encode_signal_p9,]
#layer.red_node_class = mdp.nodes.HeadNode
#layer.red_out_dim = int(tuning_parameter)
layer.sfa_node_class = mdp.nodes.GSFANode #mdp.nodes.GSFANode
layer.sfa_out_dim = 3 
####################################################################
#####terms_NL_expansion == 15:
#        One-Layer Linear SFA NETWORK              ############
####################################################################  
network = SFADirectNetwork1L = system_parameters.ParamsNetwork()
network.name = "SFA 1 Layer Direct Network"
network.L0 = pSFADirectLayer
network.L1 = None #pSFADirectLayer2
network.L2 = None
network.L3 = None
network.L4 = None
network.layers = [network.L0]

print "*******************************************************************"
print "********    Creating One-Layer Linear SFA Network            ******************"
print "*******************************************************************"
print "******** Setting Layer L0 Parameters          *********************"
layer = pSFAOneLayer = system_parameters.ParamsSFASuperNode()
layer.name = "One-Node SFA Layer"
#layer.pca_node_class = None # mdp.nodes.GSFANode
#W
layer.pca_node_class = mdp.nodes.PCANode
layer.pca_args = {}
layer.pca_out_dim = 35 #WARNING: 100 or None
#layer.ord_node_class = mdp.nodes.iGSFANode
#layer.ord_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, QT],"max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.999, "output_dim":35} #output_dim":40
#layer.ord_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity],"max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.999, "output_dim":35} #output_dim":40
#layer.exp_funcs = [identity,]
#layer.exp_funcs = [encode_signal_p9,] #For next experiment: [encode_signal_p9,]
#layer.red_node_class = mdp.nodes.HeadNode
#layer.red_out_dim = int(tuning_parameter)
layer.sfa_node_class = mdp.nodes.iGSFANode #mdp.nodes.GSFANode
#layer.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":"RandomSigmoids", "expansion_starting_point":"08Exp", "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999, "expansion_output_dim":4000} 
#layer.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo],                       "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
#QT10, CT10, QT15, CT15, QT20, CT20, QT25, CT25,
terms_NL_expansion = int(tuning_parameter)
if terms_NL_expansion == 15:
    expansion = [identity, s15QT, s15CT]
elif terms_NL_expansion == 20:
    expansion = [identity, s20QT, s20CT]
elif terms_NL_expansion == 25:
    expansion = [identity, s25QT, s25CT]
elif terms_NL_expansion == 30:
    expansion = [identity, s30QT, s30CT]
elif terms_NL_expansion == 35:
    expansion = [identity, s35QT, s35CT]
elif terms_NL_expansion == 40:
    expansion = [identity, s40QT, s40CT]
else:
    er = "invalid size of NL expansion", terms_NL_expansion
    raise Exception(er) 

expansion = [identity, QT, s30CT] 

layer.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":expansion,"max_lenght_slow_part":None,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.999} 
layer.sfa_out_dim = 9 #49*2 # *3 # None

####################################################################
#####terms_NL_expansion == 15:
#        One-Layer Linear SFA NETWORK              ############
####################################################################  
network = SFANetwork1L = system_parameters.ParamsNetwork()
network.name = "SFA 1 Layer Linear Network"
network.L0 = pSFAOneLayer
network.L1 = None
network.L2 = None
network.L3 = None
network.L4 = None
network.layers = [network.L0]





network = PCANetwork1L = copy.deepcopy(SFANetwork1L)
network.L0.pca_node_class = None
network.L0.pca_args = {}
network.L0.sfa_node_class = mdp.nodes.PCANode #WhiteningNode
network.L0.sfa_out_dim = 75 #30 # 100 #49 * 2 # *3
network.L0.sfa_args = {}


network = CESFANetwork1L = copy.deepcopy(SFANetwork1L)
network.L0.pca_node_class = mdp.nodes.PCANode
network.L0.pca_out_dim = 200 # 100 #49 * 2 # *3
network.L0.pca_args = {}
network.L0.ord_node_class = he.NormalizeABNode #more_nodes.HistogramEqualizationNode #NormalizeABNode
#network.L0.ord_args = {"num_pivots":200}
network.L0.exp_funcs = [he.cos_exp_mix30_F] #[identity, sel30_QT, sel30_CT ] he.cos_exp_mix30_F] # sel60_QT # he.cos_exp_mix25_F # he.cos_exp_mix60q_F , sel25_QT, sel25_CT
network.L0.sfa_node_class = mdp.nodes.GSFANode #IdentityNode #PCANode # mdp.nodes.NLIPCANode #WhiteningNode
network.L0.sfa_out_dim = 60 # 100 #49 * 2 # *3
network.L0.sfa_args = {}
#network.L0.sfa_args = {"exp_func":he.cos_exp_mix8_F, "feats_at_once":60, "norm_class":he.NormalizeABNode,} #he.cos_exp_I_3D_F, smartD, he.cos_exp_I_smart2D_F, cos_exp_mix1_F


network = NLIPCANetwork1L = copy.deepcopy(SFANetwork1L)
network.L0.pca_node_class = mdp.nodes.PCANode
network.L0.pca_out_dim = 300 # 100 #49 * 2 # *3
network.L0.pca_args = {}
network.L0.sfa_node_class = mdp.nodes.NLIPCANode #IdentityNode #PCANode # mdp.nodes.NLIPCANode #WhiteningNode
network.L0.sfa_out_dim = 60 # 100 #49 * 2 # *3
#network.L0.sfa_args = {"exp_func":he.cos_exp_mix8block25n20n15c_F, "factor_projection_out":0.72, "factor_mode"":"constant", "feats_at_once":10, "norm_class":he.NormalizeABNode,} #he.cos_exp_I_3D_F, smartD, he.cos_exp_I_smart2D_F, cos_exp_mix1_F
network.L0.sfa_args = {"exp_func":he.cos_exp_mix8_F, "factor_projection_out":0.97, "factor_mode":"decreasing", "feats_at_once":2, "norm_class":he.NormalizeABNode,} #he.cos_exp_I_3D_F, smartD, he.cos_exp_I_smart2D_F, cos_exp_mix1_F
#cos_exp_mix8block25n20n15c_F, he.cos_exp_mix8_F,


network = HeuristicPaperNetwork = copy.deepcopy(SFANetwork1L)
network.L0.pca_node_class = mdp.nodes.GSFANode
network.L0.pca_out_dim = 60
network.exp_funcs = [identity]
network.L0.sfa_node_class = mdp.nodes.GSFANode
network.L0.sfa_out_dim = 60 # *3

####################################################################
######        2-Layer Linear SFA NETWORK TUBE           ############
####################################################################  
network = SFANetwork2T = system_parameters.ParamsNetwork()
network.name = "SFA 2 Layer Linear Network (Tube)"
network.L0 = copy.deepcopy(pSFAOneLayer)
network.L0.sfa_out_dim = 49
network.L1 = copy.deepcopy(pSFAOneLayer)
network.L1.sfa_out_dim = 49
network.L2 = None
network.L3 = None
network.L4 = None
network.layers = [network.L0, network.L1]

####################################################################
######        3-Layer Linear SFA NETWORK TUBE           ############
####################################################################  
network = SFANetwork3T = system_parameters.ParamsNetwork()
network.name = "SFA 2 Layer Linear Network (Tube)"
network.L0 = copy.deepcopy(pSFAOneLayer)
network.L1 = copy.deepcopy(pSFAOneLayer)
network.L2 = copy.deepcopy(pSFAOneLayer)
network.L3 = None
network.L4 = None
network.layers = [network.L0, network.L1, network.L2]

####  NetworkSetPCASFAExpo

####################################################################
######        One-Layer NON-Linear SFA NETWORK          ############
####################################################################  
#SFANetwork1L.layers[0].pca_node_class = mdp.nodes.GSFANode
#unsigned_08expo, pair_prodsigmoid_04_adj2_ex, unsigned_2_08expo, sel_exp(42, unsigned_08expo)
u08expoNetwork1L = NetworkSetExpFuncs([identity, sel_exp(42, unsigned_2_08expo), ], copy.deepcopy(SFANetwork1L))
#W
u08expoNetwork1L.layers[0].pca_node_class = mdp.nodes.PCANode
u08expoNetwork1L.layers[0].pca_out_dim = 500/3 #49
u08expoNetwork1L.layers[0].ord_node_class = None #mdp.nodes.GSFANode
u08expoNetwork1L.layers[0].ord_args = {"output_dim": 100}
u08expoNetwork1L.layers[0].sfa_out_dim = 30 #49

#W for 1L network
##u08expoNetwork1L.layers[0].pca_node_class = None
##u08expoNetwork1L.layers[0].ord_node_class = None
##u08expoNetwork1L.layers[0].sfa_out_dim = 49


####################################################################
######        Two-Layer NON-Linear SFA NETWORK TUBE     ############
####################################################################  
#SFANetwork1L.layers[0].pca_node_class = mdp.nodes.GSFANode
u08expoNetwork2T = NetworkSetExpFuncs([identity, unsigned_08expo], copy.deepcopy(SFANetwork2T))
#u08expoNetwork2T.layers[0].pca_node_class = mdp.nodes.GSFANode
u08expoNetwork2T.layers[0].pca_node_class = None
u08expoNetwork2T.layers[0].ord_node_class = mdp.nodes.GSFANode
u08expoNetwork2T.layers[0].ord_args = {"output_dim": 49}
u08expoNetwork2T.layers[1].pca_node_class = None
u08expoNetwork2T.layers[1].ord_node_class = None
u08expoNetwork2T.layers[1].sfa_node_class = mdp.nodes.GSFANode
u08expoNetwork2T.layers[1].sfa_out_dim = 49

####################################################################
######        One-Layer Quadratic SFA NETWORK          ############
####################################################################  
quadraticNetwork1L = NetworkSetExpFuncs([identity, pair_prod_ex], copy.deepcopy(SFANetwork1L)) #QT? CT?
quadraticNetwork1L.layers[0].pca_node_class = mdp.nodes.GSFANode
quadraticNetwork1L.layers[0].pca_out_dim = 16


#### 40, 65, 26, 35, 40
#### 60*3=180, 150, 50*3, 55*2 
##GTSRBNetwork = copy.deepcopy(SFANetwork3T)
##GTSRBNetwork.L0.pca_node_class = mdp.nodes.PCANode
##GTSRBNetwork.L0.pca_out_dim = 60 #40*3=120, 32x32=1024
##GTSRBNetwork.L0.ord_node_class = mdp.nodes.GSFANode
##GTSRBNetwork.L0.ord_args = {"output_dim": 150} #65....75/3 = 25, This number of dimensions are not expanded!
##GTSRBNetwork.L0.exp_funcs = [identity, sel_exp(42, unsigned_08expo)] #pair_prodsigmoid_04_adj2_ex
##GTSRBNetwork.L0.sfa_node_class = mdp.nodes.GSFANode     
##GTSRBNetwork.L0.sfa_out_dim = 50 # 17*3 = 51  26*3 = 78
##
##GTSRBNetwork.L1.exp_funcs = [identity, sel_exp(42, unsigned_08expo)] #pair_prodsigmoid_04_adj2_ex
##GTSRBNetwork.L1.sfa_node_class = mdp.nodes.GSFANode     
##GTSRBNetwork.L1.sfa_out_dim = 55 # 35 * 2 = 70
##
##GTSRBNetwork.L2.exp_funcs = [identity, sel_exp(42, unsigned_08expo)] #pair_prodsigmoid_04_adj2_ex
##GTSRBNetwork.L2.sfa_node_class = mdp.nodes.GSFANode     
##GTSRBNetwork.L2.sfa_out_dim = 40 # 40 *1.5 = 60

GTSRBNetwork = copy.deepcopy(SFANetwork1L)
GTSRBNetwork.L0.pca_node_class = mdp.nodes.PCANode
GTSRBNetwork.L0.pca_args = {}

#GTSRBNetwork.L0.pca_node_class = None
#W 150
GTSRBNetwork.L0.pca_out_dim = 200 #627 # int(1568/2.5) #300/3 # 200 #WW 50  #32x32x3=1024x3 
#GTSRBNetwork.L0.ord_node_class = mdp.nodes.GSFANode
#GTSRBNetwork.L0.ord_args = {"output_dim": 120} #75/3 = 25, This number of dimensions are not expanded! sel_exp(42, unsigned_08expo)
#GTSRBNetwork.L0.exp_funcs = [identity, unsigned_08expo] #pair_prodsigmoid_04_adj2_ex, unsigned_08expo, unsigned_2_08expo
#GTSRBNetwork.L0.exp_funcs = [identity, unsigned_08expo, numpy.sign, pair_smax25_mix1_ex, pair_smax50_mix1_ex, pair_smax_mix1_ex], sel14_MaxE, pair_max_mix1_ex]

GTSRBNetwork.L0.ord_node_class = mdp.nodes.HeadNode
GTSRBNetwork.L0.ord_args = {"output_dim":120}

GTSRBNetwork.L0.exp_funcs = [identity, QT] #, QT, maximum_99mix2_s08_ex, sel80_QT, QT] #maximum_99mix2_s08_ex, sel80_QT
#maximum_75mix2_ex, media50_adj2_ex, pair_max90_mix1_ex, modulation50_adj1_08_ex, unsigned_08expo, sel90_unsigned_08expo
#GTSRBNetwork.L0.exp_funcs = [identity,]
#For SFA features on img: unsigned_08expo, for final system img+hog: unsigned_08expo also

GTSRBNetwork.L0.sfa_node_class = mdp.nodes.GSFANode     #SFANode
GTSRBNetwork.L0.sfa_args = {}
#GTSRBNetwork.L0.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":expansion,"max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.999} 
GTSRBNetwork.L0.sfa_out_dim = 75 # WW 26*3 # 17*3 = 51 ## FOR RGB 26, for L/HOG/SFA
GTSRBNetwork.layers=[GTSRBNetwork.L0]

Q_N_k1_d2_L = Q_N_L(k=1.0, d=2.0)


##GTSRBNetwork = copy.deepcopy(SFANetwork2T)
##GTSRBNetwork.L0.pca_node_class = mdp.nodes.GSFANode #SFAPCANode
##GTSRBNetwork.L0.pca_out_dim = 150 # 40 #32x32=1024
###GTSRBNetwork.L0.ord_node_class = mdp.nodes.SFAPCANode
###GTSRBNetwork.L0.ord_args = {"output_dim": 65} #75/3 = 25, This number of dimensions are not expanded!
##GTSRBNetwork.L0.exp_funcs = [identity, unsigned_08expo] #sel_exp(42, unsigned_08expo)] #pair_prodsigmoid_04_adj2_ex
##GTSRBNetwork.L0.sfa_node_class = mdp.nodes.GSFANode     #SFAPCANode
##GTSRBNetwork.L0.sfa_out_dim = 10 #200 # 17*3 = 51 
##
##
##GTSRBNetwork.L1.exp_funcs =  [identity, unsigned_08expo] # Q_N_k1_d2_L # [identity, unsigned_08expo] # sel_exp(42, unsigned_08expo)] #pair_prodsigmoid_04_adj2_ex
##GTSRBNetwork.L1.pca_node_class = None
##GTSRBNetwork.L1.sfa_node_class = mdp.nodes.GSFANode     #SFAPCANode
##GTSRBNetwork.L1.sfa_out_dim = 50


##GTSRBNetwork.L2.exp_funcs = [identity, unsigned_08expo] # sel_exp(42, unsigned_08expo)] #pair_prodsigmoid_04_adj2_ex
##GTSRBNetwork.L2.pca_node_class = None
##GTSRBNetwork.L2.sfa_node_class = mdp.nodes.SFAPCANode     
##GTSRBNetwork.L2.sfa_out_dim = 200

#GTSRBNetwork = copy.deepcopy(SFANetwork2T)
#GTSRBNetwork.L0.pca_node_class = mdp.nodes.PCANode
#GTSRBNetwork.L0.pca_out_dim = 75 #32x32=1024
#GTSRBNetwork.L0.ord_node_class = mdp.nodes.GSFANode
#GTSRBNetwork.L0.ord_args = {"output_dim": 50} #75/3 = 25, This number of dimensions are not expanded!
#GTSRBNetwork.L0.exp_funcs = [identity, unsigned_08expo] #pair_prodsigmoid_04_adj2_ex
#GTSRBNetwork.L0.sfa_node_class = mdp.nodes.GSFANode     
#GTSRBNetwork.L0.sfa_out_dim = 20
#
##GTSRBNetwork.L0.ord_node_class = mdp.nodes.GSFANode
##GTSRBNetwork.L0.ord_args = {"output_dim": 45} #75/3 = 25, This number of dimensions are not expanded!
#GTSRBNetwork.L1.exp_funcs = [identity, unsigned_08expo] #pair_prodsigmoid_04_adj2_ex
#GTSRBNetwork.L1.sfa_node_class = mdp.nodes.GSFANode     
#GTSRBNetwork.L1.sfa_out_dim = 25
#
#print u08expoNetwork1L
#print u08expoNetwork1L.layers
#print u08expoNetwork1L.layers[0]
#print u08expoNetwork1L.layers[0].pca_node_class

##print "*******************************************************************"
##print "*****   Creating One-Layer Non-Linear SFA Network   ***************"
##print "*******************************************************************"
##print "******** Setting Layer L0 Parameters          *********************"
##layer = pSFAOneNLayer = system_parameters.ParamsSFASuperNode()
##layer.name = "One-Node SFA NL Layer"
##layer.pca_node_class = None
##layer.exp_funcs = [identity,]
##layer.red_node_class = None
##layer.sfa_node_class = mdp.nodes.GSFANode
##layer.sfa_args = {}
##layer.sfa_out_dim = None
##
######################################################################
########        One-Layer Linear SFA NETWORK              ############
######################################################################  
##network = SFANetwork1L = system_parameters.ParamsNetwork()
##network.name = "SFA 1 Layer Linear Network"
##network.L0 = pSFAOneLayer
##network.L1 = None
##network.L2 = None
##network.L3 = None
##network.L4 = None
##network.layers = [network.L0]

print "*******************************************************************"
print "********     Creating 2L Network for MNIST   ******************"
print "*******************************************************************"
print "******** Setting Layer L0 Parameters          *********************"
pSFALayerL0 = system_parameters.ParamsSFALayer()
pSFALayerL0.name = "Homogeneous Linear Layer L0 6x6 2 Nodes"
pSFALayerL0.x_field_channels=28
pSFALayerL0.y_field_channels=28
pSFALayerL0.x_field_spacing=28
pSFALayerL0.y_field_spacing=28
#pSFALayerL0.in_channel_dim=1

pSFALayerL0.pca_node_class = mdp.nodes.PCANode
pSFALayerL0.pca_out_dim = 120 #95
#pSFALayerL0.pca_args = {"block_size": block_size}
#pSFALayerL0.pca_args = {"block_size": -1, "train_mode": -1}
pSFALayerL0.pca_args = {}

pSFALayerL0.ord_node_class = mdp.nodes.HeadNode
pSFALayerL0.pca_out_dim = 120

pSFALayerL0.exp_funcs = [identity,]
#pSFALayerL0.exp_funcs = "RandomSigmoids" #[identity,]
#pSFALayerL0.exp_args = {output_dim:60} 

pSFALayerL0.red_node_class = None
pSFALayerL0.red_out_dim = 0
pSFALayerL0.red_args = {}

pSFALayerL0.sfa_node_class = mdp.nodes.iGSFANode #mdp.nodes.GSFANode
pSFALayerL0.sfa_out_dim = 60
pSFALayerL0.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":"RandomSigmoids", "expansion_starting_point":None, "max_lenght_slow_part":None,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":4.99999, "expansion_output_dim":300} 

#pSFALayerL0.sfa_args = {"block_size": -1, "train_mode": -1}

pSFALayerL0.cloneLayer = False
pSFALayerL0.name = comp_layer_name(pSFALayerL0.cloneLayer, pSFALayerL0.exp_funcs, pSFALayerL0.x_field_channels, pSFALayerL0.y_field_channels, pSFALayerL0.pca_out_dim, pSFALayerL0.sfa_out_dim)
system_parameters.test_object_contents(pSFALayerL0)


pSFALayerL1 = system_parameters.ParamsSFASuperNode()
pSFALayerL1.name = "SFA Linear Super Node L3  all =>  9"
#pSFALayerL1.in_channel_dim = pSFALayerL2.sfa_out_dim
#pca_node_L3 = mdp.nodes.WhiteningNode(output_dim=pca_out_dim_L3) 
pSFALayerL1.pca_node_class = None
pSFALayerL1.ord_node_class = None

pSFALayerL1.exp_funcs = [identity,]
#pSFALayerL1.exp_args = {"output_dim":60}

pSFALayerL1.red_node_class = None

pSFALayerL1.sfa_node_class = mdp.nodes.iGSFANode #mdp.nodes.GSFANode
pSFALayerL1.sfa_out_dim = 9
expansion_output_dim = int(tuning_parameter)
pSFALayerL1.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":"RandomSigmoids", "expansion_starting_point":None, "max_lenght_slow_part":None,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":4.99999, "expansion_output_dim":expansion_output_dim} 
pSFALayerL1.cloneLayer = False
pSFALayerL1.name = comp_supernode_name(pSFALayerL1.exp_funcs, pSFALayerL1.pca_out_dim, pSFALayerL1.sfa_out_dim)
system_parameters.test_object_contents(pSFALayerL1)

network = SFANetworkMNIST2L = system_parameters.ParamsNetwork()
network.name = "2L network 6x6 pixels (4x4=16 Nodes) for MNIST"
network.L0 = copy.deepcopy(pSFALayerL0)
network.L1 = copy.deepcopy(pSFALayerL1)
network.L2 = None
network.L3 = None
network.L4 = None
network.layers = [network.L0]



print "*******************************************************************"
print "*****   Creating 7L MMNIST Network  MNISTNetwork_24x24_7L *********"
print "*******************************************************************"
print "******** Setting Layer L0 Parameters          *********************"
pSFALayerL0 = system_parameters.ParamsSFALayer()
pSFALayerL0.name = "Homogeneous Linear Layer L0 3x3 8x8"
pSFALayerL0.x_field_channels=3
pSFALayerL0.y_field_channels=3
pSFALayerL0.x_field_spacing=3
pSFALayerL0.y_field_spacing=3
#pSFALayerL0.in_channel_dim=1

pSFALayerL0.pca_node_class = mdp.nodes.PCANode
pSFALayerL0.pca_out_dim = 9
pSFALayerL0.pca_args = {}

#pSFALayerL0.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class
pSFALayerL0.exp_funcs = [identity,]

pSFALayerL0.red_node_class = None
pSFALayerL0.red_out_dim = 0
pSFALayerL0.red_args = {}

pSFALayerL0.sfa_node_class = mdp.nodes.iGSFANode #mdp.nodes.GSFANode
pSFALayerL0.sfa_out_dim = 16
pSFALayerL0.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.9999} 

#pSFALayerL0.sfa_args = {"block_size": -1, "train_mode": -1}
pSFALayerL0.cloneLayer = False
pSFALayerL0.name = comp_layer_name(pSFALayerL0.cloneLayer, pSFALayerL0.exp_funcs, pSFALayerL0.x_field_channels, pSFALayerL0.y_field_channels, pSFALayerL0.pca_out_dim, pSFALayerL0.sfa_out_dim)
system_parameters.test_object_contents(pSFALayerL0)

pSFALayerL2H = system_parameters.ParamsSFALayer()
pSFALayerL2H.name = "Homogeneous Linear Layer LH 2x1 4x8 Nodes"
pSFALayerL2H.x_field_channels=2
pSFALayerL2H.y_field_channels=1
pSFALayerL2H.x_field_spacing=2
pSFALayerL2H.y_field_spacing=1

pSFALayerL2H.pca_node_class = None

pSFALayerL2H.exp_funcs = [identity,]

pSFALayerL2H.red_node_class = None
pSFALayerL2H.red_out_dim = 0
pSFALayerL2H.red_args = {}

pSFALayerL2H.sfa_node_class = mdp.nodes.iGSFANode #mdp.nodes.GSFANode
pSFALayerL2H.sfa_out_dim = 16
pSFALayerL2H.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 

#pSFALayerL0.sfa_args = {"block_size": -1, "train_mode": -1}
pSFALayerL2H.cloneLayer = False
pSFALayerL2H.name = comp_layer_name(pSFALayerL2H.cloneLayer, pSFALayerL2H.exp_funcs, pSFALayerL2H.x_field_channels, pSFALayerL2H.y_field_channels, pSFALayerL2H.pca_out_dim, pSFALayerL2H.sfa_out_dim)
system_parameters.test_object_contents(pSFALayerL2H)


pSFALayerL2V = system_parameters.ParamsSFALayer()
pSFALayerL2V.name = "Homogeneous Linear Layer LH 2x1 4x8 Nodes"
pSFALayerL2V.x_field_channels=1
pSFALayerL2V.y_field_channels=2
pSFALayerL2V.x_field_spacing=1
pSFALayerL2V.y_field_spacing=2

pSFALayerL2V.pca_node_class = None

pSFALayerL2V.exp_funcs = [identity,]

pSFALayerL2V.red_node_class = None
pSFALayerL2V.red_out_dim = 0
pSFALayerL2V.red_args = {}

pSFALayerL2V.sfa_node_class = mdp.nodes.iGSFANode #mdp.nodes.GSFANode
pSFALayerL2V.sfa_out_dim = 16
pSFALayerL2V.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 

#pSFALayerL2V.sfa_args = {"block_size": -1, "train_mode": -1}
pSFALayerL2V.cloneLayer = False
pSFALayerL2V.name = comp_layer_name(pSFALayerL2V.cloneLayer, pSFALayerL2V.exp_funcs, pSFALayerL2V.x_field_channels, pSFALayerL2V.y_field_channels, pSFALayerL2V.pca_out_dim, pSFALayerL2V.sfa_out_dim)
system_parameters.test_object_contents(pSFALayerL2V)


pSFALayerL1H = copy.deepcopy(pSFALayerL2H)
pSFALayerL1V = copy.deepcopy(pSFALayerL2V)
pSFALayerL2H = copy.deepcopy(pSFALayerL2H)
pSFALayerL2V = copy.deepcopy(pSFALayerL2V)
pSFALayerL3H = copy.deepcopy(pSFALayerL2H)
pSFALayerL3V = copy.deepcopy(pSFALayerL2V)

pSFALayerL0.sfa_out_dim = 12 # 9 + 4 
pSFALayerL1H.sfa_out_dim = 28 # 2*9 + 10
pSFALayerL1V.sfa_out_dim = 50 # 2*9 + 28
pSFALayerL2H.sfa_out_dim = 75 # 2*9 + 25
pSFALayerL2V.sfa_out_dim = 125 # 2*9 + 30
pSFALayerL3H.sfa_out_dim = 160 # 2*9 + 35
pSFALayerL3V.sfa_out_dim = 329 # 2*9 + 40
pSFALayerL3V.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo], "expansion_starting_point":None, "max_lenght_slow_part":10, "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999, "expansion_output_dim":None} 
#THIS CODE IMPLEMENTS A NEW TYPE OF EXPANSION
#pSFALayerL3V.sfa_args = {"pre_expansion_node_class":none, "expansion_funcs":"RandomSigmoids", "expansion_starting_point":"08Exp", "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999, "expansion_output_dim":800} 


network = HiGSFA_MNISTNetwork_24x24_7L = system_parameters.ParamsNetwork()
network.name = "HiGSFA MNIST Network 7L 24x24"
network.L0 = copy.deepcopy(pSFALayerL0)
network.L1 = copy.deepcopy(pSFALayerL1H)
network.L2 = copy.deepcopy(pSFALayerL1V)
network.L3 = copy.deepcopy(pSFALayerL2H)
network.L4 = copy.deepcopy(pSFALayerL2V)
network.L5 = copy.deepcopy(pSFALayerL3H)
network.L6 = copy.deepcopy(pSFALayerL3V)
network.layers = [network.L0, network.L1, network.L2, network.L3, network.L4, network.L5, network.L6]


#Experimental MNIST network with no overlap, and weight sharing in the first layer
network = HiGSFA_MNISTNetwork_24x24_WeightSharing_7L = copy.deepcopy(HiGSFA_MNISTNetwork_24x24_7L)
L0 = network.layers[0]
L0.cloneLayer = True
L0.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":9, "reconstruct_with_sfa":False, "offsetting_mode":None, "max_preserved_sfa":9} 
L0.sfa_out_dim=17
L1 = network.layers[1]
L1.sfa_out_dim=40

print "*******************************************************************"
print "*****   Creating 7L MMNIST Network  MNISTNetwork_24x24_7L_B *********"
print "*******************************************************************"

def signed_03expo(x):
    return signed_expo(x, 0.3)

def signed_08expo(x):
    return signed_expo(x, 0.8)


def unsigned_08expo_p075(x):
    return unsigned_08expo(x+0.75)

def unsigned_08expo_m075(x):
    return unsigned_08expo(x-0.75)

def unsigned_08expo_p15(x):
    return unsigned_08expo(x+1.5)

def unsigned_08expo_m15(x):
    return unsigned_08expo(x-1.5)

def unsigned_08expo_75(x):
    return unsigned_08expo(x[:,0:75])

def unsigned_08expo_115(x):
    return unsigned_08expo(x[:,0:115])

def QT_50_AP03(x):
    return QT_AP(x[:,0:50], d=0.3)

def QT_50_AP06(x):
    return QT_AP(x[:,0:50], d=0.6)

def QT_50_AP08(x):
    return QT_AP(x[:,0:50], d=0.8)

def QT_60_AP08(x):
    return QT_AP(x[:,0:60], d=0.8)

def QT_70_AP08(x):
    return QT_AP(x[:,0:70], d=0.8)

def QT_90_AP08(x):
    return QT_AP(x[:,0:90], d=0.8)

def QT_2Split_25(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT(x[:,0:25]), QT(x[:,s:s+25])), axis=1)

def QT_2Split_15(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT(x[:,0:15]), QT(x[:,s:s+15])), axis=1)

def QT_2Split_35(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT(x[:,0:35]), QT(x[:,s:s+35])), axis=1)

def QT_2Split_40(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT(x[:,0:40]), QT(x[:,s:s+40])), axis=1)

def QT_2Split_45(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT(x[:,0:45]), QT(x[:,s:s+45])), axis=1)

def QT_2Split_50(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT(x[:,0:50]), QT(x[:,s:s+50])), axis=1)

def QT_2Split_55(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT(x[:,0:55]), QT(x[:,s:s+55])), axis=1)

def QT_2Split_60(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT(x[:,0:60]), QT(x[:,s:s+60])), axis=1)

def QT_2Split_65(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT(x[:,0:65]), QT(x[:,s:s+65])), axis=1)

def QT_2Split_70(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT(x[:,0:70]), QT(x[:,s:s+70])), axis=1)

def QT_2Split_15_AP08(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT_AP(x[:,0:15], d=0.8), QT_AP(x[:,s:s+15],  d=0.8)), axis=1)

def QT_2Split_50_AP02(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT_AP(x[:,0:50], d=0.2), QT_AP(x[:,s:s+50],  d=0.2)), axis=1)

def QT_2Split_50_AP03(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT_AP(x[:,0:50], d=0.3), QT_AP(x[:,s:s+50],  d=0.3)), axis=1)

def QT_2Split_50_AP06(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT_AP(x[:,0:50], d=0.6), QT_AP(x[:,s:s+50],  d=0.6)), axis=1)

def QT_2Split_50_AP08(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT_AP(x[:,0:50], d=0.8), QT_AP(x[:,s:s+50],  d=0.8)), axis=1)

def QT_2Split_60_AP08(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT_AP(x[:,0:60], d=0.8), QT_AP(x[:,s:s+60],  d=0.8)), axis=1)

def QT_2Split_70_AP08(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT_AP(x[:,0:70], d=0.8), QT_AP(x[:,s:s+70],  d=0.8)), axis=1)

def QT_2Split_55_AP08(x):
    s = x.shape[1]/2
    return numpy.concatenate((QT_AP(x[:,0:55], d=0.8), QT_AP(x[:,s:s+55],  d=0.8)), axis=1)

def QT_3Split_10(x):
    s = x.shape[1]/3
    return numpy.concatenate((QT(x[:,0:10]), QT(x[:,s:s+10]), QT(x[:,2*s:2*s+10])), axis=1)

def QT_3Split_15(x):
    s = x.shape[1]/3
    return numpy.concatenate((QT(x[:,0:15]), QT(x[:,s:s+15]), QT(x[:,2*s:2*s+15])), axis=1)

def QT_3Split_15_AP02(x):
    s = x.shape[1]/3
    return numpy.concatenate((QT_AP(x[:,0:15], d=0.2), QT_AP(x[:,s:s+15], d=0.2), QT_AP(x[:,2*s:2*s+15], d=0.2)), axis=1)

def QT_3Split_15_AP03(x):
    s = x.shape[1]/3
    return numpy.concatenate((QT_AP(x[:,0:15], d=0.3), QT_AP(x[:,s:s+15], d=0.3), QT_AP(x[:,2*s:2*s+15], d=0.3)), axis=1)

def QT_3Split_15_AP06(x):
    s = x.shape[1]/3
    return numpy.concatenate((QT_AP(x[:,0:15], d=0.6), QT_AP(x[:,s:s+15], d=0.6), QT_AP(x[:,2*s:2*s+15], d=0.6)), axis=1)

def QT_3Split_15_AP08(x):
    s = x.shape[1]/3
    return numpy.concatenate((QT_AP(x[:,0:15], d=0.8), QT_AP(x[:,s:s+15], d=0.8), QT_AP(x[:,2*s:2*s+15], d=0.8)), axis=1)


def QT_3Split_20(x):
    s = x.shape[1]/3
    return numpy.concatenate((QT(x[:,0:20]), QT(x[:,s:s+20]), QT(x[:,2*s:2*s+20])), axis=1)


def QT_3Split_25(x):
    s = x.shape[1]/3
    return numpy.concatenate((QT(x[:,0:25]), QT(x[:,s:s+25]), QT(x[:,2*s:2*s+25])), axis=1)

def QT_3Split_35(x):
    s = x.shape[1]/3
    return numpy.concatenate((QT(x[:,0:35]), QT(x[:,s:s+35]), QT(x[:,2*s:2*s+35])), axis=1)

def QT_3Split_35_AP08(x):
    s = x.shape[1]/3
    return numpy.concatenate((QT_AP(x[:,0:35], d=0.8), QT_AP(x[:,s:s+35], d=0.8), QT_AP(x[:,2*s:2*s+35], d=0.8)), axis=1)


def QT_3Split_45(x):
    s = x.shape[1]/3
    return numpy.concatenate((QT(x[:,0:45]), QT(x[:,s:s+45]), QT(x[:,2*s:2*s+45])), axis=1)

def QT_3Split_50(x):
    s = x.shape[1]/3
    return numpy.concatenate((QT(x[:,0:50]), QT(x[:,s:s+50]), QT(x[:,2*s:2*s+50])), axis=1)

def CT_20_AP08(x):
    return CT_AP(x[:,0:20], d=0.8)

def CT_25_AP08(x):
    return CT_AP(x[:,0:25], d=0.8)

def CT_30_AP03(x):
    return CT_AP(x[:,0:30], d=0.3)

def CT_30_AP06(x):
    return CT_AP(x[:,0:30], d=0.6)

def CT_30_AP08(x):
    return CT_AP(x[:,0:30], d=0.8)

def CT_35_AP03(x):
    return CT_AP(x[:,0:35], d=0.3)

def CT_35_AP08(x):
    return CT_AP(x[:,0:35], d=0.8)

def CT_40_AP08(x):
    return CT_AP(x[:,0:40], d=0.8)

def CT_45_AP08(x):
    return CT_AP(x[:,0:45], d=0.8)

def CT_50_AP08(x):
    return CT_AP(x[:,0:50], d=0.8)

def CT_55_AP08(x):
    return CT_AP(x[:,0:55], d=0.8)

def CT_9_39(x):
    return CT(x[:,9:39])

def CT_2Split_20(x):
    s = x.shape[1]/2
    return numpy.concatenate((CT(x[:,0:20]), CT(x[:,s:s+20])), axis=1)

def CT_2Split_25(x):
    s = x.shape[1]/2
    return numpy.concatenate((CT(x[:,0:25]), CT(x[:,s:s+25])), axis=1)

def CT_2Split_30(x):
    s = x.shape[1]/2
    return numpy.concatenate((CT(x[:,0:30]), CT(x[:,s:s+30])), axis=1)

def CT_3Split_15(x):
    s = x.shape[1]/3
    return numpy.concatenate((CT(x[:,0:15]), CT(x[:,s:s+15]), CT(x[:,2*s:2*s+15])), axis=1)

def CT_3Split_20(x):
    s = x.shape[1]/3
    return numpy.concatenate((CT(x[:,0:20]), CT(x[:,s:s+20]), CT(x[:,2*s:2*s+20])), axis=1)


print "******** Setting Layer L0 Parameters          *********************"
pSFALayerL0_4x4 = copy.deepcopy(pSFALayerL0) #L1
pSFALayerL0_4x4.name = "Homogeneous Linear Layer L0 S=4x4 D=2x2"
pSFALayerL0_4x4.x_field_channels=4 #4 for 24x24 and 28x28, 5 for 29x29
pSFALayerL0_4x4.y_field_channels=4
pSFALayerL0_4x4.x_field_spacing=2 #2 for 24x24and 28x28, 3 for 29x29
pSFALayerL0_4x4.y_field_spacing=2
pSFALayerL0_4x4.pca_out_dim = 13 #12 for 24x24and 28x28, 20 for 29x29
pSFALayerL0_4x4.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, ], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 

pSFALayerL1H_S3_D2 = copy.deepcopy(pSFALayerL1H) #L2
pSFALayerL1H_S3_D2.name = "Homogeneous Linear Layer L1H S=3x1 D=2x1"
pSFALayerL1H_S3_D2.x_field_channels=3
pSFALayerL1H_S3_D2.y_field_channels=1
pSFALayerL1H_S3_D2.x_field_spacing=2
pSFALayerL1H_S3_D2.y_field_spacing=1
#pSFALayerL1H_S3_D2.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, QT_3Split_15], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
pSFALayerL1H_S3_D2.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, ], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 

pSFALayerL2H_S3_D2 = copy.deepcopy(pSFALayerL1H) #L4
pSFALayerL2H_S3_D2.name = "Homogeneous Linear Layer L2H S=3x1 D=2x1"
pSFALayerL2H_S3_D2.x_field_channels=2 #3 for 24x24 and 29x29, 2 for 28x28
pSFALayerL2H_S3_D2.y_field_channels=1
pSFALayerL2H_S3_D2.x_field_spacing=2 #2 for 24x24, 1 for 29x29
pSFALayerL2H_S3_D2.y_field_spacing=1
#pSFALayerL2H_S3_D2.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, QT_3Split_25, CT_3Split_20], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
#pSFALayerL2H_S3_D2.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo,  QT_2Split_15_AP08], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
pSFALayerL2H_S3_D2.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, ], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 

pSFALayerL3H_S2_D1 = copy.deepcopy(pSFALayerL1H) #L6
pSFALayerL3H_S2_D1.name = "Homogeneous Linear Layer L3H S=2x1 D=1x1"
pSFALayerL3H_S2_D1.x_field_channels=3 #2 for 24x24 and 29x29, 3 for 28x28
pSFALayerL3H_S2_D1.y_field_channels=1
pSFALayerL3H_S2_D1.x_field_spacing=1
pSFALayerL3H_S2_D1.y_field_spacing=1
#sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, QT_2Split_25, CT_2Split_20], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
#pSFALayerL3H_S2_D1.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, QT_2Split_50], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
#pSFALayerL3H_S2_D1.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo,  QT_3Split_35_AP08], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
pSFALayerL3H_S2_D1.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, ], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 

#***************************************************************************
pSFALayerL1V_S3_D2 = copy.deepcopy(pSFALayerL1H) #L3 
pSFALayerL1V_S3_D2.name = "Homogeneous Linear Layer L1V S=1x3 D=1x2"
pSFALayerL1V_S3_D2.x_field_channels=1
pSFALayerL1V_S3_D2.y_field_channels=3
pSFALayerL1V_S3_D2.x_field_spacing=1
pSFALayerL1V_S3_D2.y_field_spacing=2
#sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, QT_3Split_20, CT_3Split_15], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
pSFALayerL1V_S3_D2.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, ], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 

pSFALayerL2V_S3_D2 = copy.deepcopy(pSFALayerL1H) #L5
pSFALayerL2V_S3_D2.name = "Homogeneous Linear Layer L2V S=1x3 D=1x2"
pSFALayerL2V_S3_D2.x_field_channels=1
pSFALayerL2V_S3_D2.y_field_channels=2
pSFALayerL2V_S3_D2.x_field_spacing=1
pSFALayerL2V_S3_D2.y_field_spacing=2 #2 for 24x24, 1 for 29x29
#sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, QT_3Split_25, CT_3Split_20], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
pSFALayerL2V_S3_D2.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, ], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 

pSFALayerL3V_S2_D1 = copy.deepcopy(pSFALayerL1H) #L7
pSFALayerL3V_S2_D1.name = "Homogeneous Linear Layer L3V S=1x2 D=1x1"
pSFALayerL3V_S2_D1.x_field_channels=1
pSFALayerL3V_S2_D1.y_field_channels=3
pSFALayerL3V_S2_D1.x_field_spacing=1
pSFALayerL3V_S2_D1.y_field_spacing=1
#sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, QT_2Split_25, CT_2Split_20], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
#pSFALayerL3V_S2_D1.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, QT_2Split_35, CT_2Split_25], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
pSFALayerL3V_S2_D1.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, ], "max_lenght_slow_part":1,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 

pSFALayerL0_4x4.sfa_out_dim = 13 #Was 15 #Usually 16 L1 # 9 + 5 
pSFALayerL1H_S3_D2.sfa_out_dim = 20 #Was 28 #Usually 30 L2 # 2*9 + 10
pSFALayerL1V_S3_D2.sfa_out_dim = 50 #L3 # 2*9 + 28
pSFALayerL2H_S3_D2.sfa_out_dim = 70 #L4 #60 # 2*9 + 25
pSFALayerL2V_S3_D2.sfa_out_dim = 90 #L5 #70 # 2*9 + 30
pSFALayerL3H_S2_D1.sfa_out_dim = 120 #L6 #70 #44 #265 # 2*9 + 35
pSFALayerL3V_S2_D1.sfa_out_dim = 160 #L7 #130 #150 # 2*9 + 40
#pSFALayerL3V_S2_D1.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
#pSFALayerL3V_S2_D1.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":"RandomSigmoids", "expansion_starting_point":"08Exp", "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999, "expansion_output_dim":2000} 
#pSFALayerL3V_S2_D1.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, QT, CT], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
pSFALayerL0_4x4.sfa_args["max_preserved_sfa"]=4
pSFALayerL1H_S3_D2.sfa_args["max_preserved_sfa"]=4
pSFALayerL1V_S3_D2.sfa_args["max_preserved_sfa"]=4
pSFALayerL2H_S3_D2.sfa_args["max_preserved_sfa"]=4
pSFALayerL2V_S3_D2.sfa_args["max_preserved_sfa"]=4
pSFALayerL3H_S2_D1.sfa_args["max_preserved_sfa"]=4
pSFALayerL3V_S2_D1.sfa_args["max_preserved_sfa"]=9


pSFALayerSupernode = system_parameters.ParamsSFASuperNode() #L8
pSFALayerSupernode.name = "SFA Super Node Layer"
pSFALayerSupernode.pca_node_class = None
pSFALayerSupernode.ord_node_class = mdp.nodes.HeadNode
pSFALayerSupernode.ord_args = {"output_dim":115}
#pSFALayerSupernode.exp_funcs = [identity, unsigned_08expo, unsigned_08expo_p15, unsigned_08expo_m15, signed_08expo, QT_90_AP08, CT_30_AP08,] #signed_08expo
pSFALayerSupernode.exp_funcs = [identity, unsigned_08expo, signed_08expo, QT_90_AP08, CT_30_AP08,] #signed_08expo
#pSFALayerSupernode.exp_funcs = [identity, QT, CT]
#pSFALayerSupernode.red_node_class = None
pSFALayerSupernode.sfa_node_class = mdp.nodes.GSFANode
#pSFALayerSupernode.sfa_node_class = mdp.nodes.iGSFANode #mdp.nodes.GSFANode
#pSFALayerSupernode.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, QT_50, CT_30],                     "max_lenght_slow_part":1, ,  "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
#pSFALayerSupernode.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo_75],                     "max_lenght_slow_part":1, ,  "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
pSFALayerSupernode.sfa_out_dim = 80


########################################## New Improved Network for MNIST ############################
network = MNISTNetwork_24x24_7L_Overlap = system_parameters.ParamsNetwork()
network.name = "MNIST Network 7L 24x24_Overlap"
network.L0 = copy.deepcopy(pSFALayerL0_4x4)
network.L1 = copy.deepcopy(pSFALayerL1H_S3_D2)
network.L2 = copy.deepcopy(pSFALayerL1V_S3_D2)
network.L3 = copy.deepcopy(pSFALayerL2H_S3_D2)
network.L4 = copy.deepcopy(pSFALayerL2V_S3_D2)
network.L5 = copy.deepcopy(pSFALayerL3H_S2_D1)
network.L6 = copy.deepcopy(pSFALayerL3V_S2_D1)
network.L7 = copy.deepcopy(pSFALayerSupernode)
network.layers = [network.L0, network.L1, network.L2, network.L3, network.L4, network.L5, network.L6, network.L7]
#network.L1 = network.L2 = network.L3 = network.L4 = network.L5 = network.L6 = None
#network.layers = [network.L0, None, None, None, None, None, None]

network = MNISTNetwork_24x24_5L_OverlapNew = copy.deepcopy(MNISTNetwork_24x24_7L_Overlap)
network.L0 = system_parameters.ParamsSFALayer()
network.L0.x_field_channels=6
network.L0.y_field_channels=6
network.L0.x_field_spacing=3
network.L0.y_field_spacing=3
network.L0.pca_node_class = mdp.nodes.PCANode
network.L0.pca_args = {}
network.L0.sfa_node_class = mdp.nodes.iGSFANode #mdp.nodes.GSFANode
network.L0.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, s7QT], "max_lenght_slow_part":9,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
network.L0.cloneLayer = False

network.L1 = system_parameters.ParamsSFALayer()
network.L1.x_field_channels=3
network.L1.y_field_channels=1
network.L1.x_field_spacing=2
network.L1.y_field_spacing=1
network.L1.pca_node_class = None
network.L1.sfa_node_class = mdp.nodes.iGSFANode #mdp.nodes.GSFANode
network.L1.sfa_out_dim = 35
network.L1.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, ch3o3s3QT, ch3o0s3QT], "max_lenght_slow_part":9,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
network.L1.cloneLayer = False

network.L2 = system_parameters.ParamsSFALayer()
network.L2.x_field_channels=1
network.L2.y_field_channels=3
network.L2.x_field_spacing=1
network.L2.y_field_spacing=2
network.L2.pca_node_class = None
network.L2.sfa_node_class = mdp.nodes.iGSFANode #mdp.nodes.GSFANode
network.L2.sfa_out_dim = 45
network.L2.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, ch3o4s9QT, ch3o0s4QT], "max_lenght_slow_part":9,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 
network.L2.cloneLayer = False

network.L3 = copy.deepcopy(network.L1)
network.L3.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, ch3o6s16QT, ch3o0s4QT], "max_lenght_slow_part":9,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 

network.L4 = copy.deepcopy(network.L2)
network.L4.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, ch3o9s24QT, ch3o9s13CT], "max_lenght_slow_part":9,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} 


network.layers = [network.L0,network.L1,network.L2,network.L3,network.L4 ] #,network.L5,network.L6]
network.L0.pca_out_dim = 30
HiGSFANet_out_dims = [39, 50, 60, 75, 75] #= [30, 45, 60, 75, 75]

max_preserved_sfas = [3, 4, 6, 9, 9]
for i in range(len(network.layers)):
    network.layers[i].sfa_out_dim = HiGSFANet_out_dims[i]
    network.layers[i].sfa_args["max_preserved_sfa"] = max_preserved_sfas[i]

#Experimental MNIST network with overlap, and weight sharing in the first layer
network = MNISTNetwork_24x24_5L_WeightSharing_OverlapNew = copy.deepcopy(MNISTNetwork_24x24_5L_OverlapNew)
L0 = network.layers[0]
L0.cloneLayer = True
L0.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo, s7QT], "max_lenght_slow_part":9, "reconstruct_with_sfa":False, "offsetting_mode":None, "max_preserved_sfa":4} 

#Experimental MNIST network with overlap, and weight sharing in the first layer
network = MNISTNetwork_24x24_5L_WeightSharing_Basic_OverlapNew = copy.deepcopy(MNISTNetwork_24x24_5L_OverlapNew)
for layer in network.layers:
    layer.sfa_args["expansion_funcs"] = [identity, unsigned_08expo]

L0 = network.layers[0]
L0.cloneLayer = True
L0.sfa_args["reconstruct_with_sfa"] = False
L0.sfa_args["offsetting_mode"] = None
L0.sfa_args["max_preserved_sfa"]=9

L1 = network.layers[1]
L1.cloneLayer = True
L1.sfa_args["reconstruct_with_sfa"] = False
L1.sfa_args["offsetting_mode"] = None
L1.sfa_args["max_preserved_sfa"]=9
L1.sfa_args["expansion_funcs"] = [identity, unsigned_08expo, ]

L2 = network.layers[2]
#L2.cloneLayer = True
L2.sfa_args["reconstruct_with_sfa"] = False
L2.sfa_args["offsetting_mode"] = None #"data_dependent"
L2.sfa_args["max_preserved_sfa"]=9
L2.sfa_args["expansion_funcs"] = [identity, unsigned_08expo] #DEBUG, ch3o0s9QT, ch3o9s10QT]

L3 = network.layers[3]
#L3.sfa_args["reconstruct_with_sfa"] = False
#L3.sfa_args["offsetting_mode"] = None #"data_dependent"
L3.sfa_args["expansion_funcs"] = [identity, unsigned_08expo] #DEBUG, ch3o9s10QT] #ch3o0s20QT, unsigned_08expo,

L4 = network.layers[4]
#L4.sfa_args["reconstruct_with_sfa"] = False
#L4.sfa_args["offsetting_mode"] = "data_dependent" # None, "data_dependent", "sensitivity_based_pure"
L4.sfa_args["expansion_funcs"] = [identity, unsigned_08expo] #DEBUG ch3o9s32QT] #, ch3o9s13CT], unsigned_08expo,
#ch3o4s9QT, ch3o0s4QT
#ch3o6s16QT, ch3o0s4QT]
#ch3o9s24QT, ch3o9s13CT


print "*******************************************************************"
print "******** Creating Linear 4L SFA Network          ******************"
print "*******************************************************************"
print "******** Setting Layer L0 Parameters          *********************"
pSFALayerL0 = system_parameters.ParamsSFALayer()
pSFALayerL0.name = "Homogeneous Linear Layer L0 5x5 => 15"
pSFALayerL0.x_field_channels=5
pSFALayerL0.y_field_channels=5
pSFALayerL0.x_field_spacing=5
pSFALayerL0.y_field_spacing=5
#pSFALayerL0.in_channel_dim=1

pSFALayerL0.pca_node_class = None #mdp.nodes.GSFANode
pSFALayerL0.pca_out_dim = 16
#pSFALayerL0.pca_args = {"block_size": block_size}
pSFALayerL0.pca_args = {"block_size": -1, "train_mode": -1}

pSFALayerL0.ord_node_class = None #system_parameters.ParamsSFALayer.ord_node_class

pSFALayerL0.exp_funcs = [identity,]

pSFALayerL0.red_node_class = None #mdp.nodes.WhiteningNode
pSFALayerL0.red_out_dim = 0.9999999
pSFALayerL0.red_args = {}


pSFALayerL0.sfa_node_class = mdp.nodes.SFANode
pSFALayerL0.sfa_out_dim = 16
#pSFALayerL0.sfa_args = {"block_size": -1, "train_mode": -1}

pSFALayerL0.cloneLayer = False
pSFALayerL0.name = comp_layer_name(pSFALayerL0.cloneLayer, pSFALayerL0.exp_funcs, pSFALayerL0.x_field_channels, pSFALayerL0.y_field_channels, pSFALayerL0.pca_out_dim, pSFALayerL0.sfa_out_dim)
system_parameters.test_object_contents(pSFALayerL0)

print "******** Setting Layer L1 Parameters *********************"
pSFALayerL1 = system_parameters.ParamsSFALayer()
pSFALayerL1.name = "Homogeneous Linear Layer L1 3x3 => 30"
pSFALayerL1.x_field_channels=3
pSFALayerL1.y_field_channels=3
pSFALayerL1.x_field_spacing=3
pSFALayerL1.y_field_spacing=3
#pSFALayerL1.in_channel_dim = pSFALayerL0.sfa_out_dim

pSFALayerL1.pca_node_class = mdp.nodes.WhiteningNode
#pca_out_dim_L1 = 90
#pca_out_dim_L1 = sfa_out_dim_L0 x_field_channels_L1 * x_field_channels_L1 * 0.75 
pSFALayerL1.pca_out_dim = 125
#pSFALayerL1.pca_args = {"block_size": block_size}
pSFALayerL1.pca_args = {}

pSFALayerL1.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class

pSFALayerL1.exp_funcs = [identity,]

pSFALayerL1.red_node_class = mdp.nodes.WhiteningNode
pSFALayerL1.red_out_dim = 125
pSFALayerL1.red_args = {}

pSFALayerL1.sfa_node_class = mdp.nodes.GSFANode
#sfa_out_dim_L1 = 12
pSFALayerL1.sfa_out_dim = 30
#pSFALayerL1.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL1 = False
#WARNING, DEFAULT IS: pSFALayerL1.cloneLayer = True
pSFALayerL1.cloneLayer = False
pSFALayerL1.name = comp_layer_name(pSFALayerL1.cloneLayer, pSFALayerL1.exp_funcs, pSFALayerL1.x_field_channels, pSFALayerL1.y_field_channels, pSFALayerL1.pca_out_dim, pSFALayerL1.sfa_out_dim)
system_parameters.test_object_contents(pSFALayerL1)

print "******** Setting Layer L2 Parameters *********************"
pSFALayerL2 = system_parameters.ParamsSFALayer()
pSFALayerL2.name = "Inhomogeneous Linear Layer L2 3x3 => 40"
pSFALayerL2.x_field_channels=3
pSFALayerL2.y_field_channels=3
pSFALayerL2.x_field_spacing=3
pSFALayerL2.y_field_spacing=3
#pSFALayerL2.in_channel_dim = pSFALayerL1.sfa_out_dim

pSFALayerL2.pca_node_class = mdp.nodes.WhiteningNode
#pca_out_dim_L2 = 90
#pca_out_dim_L2 = sfa_out_dim_L1 x_field_channels_L2 * x_field_channels_L2 * 0.75 
pSFALayerL2.pca_out_dim = 200 #100
#pSFALayerL2.pca_args = {"block_size": block_size}
pSFALayerL2.pca_args = {}

pSFALayerL2.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class

pSFALayerL2.exp_funcs = [identity,]

pSFALayerL2.red_node_class = mdp.nodes.WhiteningNode
pSFALayerL2.red_out_dim = 200
pSFALayerL2.red_args = {}

pSFALayerL2.sfa_node_class = mdp.nodes.GSFANode
#sfa_out_dim_L2 = 12
pSFALayerL2.sfa_out_dim = 40
#pSFALayerL2.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL2 = False
pSFALayerL2.cloneLayer = False
pSFALayerL2.name = comp_layer_name(pSFALayerL2.cloneLayer, pSFALayerL2.exp_funcs, pSFALayerL2.x_field_channels, pSFALayerL2.y_field_channels, pSFALayerL2.pca_out_dim, pSFALayerL2.sfa_out_dim)
system_parameters.test_object_contents(pSFALayerL2)


print "******** Setting Layer L3 Parameters *********************"
pSFAL3 = system_parameters.ParamsSFASuperNode()
pSFAL3.name = "SFA Linear Super Node L3  all =>  300 => 40"
#pSFAL3.in_channel_dim = pSFALayerL2.sfa_out_dim

#pca_node_L3 = mdp.nodes.WhiteningNode(output_dim=pca_out_dim_L3) 
pSFAL3.pca_node_class = mdp.nodes.GSFANode
#pca_out_dim_L3 = 210
#pca_out_dim_L3 = 0.999
#WARNING!!! CHANGED PCA TO SFA
pSFAL3.pca_out_dim = 300
#pSFALayerL1.pca_args = {"block_size": block_size}
pSFAL3.pca_args = {"block_size": -1, "train_mode": -1}

pSFAL3.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class

#exp_funcs_L3 = [identity, pair_prod_ex, pair_prod_adj1_ex, pair_prod_adj2_ex, pair_prod_adj3_ex]
pSFAL3.exp_funcs = [identity,]
pSFAL3.inv_use_hint = True
pSFAL3.max_steady_factor=0.35
pSFAL3.delta_factor=0.6
pSFAL3.min_delta=0.0001

pSFAL3.red_node_class = mdp.nodes.WhiteningNode
pSFAL3.red_out_dim = 0.999999
pSFAL3.red_args = {}

pSFAL3.sfa_node_class = mdp.nodes.GSFANode
#sfa_out_dim_L1 = 12
pSFAL3.sfa_out_dim = 40
#pSFAL3.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL1 = False
pSFAL3.cloneLayer = False
pSFAL3.name = comp_supernode_name(pSFAL3.exp_funcs, pSFAL3.pca_out_dim, pSFAL3.sfa_out_dim)
system_parameters.test_object_contents(pSFAL3)


print "******** Setting Layer L4 Parameters *********************"
pSFAL4 = system_parameters.ParamsSFASuperNode()
pSFAL4.name = "SFA Linear Super Node L4  all => 40 => 40"
#pSFAL4.in_channel_dim = pSFAL3.sfa_out_dim

#pca_node_L3 = mdp.nodes.WhiteningNode(output_dim=pca_out_dim_L3) 
pSFAL4.pca_node_class = mdp.nodes.GSFANode
#pca_out_dim_L3 = 210
#pca_out_dim_L3 = 0.999
#WARNING!!! CHANGED PCA TO SFA
pSFAL4.pca_out_dim = 40
#pSFALayerL1.pca_args = {"block_size": block_size}
pSFAL4.pca_args = {"block_size": -1, "train_mode": -1}

pSFAL4.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class

#exp_funcs_L3 = [identity, pair_prod_ex, pair_prod_adj1_ex, pair_prod_adj2_ex, pair_prod_adj3_ex]
pSFAL4.exp_funcs = [identity,]
pSFAL4.inv_use_hint = True
pSFAL4.max_steady_factor=0.35
pSFAL4.delta_factor=0.6
pSFAL4.min_delta=0.0001

pSFAL4.red_node_class = mdp.nodes.WhiteningNode
pSFAL4.red_out_dim = 0.999999
pSFAL4.red_args = {}

pSFAL4.sfa_node_class = mdp.nodes.GSFANode
#sfa_out_dim_L1 = 12
pSFAL4.sfa_out_dim = 40
#pSFAL4.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL1 = False
pSFAL4.cloneLayer = False
pSFAL4.name = comp_supernode_name(pSFAL4.exp_funcs, pSFAL4.pca_out_dim, pSFAL4.sfa_out_dim)
system_parameters.test_object_contents(pSFAL4)


####################################################################
###########               LINEAR NETWORK                ############
####################################################################  
linearNetwork4L = system_parameters.ParamsNetwork()
linearNetwork4L.name = "Linear 4 Layer Network"
linearNetwork4L.L0 = pSFALayerL0
linearNetwork4L.L1 = pSFALayerL1
linearNetwork4L.L2 = pSFALayerL2
linearNetwork4L.L3 = pSFAL3
network = linearNetwork4L 
network.layers = [network.L0, network.L1, network.L2, network.L3]

linearNetwork5L = system_parameters.ParamsNetwork()
linearNetwork5L.name = "Linear 5 Layer Network"
linearNetwork5L.L0 = pSFALayerL0
linearNetwork5L.L1 = pSFALayerL1
linearNetwork5L.L2 = pSFALayerL2
linearNetwork5L.L3 = pSFAL3
linearNetwork5L.L4 = pSFAL4
network = linearNetwork5L 
network.layers = [network.L0, network.L1, network.L2, network.L3, network.L4]

u08expoNetwork4L = NetworkSetExpFuncs([identity, unsigned_08expo], copy.deepcopy(linearNetwork4L))
u08expoNetwork4L.L3.exp_funcs = [identity,]
u08expoNetwork4L.L1.exp_funcs = [identity,]
u08expoNetwork4L.L0.exp_funcs = [identity,]

#####################################################################
############    NON-LINEAR LAYERS                        ############
#####################################################################  
print "*******************************************************************"
print "******** Creating Non-Linear 4L SFA Network      ******************"
print "*******************************************************************"
print "******** Setting Layer NL0 Parameters          ********************"
pSFALayerNL0 = system_parameters.ParamsSFALayer()
pSFALayerNL0.name = "Homogeneous Non-Linear Layer L0 3x3 => 15"
pSFALayerNL0.x_field_channels=5
pSFALayerNL0.y_field_channels=5
pSFALayerNL0.x_field_spacing=5
pSFALayerNL0.y_field_spacing=5
#pSFALayerL0.in_channel_dim=1

pSFALayerNL0.pca_node_class = mdp.nodes.GSFANode
pSFALayerNL0.pca_out_dim = 16
#pSFALayerL0.pca_args = {"block_size": block_size}
pSFALayerNL0.pca_args = {"block_size": -1, "train_mode": -1}

pSFALayerNL0.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class

#exp_funcs_L3 = [identity, pair_prod_ex, pair_prod_adj1_ex, pair_prod_adj2_ex, pair_prod_adj3_ex]
pSFALayerNL0.exp_funcs = [identity, pair_prod_mix1_ex]
pSFALayerNL0.inv_use_hint = True
pSFALayerNL0.max_steady_factor=6.5
pSFALayerNL0.delta_factor=0.8
pSFALayerNL0.min_delta=0.0000001

#default
#self.inv_use_hint = True
#self.inv_max_steady_factor=0.35
#self.inv_delta_factor=0.6
#self.inv_min_delta=0.0001
        

pSFALayerNL0.red_node_class = mdp.nodes.WhiteningNode
pSFALayerNL0.red_out_dim = 0.9999999
pSFALayerNL0.red_args = {}

pSFALayerNL0.sfa_node_class = mdp.nodes.GSFANode
pSFALayerNL0.sfa_out_dim = 16
#pSFALayerNL0.sfa_args = {"block_size": -1, "train_mode": -1}

pSFALayerNL0.cloneLayer = True
pSFALayerNL0.name = comp_layer_name(pSFALayerNL0.cloneLayer, pSFALayerNL0.exp_funcs, pSFALayerNL0.x_field_channels, pSFALayerNL0.y_field_channels, pSFALayerNL0.pca_out_dim, pSFALayerNL0.sfa_out_dim)
system_parameters.test_object_contents(pSFALayerNL0)

print "******** Setting Layer NL1 Parameters *********************"
pSFALayerNL1 = system_parameters.ParamsSFALayer()
pSFALayerNL1.name = "Homogeneous Non-Linear Layer L1 3x3 => 30"
pSFALayerNL1.x_field_channels=3
pSFALayerNL1.y_field_channels=3
pSFALayerNL1.x_field_spacing=3
pSFALayerNL1.y_field_spacing=3
#pSFALayerL1.in_channel_dim = pSFALayerL0.sfa_out_dim

pSFALayerNL1.pca_node_class = mdp.nodes.GSFANode
#pSFALayerL0.pca_args = {"block_size": block_size}
pSFALayerNL1.pca_args = {"block_size": -1, "train_mode": -1}

#pca_out_dim_L1 = 90
#pca_out_dim_L1 = sfa_out_dim_L0 x_field_channels_L1 * x_field_channels_L1 * 0.75 
#pSFALayerNL1.pca_out_dim = 125
pSFALayerNL1.pca_out_dim = 125
#pSFALayerL1.pca_args = {"block_size": block_size}
#pSFALayerNL1.pca_args = {}

pSFALayerNL1.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class

pSFALayerNL1.exp_funcs = [identity, pair_prod_mix1_ex ]
pSFALayerNL1.inv_use_hint = True
pSFALayerNL1.max_steady_factor=6.5
pSFALayerNL1.delta_factor=0.8
pSFALayerNL1.min_delta=0.0000001

pSFALayerNL1.red_node_class = mdp.nodes.WhiteningNode
pSFALayerNL1.red_out_dim = 0.99999
pSFALayerNL1.red_args = {}

pSFALayerNL1.sfa_node_class = mdp.nodes.GSFANode
#sfa_out_dim_L1 = 12
pSFALayerNL1.sfa_out_dim = 30
#pSFALayerNL1.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL1 = False
pSFALayerNL1.cloneLayer = True
pSFALayerNL1.name = comp_layer_name(pSFALayerNL1.cloneLayer, pSFALayerNL1.exp_funcs, pSFALayerNL1.x_field_channels, pSFALayerNL1.y_field_channels, pSFALayerNL1.pca_out_dim, pSFALayerNL1.sfa_out_dim)
system_parameters.test_object_contents(pSFALayerNL1)

print "******** Setting Layer NL2 Parameters *********************"
pSFALayerNL2 = system_parameters.ParamsSFALayer()
pSFALayerNL2.name = "Inhomogeneous Non-Linear Layer L2 3x3 => 300 => 40"
pSFALayerNL2.x_field_channels=3
pSFALayerNL2.y_field_channels=3
pSFALayerNL2.x_field_spacing=3
pSFALayerNL2.y_field_spacing=3
#pSFALayerL2.in_channel_dim = pSFALayerL1.sfa_out_dim

pSFALayerNL2.pca_node_class = mdp.nodes.GSFANode
#pca_out_dim_L2 = 90
#pca_out_dim_L2 = sfa_out_dim_L0 x_field_channels_L2 * x_field_channels_L2 * 0.75 
pSFALayerNL2.pca_out_dim = 270
pSFALayerNL2.pca_args = {"block_size": -1, "train_mode": -1}

pSFALayerNL2.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class

pSFALayerNL2.exp_funcs = [identity, pair_prod_mix1_ex]
pSFALayerNL2.inv_use_hint = True
pSFALayerNL2.max_steady_factor=6.5
pSFALayerNL2.delta_factor=0.8
pSFALayerNL2.min_delta=0.0000001

pSFALayerNL2.red_node_class = mdp.nodes.WhiteningNode
pSFALayerNL2.red_out_dim = 0.99999
pSFALayerNL2.red_args = {}

pSFALayerNL2.sfa_node_class = mdp.nodes.GSFANode
#sfa_out_dim_L2 = 12
pSFALayerNL2.sfa_out_dim = 40
#pSFALayerNL2.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL2 = False
pSFALayerNL2.cloneLayer = False
pSFALayerNL2.name = comp_layer_name(pSFALayerNL2.cloneLayer, pSFALayerNL2.exp_funcs, pSFALayerNL2.x_field_channels, pSFALayerNL2.y_field_channels, pSFALayerNL2.pca_out_dim, pSFALayerNL2.sfa_out_dim)
system_parameters.test_object_contents(pSFALayerNL2)


print "******** Setting Layer NL3 Parameters *********************"
pSFANL3 = system_parameters.ParamsSFASuperNode()
pSFANL3.name = "SFA Non-Linear Super Node L3  all => 300 => 40"
#pSFAL3.in_channel_dim = pSFALayerL2.sfa_out_dim

#pca_node_L3 = mdp.nodes.WhiteningNode(output_dim=pca_out_dim_L3) 
pSFANL3.pca_node_class = mdp.nodes.GSFANode
#pca_out_dim_L3 = 210
#pca_out_dim_L3 = 0.999
#WARNING!!! CHANGED PCA TO SFA
pSFANL3.pca_out_dim = 300
#pSFALayerL1.pca_args = {"block_size": block_size}
pSFANL3.pca_args = {"block_size": -1, "train_mode": -1}

pSFANL3.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class

#exp_funcs_L3 = [identity, pair_prod_ex, pair_prod_mix1_ex, pair_prod_adj2_ex, pair_prod_adj3_ex]
pSFANL3.exp_funcs = [identity, pair_prod_mix1_ex]
pSFANL3.inv_use_hint = True
pSFANL3.max_steady_factor=6.5
pSFANL3.delta_factor=0.8
pSFANL3.min_delta=0.0000001

pSFANL3.red_node_class = mdp.nodes.WhiteningNode
pSFANL3.red_out_dim = 0.999999
pSFANL3.red_args = {}

pSFANL3.sfa_node_class = mdp.nodes.GSFANode
#sfa_out_dim_L1 = 12
pSFANL3.sfa_out_dim = 40
#pSFANL3.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL1 = False
pSFANL3.cloneLayer = False
pSFANL3.name = comp_supernode_name(pSFANL3.exp_funcs, pSFANL3.pca_out_dim, pSFANL3.sfa_out_dim)
system_parameters.test_object_contents(pSFANL3)


print "******** Setting Layer NL4 Parameters *********************"
pSFANL4 = system_parameters.ParamsSFASuperNode()
pSFANL4.name = "SFA Linear Super Node L4  all => 40 => 40"
#pSFAL4.in_channel_dim = pSFAL3.sfa_out_dim

#pca_node_L3 = mdp.nodes.WhiteningNode(output_dim=pca_out_dim_L3) 
pSFANL4.pca_node_class = mdp.nodes.GSFANode
#pca_out_dim_L3 = 210
#pca_out_dim_L3 = 0.999
#WARNING!!! CHANGED PCA TO SFA
pSFANL4.pca_out_dim = 40
#pSFALayerL1.pca_args = {"block_size": block_size}
pSFANL4.pca_args = {"block_size": -1, "train_mode": -1}

pSFANL4.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class

#exp_funcs_L3 = [identity, pair_prod_ex, pair_prod_adj1_ex, pair_prod_adj2_ex, pair_prod_adj3_ex]
pSFANL4.exp_funcs = [identity, pair_prod_mix1_ex]
pSFANL4.inv_use_hint = True
pSFANL4.max_steady_factor=6.5
pSFANL4.delta_factor=0.8
pSFANL4.min_delta=0.0000001

pSFANL4.red_node_class = mdp.nodes.WhiteningNode
pSFANL4.red_out_dim = 0.99999
pSFANL4.red_args = {}

pSFANL4.sfa_node_class = mdp.nodes.GSFANode
#sfa_out_dim_L1 = 12
pSFANL4.sfa_out_dim = 40
#pSFANL4.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL1 = False
pSFANL4.cloneLayer = False
pSFANL4.name = comp_supernode_name(pSFANL4.exp_funcs, pSFANL4.pca_out_dim, pSFANL4.sfa_out_dim)
system_parameters.test_object_contents(pSFANL4)


####################################################################
###########            NON-LINEAR NETWORKS              ############
####################################################################  
NL_Network4L = system_parameters.ParamsNetwork()
NL_Network4L.name = "Fully Non-Linear 4 Layer Network"
NL_Network4L.L0 = pSFALayerNL0
NL_Network4L.L1 = pSFALayerNL1
NL_Network4L.L2 = pSFALayerNL2
NL_Network4L.L3 = pSFANL3

NL_Network5L = system_parameters.ParamsNetwork()
NL_Network5L.name = "Fully Non-Linear 5 Layer Network"
NL_Network5L.L0 = pSFALayerNL0
NL_Network5L.L1 = pSFALayerNL1
NL_Network5L.L2 = pSFALayerNL2
NL_Network5L.L3 = pSFANL3
NL_Network5L.L4 = pSFANL4

Test_Network = system_parameters.ParamsNetwork()
Test_Network.name = "Test 5 Layer Network"
Test_Network.L0 = pSFALayerL0
Test_Network.L1 = pSFALayerL1
Test_Network.L2 = pSFALayerL2
Test_Network.L3 = pSFANL3
Test_Network.L4 = pSFANL4




print "*******************************************************************"
print "******** Creating Linear Thin 6L SFA Network          ******************"
print "*******************************************************************"
print "******** Setting Layer L0 Parameters          *********************"
# 15 / 5x5 = 0.60, 12 / 4x4 = 0.75
layer=None
layer = pSFATLayerL0 = system_parameters.ParamsSFALayer()
layer.name = "Homogeneous Thin Linear Layer L0 4x4 => 13 => x => 13"
layer.x_field_channels=4
layer.y_field_channels=4
layer.x_field_spacing=4
layer.y_field_spacing=4
#layer.in_channel_dim=1

#Warning!!!
layer.pca_node_class = mdp.nodes.PCANode
#layer.pca_node_class = mdp.nodes.GSFANode
layer.pca_out_dim = 13
#layer.pca_args = {"block_size": block_size}
layer.pca_args = {}

layer.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class

layer.exp_funcs = [identity,]

#WARNING!
#layer.red_node_class = mdp.nodes.WhiteningNode
layer.red_node_class = None
layer.red_out_dim = 0.9999999
layer.red_args = {}


layer.sfa_node_class = mdp.nodes.GSFANode
layer.sfa_out_dim = 13
layer.sfa_args = {}

#Warning, default: layer.cloneLayer = True
layer.cloneLayer = False
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
system_parameters.test_object_contents(layer)

print "******** Setting Layer L1 Parameters *********************"
layer=None
layer = pSFATLayerL1 = system_parameters.ParamsSFALayer()

layer.name = "Homogeneous Thin Linear Layer L1 2x2 => 47 x => 47, => 40"
layer.x_field_channels=2
layer.y_field_channels=2
layer.x_field_spacing=2
layer.y_field_spacing=2
#layer.in_channel_dim = pSFALayerL0.sfa_out_dim

layer.pca_node_class = mdp.nodes.WhiteningNode
#pca_out_dim_L1 = 90
#pca_out_dim_L1 = sfa_out_dim_L0 x_field_channels_L1 * x_field_channels_L1 * 0.75 
#125/(9*15) = 0.926, 45/(4*12)=0.9375
layer.pca_out_dim = 47
#layer.pca_args = {"block_size": block_size}
layer.pca_args = {}

layer.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class

layer.exp_funcs = [identity,]

layer.red_node_class = mdp.nodes.WhiteningNode
layer.red_out_dim = 0.99999
layer.red_args = {}

layer.sfa_node_class = mdp.nodes.GSFANode
#sfa_out_dim_L1 = 12
#30/(9*15) = 0.222, (4*12)
layer.sfa_out_dim = 40
layer.sfa_args = {}
#Default: cloneLayerL1 = False
layer.cloneLayer = False
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
system_parameters.test_object_contents(layer)

print "******** Setting Layer L2 Parameters *********************"
layer = None
layer = pSFATLayerL2 = system_parameters.ParamsSFALayer()
layer.name = "Inhomogeneous Thin Linear Layer L2 2x2 => 158 => 158 => 70"
layer.x_field_channels=2
layer.y_field_channels=2
layer.x_field_spacing=2
layer.y_field_spacing=2
#layer.in_channel_dim = layer.sfa_out_dim

layer.pca_node_class = mdp.nodes.GSFANode
#pca_out_dim_L2 = 90
#pca_out_dim_L2 = sfa_out_dim_L1 x_field_channels_L2 * x_field_channels_L2 * 0.75 
layer.pca_out_dim = 100
#layer.pca_args = {"block_size": block_size}
layer.pca_args = {}

layer.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class

layer.exp_funcs = [identity,]

layer.red_node_class = mdp.nodes.WhiteningNode
#Note: number in (0,1) might potentially cause problems, if cells have different output_dim
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros(layer.pca_out_dim)))-2
layer.red_args = {}

layer.sfa_node_class = mdp.nodes.GSFANode
#sfa_out_dim_L2 = 12
layer.sfa_out_dim = 70
layer.sfa_args = {}
#Default: cloneLayerL2 = False
layer.cloneLayer = False
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
system_parameters.test_object_contents(layer)


print "******** Setting Layer L3 Parameters *********************"
layer = None
layer = pSFATLayerL3 = copy.deepcopy(pSFATLayerL2)
layer.name = "Inhomogeneous Thin Linear Layer L3 2x2 => 100 => I => w - 2 => 70"
layer.x_field_channels=2
layer.y_field_channels=2
layer.x_field_spacing=2
layer.y_field_spacing=2
#layer.in_channel_dim = layer.sfa_out_dim

layer.pca_node_class = mdp.nodes.GSFANode
#pca_out_dim_L2 = 90
#pca_out_dim_L2 = sfa_out_dim_L1 x_field_channels_L2 * x_field_channels_L2 * 0.75 
layer.pca_out_dim = 100
#layer.pca_args = {"block_size": block_size}
layer.pca_args = {}

layer.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class

layer.exp_funcs = [identity,]

layer.red_node_class = mdp.nodes.WhiteningNode
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros(layer.pca_out_dim)))-2
layer.red_args = {}

layer.sfa_node_class = mdp.nodes.GSFANode
#sfa_out_dim_L2 = 12
layer.sfa_out_dim = 70
layer.sfa_args = {}
#Default: cloneLayerL2 = False
layer.cloneLayer = False
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
system_parameters.test_object_contents(layer)


print "******** Setting Layer L4 Parameters *********************"
layer = None
layer = pSFATLayerL4 = copy.deepcopy(pSFATLayerL3)
layer.name = "Inhomogeneous Thin Linear Layer L4 2x2 => 278 => 278 => 70"
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
system_parameters.test_object_contents(layer)

print "******** Setting Layer L5 Parameters *********************"
layer = None
layer = pSFATLayerL5 = copy.deepcopy(pSFATLayerL4)
layer.name = "Inhomogeneous Thin Linear Layer L5 2x2 => 278 => 278 => 70"
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
system_parameters.test_object_contents(layer)


####################################################################
###########           THIN LINEAR NETWORK               ############
####################################################################  
network = linearNetworkT6L = system_parameters.ParamsNetwork()
network.name = "Linear 6 Layer Network"

network.L0 = pSFATLayerL0
network.L1 = pSFATLayerL1
network.L2 = pSFATLayerL2
network.L3 = pSFATLayerL3
network.L4 = pSFATLayerL4
network.L5 = pSFATLayerL5

#L0 has input 128x128
#L1 has input 32x32, here I will only use horizontal sparseness
#x_in_channels = 32
#base = 2
#increment = 2
#n_values = compute_lsrf_n_values(x_in_channels, base, increment)
#network.L1.nx_value = n_values[0]
#network.L2.nx_value = n_values[1]
#network.L3.nx_value = n_values[2]
#network.L4.nx_value = n_values[3]

print "*******************************************************************"
print "******** Creating Non-Linear Thin 6L SFA Network ******************"
print "*******************************************************************"
#Warning, this is based on the linear network, thus modifications to the linear 
#network also affect this non linear network
#exp_funcs = [identity, pair_prod_ex, pair_prod_mix1_ex, pair_prod_adj2_ex, pair_prod_adj3_ex]
layer = pSFATLayerNL0 = copy.deepcopy(pSFATLayerL0)
layer.exp_funcs = [identity, pair_prod_mix1_ex]
w = sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros((1,layer.pca_out_dim)))
layer.red_out_dim = len(w[0])-2

layer = pSFATLayerNL1 = copy.deepcopy(pSFATLayerL1)
layer.exp_funcs = [identity, pair_prod_mix1_ex]
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros((1,layer.pca_out_dim)))[0])-2

layer = pSFATLayerNL2 = copy.deepcopy(pSFATLayerL2)
layer.exp_funcs = [identity, pair_prod_mix1_ex]
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros((1,layer.pca_out_dim)))[0])-2

layer = pSFATLayerNL3 = copy.deepcopy(pSFATLayerL3)
layer.exp_funcs = [identity, pair_prod_mix1_ex]
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros((1,layer.pca_out_dim)))[0])-2

layer = pSFATLayerNL4 = copy.deepcopy(pSFATLayerL4)
layer.exp_funcs = [identity, pair_prod_adj2_ex]
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros((1,layer.pca_out_dim)))[0])-2

layer = pSFATLayerNL5 = copy.deepcopy(pSFATLayerL5)
layer.exp_funcs = [identity, pair_prod_adj2_ex]
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros((1,layer.pca_out_dim)))[0])-2


####################################################################
###########           THIN Non-LINEAR NETWORK               ############
####################################################################  
network = nonlinearNetworkT6L = system_parameters.ParamsNetwork()
network.name = "Non-Linear 6 Layer Network"
network.L0 = pSFATLayerNL0
network.L1 = pSFATLayerNL1
network.L2 = pSFATLayerNL2
network.L3 = pSFATLayerNL3
network.L4 = pSFATLayerNL4
network.L5 = pSFATLayerNL5


network = TestNetworkT6L = system_parameters.ParamsNetwork()
network.name = "Test Non-Linear 6 Layer Network"
network.L0 = pSFATLayerL0
network.L1 = pSFATLayerL1
network.L2 = pSFATLayerL2
network.L3 = pSFATLayerNL3
network.L4 = pSFATLayerNL4
network.L5 = pSFATLayerNL5




print "*******************************************************************"
print "******** Creating Linear Ultra Thin 11L SFA Network ***************"
print "*******************************************************************"

print "******** Copying Layer L0 Parameters from  pSFATLayerL0    ********"
pSFAULayerL0 = copy.deepcopy(pSFATLayerL0)

print "******** Setting Ultra Thin Layer L1 H Parameters *********************"
layer=None
layer = pSFAULayerL1_H = system_parameters.ParamsSFALayer()

layer.name = "Homogeneous Ultra Thin Linear Layer L1 1x2 (>= 26) => 26 x => 26, => 20"
layer.x_field_channels=2
layer.y_field_channels=1
layer.x_field_spacing=2
layer.y_field_spacing=1

#WARNING!!!!! mdp.nodes.GSFANode
layer.pca_node_class = mdp.nodes.GSFANode
layer.pca_out_dim = 26
#layer.pca_args = {"block_size": block_size, "train_mode": -1}
layer.pca_args = {}

layer.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class

layer.exp_funcs = [identity,]

layer.red_node_class = None
layer.red_out_dim = 0.99999
layer.red_args = {}

layer.sfa_node_class = mdp.nodes.GSFANode
layer.sfa_out_dim = 20
layer.sfa_args = {}
#Warning!!! layer.cloneLayer = True
layer.cloneLayer = False
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
system_parameters.test_object_contents(layer)



print "******** Setting Ultra Thin Layer L1 V Parameters *********************"
layer=None
layer = pSFAULayerL1_V = system_parameters.ParamsSFALayer()

layer.name = "Homogeneous Ultra Thin Linear Layer L1 2x1 (>= 40) => 40 x => 40, => 35"
layer.x_field_channels=1
layer.y_field_channels=2
layer.x_field_spacing=1
layer.y_field_spacing=2

#layer.in_channel_dim = pSFALayerL0.sfa_out_dim
layer.pca_node_class = mdp.nodes.GSFANode #WhiteningNode
layer.pca_out_dim = 40
#layer.pca_args = {"block_size": block_size}
layer.pca_args = {}

layer.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class

layer.exp_funcs = [identity]

layer.red_node_class = None
layer.red_out_dim = 0.99999
layer.red_args = {}

layer.sfa_node_class = mdp.nodes.GSFANode
layer.sfa_out_dim = 35
layer.sfa_args = {}
#Warning!!!! layer.cloneLayer = True
layer.cloneLayer = False
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
system_parameters.test_object_contents(layer)


print "******** Setting Layer L2 H Parameters *********************"
layer = None
layer = pSFAULayerL2_H = system_parameters.ParamsSFALayer()
layer.name = "Inhomogeneous Ultra Thin Linear Layer L2 1x2 (>=70) => 70 => x => (x-2) => 60"
layer.x_field_channels=2
layer.y_field_channels=1
layer.x_field_spacing=2
layer.y_field_spacing=1

layer.pca_node_class = mdp.nodes.GSFANode
#pca_out_dim_L2 = 90
#pca_out_dim_L2 = sfa_out_dim_L1 x_field_channels_L2 * x_field_channels_L2 * 0.75 
layer.pca_out_dim = 70
#layer.pca_args = {"block_size": block_size}
layer.pca_args = {}

layer.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class
layer.ord_args = system_parameters.ParamsSFALayer.ord_args

layer.exp_funcs = [identity,]

layer.red_node_class = None 
#mdp.nodes.PCANode
#Note: number in (0,1) might potentially cause problems, if cells have different output_dim
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros(layer.pca_out_dim)))-2
layer.red_args = {}

layer.sfa_node_class = mdp.nodes.GSFANode
#sfa_out_dim_L2 = 12
layer.sfa_out_dim = 60
#layer.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL2 = False
layer.cloneLayer = False
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
system_parameters.test_object_contents(layer)


print "******** Setting Layer L2 V Parameters *********************"
layer = None
layer = pSFAULayerL2_V = system_parameters.ParamsSFALayer()
layer.name = "Inhomogeneous Ultra Thin Linear Layer L2 2x1 (>=120) => 120 => x => (x-2) => 60"
layer.x_field_channels=1
layer.y_field_channels=2
layer.x_field_spacing=1
layer.y_field_spacing=2

layer.pca_node_class = mdp.nodes.GSFANode
#pca_out_dim_L2 = 90
#pca_out_dim_L2 = sfa_out_dim_L1 x_field_channels_L2 * x_field_channels_L2 * 0.75 
layer.pca_out_dim = 120
#layer.pca_args = {"block_size": block_size}
layer.pca_args = {"block_size": -1, "train_mode": -1}

layer.ord_node_class = system_parameters.ParamsSFALayer.ord_node_class

layer.exp_funcs = [identity,]

layer.red_node_class = None
#mdp.nodes.WhiteningNode
#Note: number in (0,1) might potentially cause problems, if cells have different output_dim
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros(layer.pca_out_dim)))-2
layer.red_args = {}

layer.sfa_node_class = mdp.nodes.GSFANode
#sfa_out_dim_L2 = 12
layer.sfa_out_dim = 60
#layer.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL2 = False
layer.cloneLayer = False
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
system_parameters.test_object_contents(layer)


print "******** Setting Layer L3 H Parameters *********************"
layer = None
layer = pSFAULayerL3_H = copy.deepcopy(pSFAULayerL2_V)
layer.name = "Inhomogeneous Ultra Linear Layer L3 1x2 (>=120) =>  120 => x => x-2 => 60"
layer.x_field_channels=2
layer.y_field_channels=1
layer.x_field_spacing=2
layer.y_field_spacing=1
layer.exp_funcs = [identity,]
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
system_parameters.test_object_contents(layer)

print "******** Setting Layer L3 V Parameters *********************"
layer = None
layer = pSFAULayerL3_V = copy.deepcopy(pSFAULayerL2_V)
layer.name = "Inhomogeneous Ultra Linear Layer L3 2x1 (>=120) =>  120 => x => x-2 => 60"
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
system_parameters.test_object_contents(layer)

print "******** Setting Layer L4 H Parameters *********************"
layer = None
layer = pSFAULayerL4_H = copy.deepcopy(pSFAULayerL3_H)
layer.name = "Inhomogeneous Ultra Linear Layer L4 1x2 (>=120) =>  120 => x => x-2 => 60"
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
system_parameters.test_object_contents(layer)

print "******** Setting Layer L4 V Parameters *********************"
layer = None
layer = pSFAULayerL4_V = copy.deepcopy(pSFAULayerL3_V)
layer.name = "Inhomogeneous Ultra Linear Layer L3 2x1 (>=120) =>  120 => x => x-2 => 60"
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
system_parameters.test_object_contents(layer)


print "******** Setting Layer L5 H Parameters *********************"
layer = None
layer = pSFAULayerL5_H = copy.deepcopy(pSFAULayerL3_H)
layer.name = "Inhomogeneous Ultra Linear Layer L4 1x2 (>=120) =>  120 => x => x-2 => 60"
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
system_parameters.test_object_contents(layer)

print "******** Setting Layer L5 V Parameters *********************"
layer = None
layer = pSFAULayerL5_V = copy.deepcopy(pSFAULayerL3_V)
layer.name = "Inhomogeneous Ultra Linear Layer L3 2x1 (>=120) =>  120 => x => x-2 => 60"
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
system_parameters.test_object_contents(layer)

####################################################################
###########           THIN LINEAR NETWORK               ############
####################################################################  
network = linearNetworkU11L = system_parameters.ParamsNetwork()
network.name = "Linear Ultra Thin 11 Layer Network"

network.L0 = pSFAULayerL0
network.L1 = pSFAULayerL1_H
network.L2 = pSFAULayerL1_V
network.L3 = pSFAULayerL2_H
network.L4 = pSFAULayerL2_V
network.L5 = pSFAULayerL3_H
network.L6 = pSFAULayerL3_V
network.L7 = pSFAULayerL4_H
network.L8 = pSFAULayerL4_V
network.L9 = pSFAULayerL5_H
network.L10 = pSFAULayerL5_V

network.layers = [network.L0, network.L1, network.L2, network.L3, network.L4, network.L5, network.L6, network.L7, \
                  network.L8, network.L9, network.L10]

for layer in network.layers:
    layer.pca_node_class = None

setup_pca_sfa_expos=False
if setup_pca_sfa_expos:
    for i, layer in enumerate(network.layers):
        if i > 0:
            layer.pca_node_class = None
        else:
            layer.pca_node_class = mdp.nodes.GSFANode
            layer.pca_args["sfa_expo"] = 1
            layer.pca_args["pca_expo"] = 1        



#Either PCANode and no expansion in L0, or
#       SFANode and some expansion possible

#Warning!!! enable_sparseness = True
#Mega Warning !!!
enable_sparseness = False
if enable_sparseness:
    print "Sparseness Activated"
    xy_in_channels = 16 #32
    base = 2
    increment = 2
    n_values = lattice.compute_lsrf_n_values(xy_in_channels, base, increment)
#L0 is done normally, after L0 there are usually 128/4 = 32 incoming blocks or 64/4 = 16 incoming blocks
    network.L1.nx_value = n_values[0]
    network.L2.ny_value = n_values[0]
    network.L3.nx_value = n_values[1]
    network.L4.ny_value = n_values[1]
    network.L5.nx_value = n_values[2]
    network.L6.ny_value = n_values[2]
#    network.L7.nx_value = n_values[3]
#    network.L8.ny_value = n_values[3]


####################################################################
###########          PCA   NETWORKS                     ############
####################################################################  
#Set uniform dimensionality reduction
#PCA_out_dim... PCA_in_dim
#num_layers=11
network = linearPCANetwork4L = copy.deepcopy(linearNetwork4L)
pca_in_dim=(5*3*3*3)**2
pca_out_dim=1000
pca_num_layers = 4
reduction_per_layer = (pca_out_dim *1.0 /pca_in_dim)**(1.0/pca_num_layers)
L0_PCA_out_dim = reduction_per_layer * (5)**2
L1_PCA_out_dim = L0_PCA_out_dim * reduction_per_layer * (3)**2
L2_PCA_out_dim = L1_PCA_out_dim * reduction_per_layer * (3)**2
L3_PCA_out_dim = pca_out_dim
L0_PCA_out_dim = int(L0_PCA_out_dim)
L1_PCA_out_dim = int(L1_PCA_out_dim)
L2_PCA_out_dim = int(L2_PCA_out_dim)
LN_PCA_out_dims = [L0_PCA_out_dim, L1_PCA_out_dim , L2_PCA_out_dim, L3_PCA_out_dim ]
print "linearPCANetwork4L, L0-3_PCA_out_dim = ", LN_PCA_out_dims
for i, layer in enumerate(network.layers):
    layer.pca_node_class = None
    layer.pca_args = {}
    layer.ord_node_class = None
    layer.ord_args = {}     
    layer.red_node_class = None
    layer.red_args = {}
    layer.sfa_node_class = mdp.nodes.PCANode
    layer.sfa_out_dim = LN_PCA_out_dims[i]
    layer.sfa_args = {}
network.layers[len(network.layers)-1].sfa_node_class = mdp.nodes.WhiteningNode

network = linearPCANetworkU11L = copy.deepcopy(linearNetworkU11L)
network.name = "linearPCANetworkU11L"
pca_in_dim=(4*1*2*1*2*1*2*1*2*1*2)**2 #
pca_out_dim=120 #120 #200
pca_num_layers = 11
reduction_per_layer = (pca_out_dim *1.0 /pca_in_dim)**(1.0/pca_num_layers)
L0_PCA_out_dim = reduction_per_layer * (4)**2
L1_PCA_out_dim = L0_PCA_out_dim * reduction_per_layer * 2
L2_PCA_out_dim = L1_PCA_out_dim * reduction_per_layer * 2
L3_PCA_out_dim = L2_PCA_out_dim * reduction_per_layer * 2
L4_PCA_out_dim = L3_PCA_out_dim * reduction_per_layer * 2
L5_PCA_out_dim = L4_PCA_out_dim * reduction_per_layer * 2
L6_PCA_out_dim = L5_PCA_out_dim * reduction_per_layer * 2
L7_PCA_out_dim = L6_PCA_out_dim * reduction_per_layer * 2
L8_PCA_out_dim = L7_PCA_out_dim * reduction_per_layer * 2
L9_PCA_out_dim = L8_PCA_out_dim * reduction_per_layer * 2
L10_PCA_out_dim = pca_out_dim
L0_PCA_out_dim = int(L0_PCA_out_dim)
L1_PCA_out_dim = int(L1_PCA_out_dim)
L2_PCA_out_dim = int(L2_PCA_out_dim)
L3_PCA_out_dim = int(L3_PCA_out_dim)
L4_PCA_out_dim = int(L4_PCA_out_dim)
L5_PCA_out_dim = int(L5_PCA_out_dim)
L6_PCA_out_dim = int(L6_PCA_out_dim)
L7_PCA_out_dim = int(L7_PCA_out_dim)
L8_PCA_out_dim = int(L8_PCA_out_dim)
L9_PCA_out_dim = int(L9_PCA_out_dim)
LN_PCA_out_dims = [ L0_PCA_out_dim, L1_PCA_out_dim , L2_PCA_out_dim, L3_PCA_out_dim, L4_PCA_out_dim, L5_PCA_out_dim , L6_PCA_out_dim, L7_PCA_out_dim, L8_PCA_out_dim, L9_PCA_out_dim , L10_PCA_out_dim]

#print "LN_PCA_out_dims=", LN_PCA_out_dims, "reduction_per_layer=", reduction_per_layer
#quit()

override_linearPCANetworkU11L_output_dims = True #and False
if override_linearPCANetworkU11L_output_dims:
    print "WARNING!!!! Overriding output_dimensionalities of PCA Network, to fit SFA Network or exceed it"
    LN_PCA_out_dims = [ 13, 20,35,60,60,60,60,60,60,60,60 ]
    LN_PCA_out_dims = [ 13, 20,35,60,100,120,120,120,120,120,120 ]    

print "linearPCANetworkU11L, L0-10_PCA_out_dim = ", LN_PCA_out_dims
for i, layer in enumerate(network.layers):
    layer.pca_node_class = None
    layer.pca_args = {}
    layer.ord_node_class = None
    layer.ord_args = {}     
    layer.red_node_class = None
    layer.red_args = {}
    layer.sfa_node_class = mdp.nodes.PCANode
    layer.sfa_out_dim = LN_PCA_out_dims[i]
    layer.sfa_args = {}

#network.layers[len(network.layers)-1].sfa_node_class = mdp.nodes.WhiteningNode
rec_field_size = 4 #6 => 192x192, 5=> 160x160, 4=>128x128
network.layers[0].x_field_channels = rec_field_size
network.layers[0].y_field_channels = rec_field_size
network.layers[0].x_field_spacing = rec_field_size
network.layers[0].y_field_spacing = rec_field_size

#******************************************************************************************
#Network for age estimation, 11 layers: 160x160, 9 layers: 80x80
network = linearPCANetworkU11L_5x5L0 = copy.deepcopy(linearPCANetworkU11L)
rec_field_size = 5 #6 => 192x192, 5=> 160x160, 4=>128x128
network.layers[0].x_field_channels = rec_field_size
network.layers[0].y_field_channels = rec_field_size
network.layers[0].x_field_spacing = rec_field_size
network.layers[0].y_field_spacing = rec_field_size
#LN_PCA_out_dims_5x5L0 = [ 22, 42, 76, 120, 150, 150, 150, 150, 150, 150, 150 ] #Base network 
#Newer version, output has 200 features (SFA+PC has 100)
LN_PCA_out_dims_5x5L0 = [ 17, 23, 31, 43, 58, 79, 108, 147, 200, 200, 200 ] #Base network 

for i, layer in enumerate(network.layers):
    layer.sfa_out_dim = LN_PCA_out_dims_5x5L0[i]
    

network = linearWhiteningNetwork11L = copy.deepcopy(linearPCANetworkU11L)
for i, layer in enumerate(network.layers):
    layer.sfa_node_class = mdp.nodes.WhiteningNode


network = HiGSFANetworkU11L_5x5L0 = copy.deepcopy(linearPCANetworkU11L)
# Original:
#HiGSFANet_out_dims = [ 13, 20,35,60,60,60,60,60,60,60,60 ]
# Accomodating more space for principal components
#HiGSFANet_out_dims = [ 22, 35, 45,60,60,60,60,60,60,60,60 ]
#HiGSFANet_out_dims = [ 25, 35,45,60,60,60,60,60,60,60,60 ]

#Enable 80x80 images
rec_field_size = 5 #6 => 192x192, 5=> 160x160, 4=>128x128
LRec_use_RGB_images = False #or True
network.layers[0].x_field_channels = rec_field_size
network.layers[0].y_field_channels = rec_field_size
network.layers[0].x_field_spacing = rec_field_size
network.layers[0].y_field_spacing = rec_field_size
network.layers[0].pca_node_class = mdp.nodes.PCANode
if LRec_use_RGB_images:
    network.layers[0].pca_out_dim = 50 #50 Problem, more color components also remove higher frequency ones = code image as intensity+color!!!
else:
    network.layers[0].pca_out_dim = 20 #20 #30 #28 #20 #22 more_feats2, 50 for RGB

#print "network.layers[0].x_field_channels=", network.layers[0].x_field_channels
#print "network.L0.x_field_channels=", network.L0.x_field_channels
#quit()

if LRec_use_RGB_images == False:
#25+14=39 => 30 (16 PCA/25); 16+13=29 => 22 (9 PCA/16)
#First 80x80 Network:
#HiGSFANet_out_dims = [ 30, 42, 52,60,60,60,60,60,60,60,60 ] 
#Improved 80x80 Network: (more feats 2) ******* TOP PARAMS for grayscale
#    HiGSFANet_out_dims = [ 31, 42, 60,70,70,70,70,70,70,70,70 ] #Official for SFA Net
#    HiGSFANet_out_dims = [ 31, 42, 60,75,75,75,75,75,75,75,75 ] #OFFICIAL LRecNet
#    HiGSFANet_out_dims = [ 31, 52, 75, 85, 90, 90, 85, 100, 75,75,75 ] #OFFICIAL FOR AGE ARTICLE HGSFA+PC, TOP NETWORK, 3 FEATS
    HiGSFANet_out_dims = [ 28, 52, 75, 85, 90, 90, 85, 100, 75,75,75 ] #Experiment with control2: Dt=1.9999
#    HiGSFANet_out_dims = [ 22, 49, 75, 85, 90, 90, 85, 100, 75,75,75 ] #Experiment with control2: Dt=1.999
#    HiGSFANet_out_dims = [ 21, 43, 75, 85, 90, 90, 85, 100, 75,75,75 ] #Experiment with control2: Dt=1.99
#    HiGSFANet_out_dims = [ 20, 40, 75, 85, 90, 90, 85, 100, 75,75,75 ] #Experiment with control2: Dt=1.9
#    HiGSFANet_out_dims = [ 20, 30, 43, 70, 82, 76, 84, 82, 75, 75, 75] #CONTROL EXPERIMENT FOR AGE ARTICLE, HGSFA, PLAIN SFA ARCHITECTURE WITHOUT PRINCIPAL COMPONENTS, 3Feats
    
#    HiGSFANet_out_dims = [ 31, 42, 60,80,80,80,80,80,80,80,80 ] #Test F2
#Even more features:
#HiGSFANet_out_dims = [ 32, 46, 65,75,75,75,75,75,75,75,75 ] #Is this official for HiGSFA Net?
#Base 96x96 Network: (more feats 4)
#HiGSFANet_out_dims = [ 38, 48, 65,70,70,70,70,70,70,70,70 ]
#Base 96x96 Network: (more feats 5)
#HiGSFANet_out_dims = [ 40, 50, 70,70,70,70,70,70,70,70,70 ]
    print "Number of network features set without RGB:", HiGSFANet_out_dims
else:
#Improved 80x80 Network, RGB version (more feats 2)
#    HiGSFANet_out_dims = [ 48, 60, 70,70,70,70,70,70,70,70,70 ] #(Orig)
    HiGSFANet_out_dims = [ 39, 51, 65,70,70,70,70,70,70,70,70 ] #(less features)
    print "Adjusting number of network features due to RGB:", HiGSFANet_out_dims

print "HiGSFANetworkU11L_5x5L0 L0-10_SFA_out_dim = ", HiGSFANet_out_dims

for i, layer in enumerate(network.layers):
    layer.sfa_node_class = mdp.nodes.iGSFANode
    layer.sfa_out_dim = HiGSFANet_out_dims[i]
    #QT_AP_L(k=nan, d=0.8), Q_AN_exp
    #layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":10, ,  "max_preserved_sfa":2.0}
#    layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":1, ,  "offsetting_mode":"sensitivity_based", "max_preserved_sfa":2.0}
#    layer.sfa_args = {"pre_expansion_node_class":mdp.nodes.GSFANode, "expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":2.0}
    layer.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":None,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.9999} #AGE ARTICLE: 1.99999 official, 1.99999, 2.0. Control2: 1.9999
    #sel60_unsigned_08expo sel_exp(60, unsigned_08expo), unsigned_08expo

    #"offsetting_mode": "QR_decomposition", "sensitivity_based_pure", "sensitivity_based_normalized", "max_comp features"
#    layer.sfa_args = {"expansion_funcs":None, "use_pca":True, "operation":"lin_app", ""max_lenght_slow_part"":10, "max_num_samples_for_ev":1200, "max_test_samples_for_ev":1200, "k":200}
#    layer.sfa_args = {"expansion_funcs":None, "use_pca":True, "max_lenght_slow_part":6, "max_num_samples_for_ev":600, "max_test_samples_for_ev":600, "k":16}
#network.layers[0].pca_node_class = mdp.nodes.PCANode
#network.layers[0].pca_out_dim = 13

#WARNING, ADDING AN ADDITIONAL SFA NODE IN THE LAST LAYER, 80x80 resolution (Node 9)
double_SFA_top_node = True #and False
if double_SFA_top_node:
    layer = network.layers[8]
    layer.pca_node_class = mdp.nodes.iGSFANode
    layer.pca_out_dim = HiGSFANet_out_dims[8]
    layer.pca_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, div2_sel75_unsigned_08expo], "max_lenght_slow_part":None,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} #2.0
    layer.sfa_node_class = mdp.nodes.iGSFANode
    layer.sfa_out_dim = HiGSFANet_out_dims[8]
    layer.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, sel8_04QT], "max_lenght_slow_part":None,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} #2.0
    

#WARNING, EXPERIMENTAL CODE TO TEST OTHER EXPANSIONS
#network.layers[10].sfa_node_class = mdp.nodes.GSFANode
#network.layers[6].sfa_args = {"expansion_funcs":[Q_exp], "max_lenght_slow_part":10,   "max_preserved_sfa":2.0}






#Networks for AGE estimation MORPH-II
################## NETWORK FOR TESTING ACCORDING TO GUO ET AL, USES 3 LABELS ######################################
network = HiGSFANetworkU11L_Overlap6x6L0_GUO_3Labels = copy.deepcopy(linearPCANetworkU11L)
rec_field_size = 6 #6 => 192x192, 5=> 160x160, 4=>128x128
LRec_use_RGB_images = False #or True
network.layers[0].x_field_channels = rec_field_size
network.layers[0].y_field_channels = rec_field_size
#overlapping L0
network.layers[0].x_field_spacing = int(numpy.ceil(rec_field_size/2.0)) #rec_field_size rec_field_size/2
network.layers[0].y_field_spacing = int(numpy.ceil(rec_field_size/2.0)) #rec_field_size rec_field_size/2
network.layers[0].pca_node_class = mdp.nodes.PCANode
if LRec_use_RGB_images:
    network.layers[0].pca_out_dim = 56 #50 Problem, more color components also remove higher frequency ones = code image as intensity+color!!!
else:
    network.layers[0].pca_out_dim = 20 #16 #20 #30 #28 #20 #22 more_feats2, 50 for RGB

#network.layers[0].red_node_class = mdp.nodes.HeadNode
#network.layers[0].red_out_dim = 18 #18

for i in range(1,len(network.layers),2):
    network.layers[i].x_field_channels = 3
    network.layers[i].y_field_channels = 1
    network.layers[i].x_field_spacing = 2
    network.layers[i].y_field_spacing = 1
for i in range(2,len(network.layers),2):
    network.layers[i].x_field_channels = 1
    network.layers[i].y_field_channels = 3
    network.layers[i].x_field_spacing = 1
    network.layers[i].y_field_spacing = 2

if LRec_use_RGB_images == False:
    HiGSFANet_out_dims = [ 18, 27, 37, 66, 79, 88, 88, 93, 95, 75, 75 ] #New test
    #HiGSFANet_out_dims = [ 16, 24, 33, 60, 72, 80, 80, 85, 75, 75, 75 ] #Experiment with control2: Dt=1.9999
#    HiGSFANet_out_dims = [ 16, 24, 33, 60, 72, 80, 80, 75, 75, 50, 75 ] #Experiment with control2: Dt=1.9999
    print "Number of network features set without RGB:", HiGSFANet_out_dims
else:
    print "unsupported"
    quit()
    HiGSFANet_out_dims = [ 39, 51, 65,70,70,70,70,70,70,70,70 ] #(less features)
    print "Adjusting number of network features due to RGB:", HiGSFANet_out_dims

print "HiGSFANetworkU11L_Overlap6x6L0 L0-10_SFA_out_dim = ", HiGSFANet_out_dims

for i, layer in enumerate(network.layers):
    layer.sfa_node_class = mdp.nodes.iGSFANode
    layer.sfa_out_dim = HiGSFANet_out_dims[i]
    layer.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":None,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.91} #1.85, 1.91 #only for tuning/experimentation, official is below 

network.layers[0].sfa_args["expansion_funcs"]= [s18, s15u08ex, s17Max, s10QT] # ** s18, s15u08ex, s17Max, s10QT] #identity, s14u08ex, s14Max, s1QT, QT, s14u08ex, unsigned_08expo, s14QT, maximum_mix1_ex, s16QT, s10Max
network.layers[0].sfa_args["max_preserved_sfa"]= 3 #T 2 #1 #! 2 # **1 #1 3

network.layers[1].sfa_args["expansion_funcs"]= [ch3s20, ch3s20u08, ch3s20max, ch3o3s4QT, ch3o0s3QT] # ** ch3s16, ch3s16u08, ch3s16max, ch3o1s4QT #unsigned_08expo, ch3_O2_s3_QT, ch3_O3_s3_QT
network.layers[1].sfa_args["max_preserved_sfa"]= 4 #T 3 #2 #! 3 # ** 2 #3

network.layers[2].sfa_args["expansion_funcs"]= [ch3s28, ch3s28u08, ch3s16max, ch3o4s6QT, ch3o0s3QT] # ** ch3s24, ch3s24u08, ch3s12max, ch3o2s6QT
network.layers[2].sfa_args["max_preserved_sfa"]= 1.933 + 0.003 + 0.0015 #T 1.9 #1.81 #! 1.9 # ** 1.81 1.91 #1.93 #(nothing) 3# (nothing)4

network.layers[3].sfa_args["expansion_funcs"]= [ch3s37, ch3s37u08, ch3s19max, ch3o4s6QT, ch3o0s3QT]   # ** ch3s33, ch3s33u08, ch3s15max, ch3o2s6QT  # maximum_mix1_exp
network.layers[3].sfa_args["max_preserved_sfa"]= 1.933 + 0.005 + 0.0015 #T 1.9 #1.81 #! 1.9  # ** 1.81 #1.91 #1.92 # (nothing) 5

network.layers[4].sfa_args["expansion_funcs"]= [ch3s60, ch3s49u08, ch3s19max, ch3o4s6QT, ] # ** ch3s60, ch3s45u08, ch3s15max, ch3o2s6QT #ch3o3s5QT, maximum_mix1_ex,unsigned_08expo] #unsigned_08expo
network.layers[4].sfa_args["max_preserved_sfa"]= 1.933 + 0.005 + 0.0023 #T 1.9 #1.81 #! 1.9 # **1.81

network.layers[5].sfa_args["expansion_funcs"]= [ch3s72, ch3s40u08, ch3s11max, ch3o5s6QT, ch3o0s3QT] # ** ch3s72, ch3s35u08, ch3s6max, ch3o3s6QT #identity, ch3s50u08, ch3o3s3QT #maximum_mix1_ex,unsigned_08expo] #unsigned_08expo
network.layers[5].sfa_args["max_preserved_sfa"]= 1.953 + 0.005 + 0.0031 #T 1.93 #1.86 #! 1.93 # ** 1.81

network.layers[6].sfa_args["expansion_funcs"]= [ch3s80, ch3s25u08, ch3s15max, ch3o6s4QT,] # ** ch3s80, ch3s20u08, ch3s10max, ch3o4s4QT 
network.layers[6].sfa_args["max_preserved_sfa"]= 1.973 + 0.000 #T 1.96 #1.89 #! 1.96 # ** 1.89

network.layers[7].sfa_args["expansion_funcs"]= [ch3s80, ch3o6s20u08,] # ** ch3s80, ch3o5s20u08 # ch3o4s5QT #unsigned_08expo, ch3o5s4QT
network.layers[7].sfa_args["max_preserved_sfa"]= 1.98 #T 1.97 #1.94 #! 1.97 # ** 1.81
 
network.layers[8].sfa_args["expansion_funcs"]= [ch3s85, ch3o6s20u08, ch3o0s3QT] # ** ch3s85, ch3o4s20u08, #ch3o4s4QT #identity, ch3s70u08, unsigned_08expo, ch3_s25_max
network.layers[8].sfa_args["max_preserved_sfa"]= 1.98 #T 1.97 #1.94 #! 1.97 # **1.81

#WARNING, ADDING AN ADDITIONAL SFA NODE IN THE LAST LAYER, 80x80 resolution (Node 9)
double_SFA_top_node = True #and False
if double_SFA_top_node:
    print "adding additional top node (two nodes in layer 8)"
    layer = network.layers[8]
    #NODE 19
    layer.pca_node_class = mdp.nodes.iGSFANode
    layer.pca_out_dim = HiGSFANet_out_dims[8]
    layer.pca_args = dict(layer.sfa_args)
    #NODE 20 / layer[9]
    layer.sfa_node_class = mdp.nodes.iGSFANode
    layer.sfa_out_dim = HiGSFANet_out_dims[9]
    layer.sfa_args = dict(layer.sfa_args)
    layer.sfa_args["expansion_funcs"]= [s75, s34u08, o7s15QT, s3QT] # ** s75, s30u08ex, o5s15QT #s50u08ex, o4s18QT , s3QT
    layer.sfa_args["max_preserved_sfa"]= 1.9 #! 1.81 # ** 1.81

my_DT=1.96 #=1.96, 3 Labels
for i in range(2, 9):
    network.layers[i].sfa_args["max_preserved_sfa"] = my_DT
network.layers[8].pca_args["max_preserved_sfa"] = my_DT

HiGSFANetworkU11L_Overlap6x6L0_GUO_3Labels_96x96 = copy.deepcopy(HiGSFANetworkU11L_Overlap6x6L0_GUO_3Labels)
HiGSFANetworkU11L_Overlap6x6L0_GUO_3Labels_96x96.layers = HiGSFANetworkU11L_Overlap6x6L0_GUO_3Labels_96x96.layers[0:9]

################### HiGSFA Network for FGNet generalization ##############
network = HiGSFANetworkU11L_Overlap6x6L0_Generalization_3Labels_96x96 = copy.deepcopy(HiGSFANetworkU11L_Overlap6x6L0_GUO_3Labels_96x96)
network.layers[0].sfa_args["expansion_funcs"]= [s18, s15u08ex, s17Max, s9QT] #s9QT #[s18, s15u08ex, s17Max, s10QT] 
network.layers[0].sfa_args["max_preserved_sfa"]= 3 
network.layers[1].sfa_args["expansion_funcs"]= [ch3s20, ch3s20u08, ch3s20max, ch3o3s3QT, ch3o0s3QT] #ch3o3s3QT #[ch3s20, ch3s20u08, ch3s20max, ch3o3s4QT, ch3o0s3QT] 
network.layers[1].sfa_args["max_preserved_sfa"]= 3 
network.layers[2].sfa_args["expansion_funcs"]= [ch3s28, ch3s28u08, ch3s16max, ch3o3s5QT, ] #ch3o3s5QT #ch3o0s3QT #[ch3s28, ch3s28u08, ch3s16max, ch3o4s6QT, ch3o0s3QT] 
network.layers[2].sfa_args["max_preserved_sfa"]= 3  
network.layers[3].sfa_args["expansion_funcs"]= [ch3s37, ch3s37u08, ch3s19max, ch3o3s4QT] #ch3o3s5QT, ch3o0s3QT #[ch3s37,ch3s37u08, ch3s19max, ch3o4s6QT, ch3o0s3QT]   
network.layers[3].sfa_args["max_preserved_sfa"]= 4
network.layers[4].sfa_args["expansion_funcs"]= [ch3s60, ch3s49u08, ch3s19max, ch3o4s4QT] #ch3o4s5QT #ch3s19max, ch3o4s5QT #[ch3s60, ch3s49u08, ch3s19max, ch3o4s6QT] 
network.layers[4].sfa_args["max_preserved_sfa"]= 4
network.layers[5].sfa_args["expansion_funcs"]= [ch3s72, ch3o4s40u08, ch3o4s11max, ch3o4s4QT, ] #ch3s40u08, ch3s11max, ch3o4s5QT #[ch3s72, ch3s40u08, ch3s11max, ch3o5s6QT, ch3o0s3QT] 
network.layers[5].sfa_args["max_preserved_sfa"]= 5
network.layers[6].sfa_args["expansion_funcs"]= [ch3s80, ch3o5s25u08, ch3o5s15max,] #ch3s25u08, ch3o5s4QT, ch3s15max, #[ch3s80, ch3s25u08, ch3s15max, ch3o6s4QT] 
network.layers[6].sfa_args["max_preserved_sfa"]= 5
network.layers[7].sfa_args["expansion_funcs"]= [ch3s80, ch3o5s20u08,] #[ch3s80, ch3o6s20u08,]
network.layers[7].sfa_args["max_preserved_sfa"]= 6
network.layers[8].sfa_args["expansion_funcs"]= [ch3s85, ch3o6s20u08, ] # ch3s20u08 #[ch3s85, ch3o6s20u08, ch3o0s3QT] 
network.layers[8].sfa_args["max_preserved_sfa"]= 8
network.layers[8].pca_node_class = None

print "############################ Genrealization network #######################"
for i, layer in enumerate(network.layers):
    print "GENERALIZATION:Layer %i has expansion"%i, layer.sfa_args["expansion_funcs"]
#    layer.sfa_args["offsetting_mode"]=None
#    layer.sfa_args["reconstruct_with_sfa"]=False

################### HSFA Network for age estimation ######################
network = HSFANetworkU11L_Overlap6x6L0_GUO_3Labels_96x96 = copy.deepcopy(HiGSFANetworkU11L_Overlap6x6L0_GUO_3Labels_96x96)
HSFANet_out_dims = [ 14, 20, 27, 49, 60, 61, 65, 65, 75, 75, 70 ]  #[ 39, 51, 65,70,70,70,70,70,70,70,70 ] 
for i, layer in enumerate(network.layers):
    #layer.sfa_node_class = mdp.nodes.SFANode
    layer.sfa_args["max_preserved_sfa"]= HSFANet_out_dims[i]
    layer.sfa_args["offsetting_mode"]=None
    layer.sfa_args["reconstruct_with_sfa"]=False    
    layer.sfa_args["expansion_funcs"] = [identity, unsigned_08expo]
    layer.sfa_out_dim = HSFANet_out_dims[i]
#network.layers[8].pca_node_class = None
network.layers[8].pca_args["max_preserved_sfa"]= 66
network.layers[8].pca_args["offsetting_mode"]=None
network.layers[8].pca_args["reconstruct_with_sfa"]=False
network.layers[8].pca_args["expansion_funcs"] = [identity, unsigned_08expo]
network.layers[8].pca_out_dim = 66

#network.layers.append(copy.deepcopy(network.layers[8]))
#network.layers[9].x_field_channels = 1
#network.layers[9].y_field_channels = 1
#network.layers[9].x_field_spacing = 1
#network.layers[9].y_field_spacing = 1


################## NETWORK FOR TESTING ACCORDING TO GUO ET AL, USES 1 LABEL ######################################
network = HiGSFANetworkU11L_Overlap6x6L0_GUO_1Label = copy.deepcopy(HiGSFANetworkU11L_Overlap6x6L0_GUO_3Labels)
if LRec_use_RGB_images:
    network.layers[0].pca_out_dim = 56 #50 Problem, more color components also remove higher frequency ones = code image as intensity+color!!!
else:
    network.layers[0].pca_out_dim = 20 #16 #20 #30 #28 #20 #22 more_feats2, 50 for RGB

#network.layers[0].red_node_class = mdp.nodes.HeadNode
#network.layers[0].red_out_dim = 18 #18

for i in range(1,len(network.layers),2):
    network.layers[i].x_field_channels = 3
    network.layers[i].y_field_channels = 1
    network.layers[i].x_field_spacing = 2
    network.layers[i].y_field_spacing = 1
for i in range(2,len(network.layers),2):
    network.layers[i].x_field_channels = 1
    network.layers[i].y_field_channels = 3
    network.layers[i].x_field_spacing = 1
    network.layers[i].y_field_spacing = 2

if LRec_use_RGB_images == False:
    HiGSFANet_out_dims = [ 16, 25, 35, 64, 77, 86, 86, 92, 93, 75, 75 ] #
#    HiGSFANet_out_dims = [ 18, 27, 37, 66, 79, 88, 88, 93, 95, 75, 75 ] #New test
    #HiGSFANet_out_dims = [ 16, 24, 33, 60, 72, 80, 80, 85, 75, 75, 75 ] #Experiment with control2: Dt=1.9999
#    HiGSFANet_out_dims = [ 16, 24, 33, 60, 72, 80, 80, 75, 75, 50, 75 ] #Experiment with control2: Dt=1.9999
    print "Number of network features set without RGB:", HiGSFANet_out_dims
else:
    print "unsupported"
    quit()
    HiGSFANet_out_dims = [ 39, 51, 65,70,70,70,70,70,70,70,70 ] #(less features)
    print "Adjusting number of network features due to RGB:", HiGSFANet_out_dims

print "HiGSFANetworkU11L_Overlap6x6L0_GUO_1Label L0-10_SFA_out_dim = ", HiGSFANet_out_dims

network.layers[0].sfa_args["expansion_funcs"]= [s18, s15u08ex, s17Max, s10QT] # ** s18, s15u08ex, s17Max, s10QT] #identity, s14u08ex, s14Max, s1QT, QT, s14u08ex, unsigned_08expo, s14QT, maximum_mix1_ex, s16QT, s10Max
network.layers[0].sfa_args["max_preserved_sfa"]= 1 #T 2 #1 #! 2 # **1 #1 3

network.layers[1].sfa_args["expansion_funcs"]= [ch3s20, ch3s20u08, ch3s20max, ch3o1s4QT] # ** ch3s16, ch3s16u08, ch3s16max, ch3o1s4QT #unsigned_08expo, ch3_O2_s3_QT, ch3_O3_s3_QT
network.layers[1].sfa_args["max_preserved_sfa"]= 2 #T 3 #2 #! 3 # ** 2 #3

network.layers[2].sfa_args["expansion_funcs"]= [ch3s28, ch3s28u08, ch3s16max, ch3o2s6QT] # ** ch3s24, ch3s24u08, ch3s12max, ch3o2s6QT
network.layers[2].sfa_args["max_preserved_sfa"]= 1.933 + 0.003 + 0.0015 #T 1.9 #1.81 #! 1.9 # ** 1.81 1.91 #1.93 #(nothing) 3# (nothing)4

network.layers[3].sfa_args["expansion_funcs"]= [ch3s37, ch3s37u08, ch3s19max, ch3o2s6QT]   # ** ch3s33, ch3s33u08, ch3s15max, ch3o2s6QT  # maximum_mix1_exp
network.layers[3].sfa_args["max_preserved_sfa"]= 1.933 + 0.005 + 0.0015 #T 1.9 #1.81 #! 1.9  # ** 1.81 #1.91 #1.92 # (nothing) 5

network.layers[4].sfa_args["expansion_funcs"]= [ch3s60, ch3s49u08, ch3s19max, ch3o2s6QT, ] # ** ch3s60, ch3s45u08, ch3s15max, ch3o2s6QT #ch3o3s5QT, maximum_mix1_ex,unsigned_08expo] #unsigned_08expo
network.layers[4].sfa_args["max_preserved_sfa"]= 1.933 + 0.005 + 0.0023 #T 1.9 #1.81 #! 1.9 # **1.81

network.layers[5].sfa_args["expansion_funcs"]= [ch3s72, ch3s40u08, ch3s11max, ch3o3s6QT] # ** ch3s72, ch3s35u08, ch3s6max, ch3o3s6QT #identity, ch3s50u08, ch3o3s3QT #maximum_mix1_ex,unsigned_08expo] #unsigned_08expo
network.layers[5].sfa_args["max_preserved_sfa"]= 1.953 + 0.005 + 0.0031 #T 1.93 #1.86 #! 1.93 # ** 1.81

network.layers[6].sfa_args["expansion_funcs"]= [ch3s80, ch3s25u08, ch3s15max, ch3o4s4QT,] # ** ch3s80, ch3s20u08, ch3s10max, ch3o4s4QT 
network.layers[6].sfa_args["max_preserved_sfa"]= 1.973 + 0.000 #T 1.96 #1.89 #! 1.96 # ** 1.89

network.layers[7].sfa_args["expansion_funcs"]= [ch3s80, ch3o6s20u08,] # ** ch3s80, ch3o5s20u08 # ch3o4s5QT #unsigned_08expo, ch3o5s4QT
network.layers[7].sfa_args["max_preserved_sfa"]= 1.98 #T 1.97 #1.94 #! 1.97 # ** 1.81
 
network.layers[8].sfa_args["expansion_funcs"]= [ch3s85, ch3o6s20u08, ] # ** ch3s85, ch3o4s20u08, #ch3o4s4QT #identity, ch3s70u08, unsigned_08expo, ch3_s25_max
network.layers[8].sfa_args["max_preserved_sfa"]= 1.98 #T 1.97 #1.94 #! 1.97 # **1.81

#WARNING, ADDING AN ADDITIONAL SFA NODE IN THE LAST LAYER, 80x80 resolution (Node 9)
double_SFA_top_node = True #and False
if double_SFA_top_node:
    print "adding additional top node (two nodes in layer 8)"
    layer = network.layers[8]
    #NODE 19
    layer.pca_node_class = mdp.nodes.iGSFANode
    layer.pca_out_dim = HiGSFANet_out_dims[8]
    layer.pca_args = dict(layer.sfa_args)
    #NODE 20 / layer[9]
    layer.sfa_node_class = mdp.nodes.iGSFANode
    layer.sfa_out_dim = HiGSFANet_out_dims[9]
    layer.sfa_args = dict(layer.sfa_args)
    layer.sfa_args["expansion_funcs"]= [s75, s34u08, o7s15QT, ] # ** s75, s30u08ex, o5s15QT #s50u08ex, o4s18QT , s3QT
    layer.sfa_args["max_preserved_sfa"]= 1.9 #! 1.81 # ** 1.81

my_DT=1.91 #Definitive, 1 Label
for i in range(2, 9):
    network.layers[i].sfa_args["max_preserved_sfa"] = my_DT
network.layers[8].pca_args["max_preserved_sfa"] = my_DT

#########################      SFANetworkU11L_Overlap6x6L0_GUO_3Labels          ###################################################
#SFANet_out_dims = [ 18, 27, 37, 66, 79, 88, 88, 93, 95, 75, 75 ]
#SFANet_out_dims = [ 12, 18, 25, 46, 55, 61, 61, 65, 66, 75, 75 ]
SFANet_out_dims = [ 14, 20, 27, 49, 60, 61, 65, 65, 66, 75, 75] #eedit
#SFANet_out_dims = [ 14, 20, 27, 49, 60, 61, 65, 65, 66, 70, 75] #eedit WARNING, REDUCING FROM 75 TO 70 FEATS TO AVOID NUMERIC ERROR
#SFANet_out_dims = [ 40, 20, 27, 49, 60, 61, 65, 65, 66, 70, 75] #eedit WARNING, REDUCING FROM 75 TO 70 FEATS TO AVOID NUMERIC ERROR
#SFANet_out_dims = [ 14, 20, 27, 49, 60, 61, 65, 65, 70, 75, 75] #eedit WARNING, INCREASING FROM 66 to 70 FEATS TO AVOID NUMERIC ERROR
factor_out_dim = 1.0

use_sfapc_nodes = False #or True
DT = [3, 4, 1.96, 1.96, 1.96, 1.96, 1.96, 1.96, 1.96, 1.96, 1.96] #last entry is not used
#DT = [3, 4, 1.9375, 1.9395, 1.9403, 1.9611, 1.973, 1.98, 1.98, 1.9, 1.9] #last entry is not used
#DT = [3, 4, 1.98, 1.98, 1.98, 1.98, 1.98, 1.98, 1.98, 1.98, 1.98] #last entry is not used

network = SFANetworkU11L_Overlap6x6L0_GUO_3Labels = copy.deepcopy(HiGSFANetworkU11L_Overlap6x6L0_GUO_3Labels)
for i, layer in enumerate(network.layers):
    if use_sfapc_nodes:
        layer.sfa_args["max_preserved_sfa"] = DT[i]
    else:
        layer.sfa_args["max_preserved_sfa"] = 4.0 # Preserve all slow features
        layer.sfa_args["offsetting_mode"] = None
        layer.sfa_args["reconstruct_with_sfa"] = False
    layer.sfa_out_dim = int(SFANet_out_dims[i]*factor_out_dim)


network.layers[8].pca_out_dim = int(SFANet_out_dims[8]*factor_out_dim)
network.layers[8].sfa_out_dim = int(SFANet_out_dims[9]*factor_out_dim)

if use_sfapc_nodes == False:
    layer.pca_args["max_preserved_sfa"] = 4.0 # Preserve all slow features
    layer.pca_args["offsetting_mode"] = None
    layer.pca_args["reconstruct_with_sfa"] = False
else:
    network.layers[8].pca_args["max_preserved_sfa"] = DT[8] 
    network.layers[8].sfa_args["max_preserved_sfa"] = DT[9]

network.layers[0].sfa_args["expansion_funcs"]= [s17, s20u08ex, s17Max, s10QT, ] #s17, s20u08ex, s17Max, s10QT
network.layers[1].sfa_args["expansion_funcs"]= [ch3s8, ch3s12u08, ch3s10max, ch3o0s3QT] #ch3s8, ch3s12u08, ch3s10max, ch3o0s3QT 
network.layers[2].sfa_args["expansion_funcs"]= [ch3s18, ch3s18u08, ch3s16max] #ch3s18, ch3s18u08, ch3s16max
network.layers[3].sfa_args["expansion_funcs"]= [ch3s25, ch3s23u08, ch3s19max, ch3o0s3QT]   #ch3s25, ch3s25u08, ch3s19max, ch3o0s3QT
network.layers[4].sfa_args["expansion_funcs"]= [ch3s49, ch3s43u08, ch3s19max, ] #ch3s49, ch3s43u08, ch3s19max,
network.layers[5].sfa_args["expansion_funcs"]= [ch3s58, ch3s37u08, ch3s13max, ch3o0s3QT] #ch3s58, ch3s37u08, ch3s13max, ch3o0s3QT
network.layers[6].sfa_args["expansion_funcs"]= [ch3s61, ch3s21u08, ch3s17max, ] #ch3s61, ch3s21u08, ch3s17max
network.layers[7].sfa_args["expansion_funcs"]= [ch3s49, ] #ch3s49,
network.layers[8].pca_args["expansion_funcs"]= [ch3s59, ch3o0s3QT] #ch3s59, ch3o0s3QT
network.layers[8].sfa_args["expansion_funcs"]= [s66, s34u08, s3QT] #s66, s34u08, s3QT

#for i, layer in enumerate(network.layers):
#    if i>0:
#        layer.sfa_args["expansion_funcs"]= [identity]


############################## NETWORK FOR AGE ESTIMATION, STANDAR ARQUITECTURE ######################
network = HiGSFANetworkU11L_Overlap6x6L0_1Label = copy.deepcopy(linearPCANetworkU11L)
rec_field_size = 6 #6 => 192x192, 5=> 160x160, 4=>128x128
LRec_use_RGB_images = False #or True
network.layers[0].x_field_channels = rec_field_size
network.layers[0].y_field_channels = rec_field_size
#overlapping L0
network.layers[0].x_field_spacing = int(numpy.ceil(rec_field_size/2.0)) #rec_field_size rec_field_size/2
network.layers[0].y_field_spacing = int(numpy.ceil(rec_field_size/2.0)) #rec_field_size rec_field_size/2
network.layers[0].pca_node_class = mdp.nodes.PCANode
if LRec_use_RGB_images:
    network.layers[0].pca_out_dim = 56 #50 Problem, more color components also remove higher frequency ones = code image as intensity+color!!!
else:
    network.layers[0].pca_out_dim = 20 #16 #20 #30 #28 #20 #22 more_feats2, 50 for RGB

network.layers[0].red_node_class = mdp.nodes.HeadNode
network.layers[0].red_out_dim = 16 #WARNING ERROR EXPERIMENT 18 #18

# network.layers[1].x_field_channels = 2
# network.layers[1].y_field_channels = 1
# network.layers[1].x_field_spacing = 1
# network.layers[1].y_field_spacing = 1
# 
# network.layers[2].x_field_channels = 1
# network.layers[2].y_field_channels = 2
# network.layers[2].x_field_spacing = 1
# network.layers[2].y_field_spacing = 1

for i in range(1,len(network.layers),2):
    network.layers[i].x_field_channels = 3
    network.layers[i].y_field_channels = 1
    network.layers[i].x_field_spacing = 2
    network.layers[i].y_field_spacing = 1
for i in range(2,len(network.layers),2):
    network.layers[i].x_field_channels = 1
    network.layers[i].y_field_channels = 3
    network.layers[i].x_field_spacing = 1
    network.layers[i].y_field_spacing = 2

#WARNING, ADJUSTMENT FOR 72x72 net
#network.layers[7].x_field_channels = 2
#network.layers[7].y_field_channels = 1
#network.layers[7].x_field_spacing = 2
#network.layers[7].y_field_spacing = 1
#network.layers[8].x_field_channels = 1
#network.layers[8].y_field_channels = 2
#network.layers[8].x_field_spacing = 1
#network.layers[8].y_field_spacing = 2


#print "network.layers[0].x_field_channels=", network.layers[0].x_field_channels
#print "network.L0.x_field_channels=", network.L0.x_field_channels
#quit()

if LRec_use_RGB_images == False:
    HiGSFANet_out_dims = [ 40, 24, 33, 60, 72, 80, 80, 85, 75, 75, 75 ] #WARNING 
#    HiGSFANet_out_dims = [ 16, 24, 33, 60, 72, 80, 80, 85, 75, 75, 75 ] 
#    HiGSFANet_out_dims = [ 16, 24, 33, 60, 72, 80, 80, 75, 75, 50, 75 ] #Experiment with control2: Dt=1.9999
    print "Number of network features set without RGB:", HiGSFANet_out_dims
else:
    print "unsupported"
    quit()
    HiGSFANet_out_dims = [ 39, 51, 65,70,70,70,70,70,70,70,70 ] #(less features)
    print "Adjusting number of network features due to RGB:", HiGSFANet_out_dims

print "HiGSFANetworkU11L_Overlap6x6L0 L0-10_SFA_out_dim = ", HiGSFANet_out_dims

for i, layer in enumerate(network.layers):
    layer.sfa_node_class = mdp.nodes.iGSFANode
    layer.sfa_out_dim = HiGSFANet_out_dims[i]
    #QT_AP_L(k=nan, d=0.8), Q_AN_exp
    #layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":10,   "max_preserved_sfa":2.0}
#    layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":1,   "offsetting_mode":"sensitivity_based", "max_preserved_sfa":2.0}
#    layer.sfa_args = {"pre_expansion_node_class":mdp.nodes.GSFANode, "expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":2.0}
    layer.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":None,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.91} #only for tuning/experimentation, official is below 
    #layer.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99} #AGE ARTICLE: 1.99999 official, 1.99999, 2.0. Control2: 1.9999
    #sel60_unsigned_08expo sel_exp(60, unsigned_08expo), unsigned_08expos14u08ex


#network.layers[0].sfa_args["expansion_funcs"]= [identity, s14u08ex, s17Max, s9QT] #identity, s14u08ex, s14Max, s10QT, QT, s14u08ex, unsigned_08expo, s14QT, maximum_mix1_ex, s16QT, s10Max
network.layers[0].sfa_args["expansion_funcs"]= [identity, s14u08ex, s12Max] #WARNING, EXPERIMENT Delta Values!
network.layers[0].sfa_args["max_preserved_sfa"]= 40 #1#1 3
#network.layers[0].sfa_args["expansion_funcs"]= [identity, QT] #unsigned_08expo

#network.layers[0].sfa_args["expansion_funcs"]= [identity, maximum_mix1_ex,unsigned_08expo] #unsigned_08expo
network.layers[1].sfa_args["expansion_funcs"]= [identity, ch3s16u08, ch3s10max, ch3o1s5QT] #ch3o1s3QT, ch3o1s5QT, unsigned_08expo, ch3_O2_s3_QT, ch3_O3_s3_QT
network.layers[1].sfa_args["max_preserved_sfa"]= 2 #2 #3
#network.layers[1].sfa_args["expansion_funcs"]= [identity, maximum_mix1_ex,unsigned_08expo] #unsigned_08expo

network.layers[2].sfa_args["expansion_funcs"]= [identity, ch3s24u08, ch3s12max, ch3o2s3QT] #identity, ch3o2s3QT, ch3s16u08, ch3s12max, unsigned_08expo, ch3s20u08, ch3s12max, ch3o3s3QT
network.layers[2].sfa_args["max_preserved_sfa"]= 1.91 #1.93 #(nothing) 3# (nothing)4

network.layers[3].sfa_args["expansion_funcs"]= [identity, ch3s33u08, ch3o3s3QT]  #ch3o3s4QT, ch3o3s26u08, ch3o3s12max, unsigned_08expo, maximum_mix1_ex,
#network.layers[3].sfa_args["expansion_funcs"]= [identity, ch3o3s26u08, ch3o3s16max,]  #ch3o3s26u08, ch3o3s12max, unsigned_08expo, maximum_mix1_ex,
network.layers[3].sfa_args["max_preserved_sfa"]= 1.91 #1.92 # (nothing) 5

network.layers[4].sfa_args["expansion_funcs"]= [identity, ch3s45u08, ch3o3s4QT] #ch3o3s5QT, maximum_mix1_ex,unsigned_08expo] #unsigned_08expo
#network.layers[4].sfa_args["max_preserved_sfa"]= 6 #8

network.layers[5].sfa_args["expansion_funcs"]= [identity, ch3s50u08, ch3o3s3QT] # maximum_mix1_ex,unsigned_08expo] #unsigned_08expo
#network.layers[5].sfa_args["max_preserved_sfa"]= 7 #9

network.layers[6].sfa_args["expansion_funcs"]= [identity, ch3s40u08, ch3o3s5QT] # ch3s40u08, ch3o3s5QT]  #unsigned_08expo, ch3o3s5QT
#network.layers[6].sfa_args["max_preserved_sfa"]= 7 #(nothing) 10

network.layers[7].sfa_args["expansion_funcs"]= [identity, ch3s40u08, ch3o4s5QT] #unsigned_08expo, ch3o5s4QT
#network.layers[7].sfa_args["max_preserved_sfa"]= 9 # 11

network.layers[8].sfa_args["expansion_funcs"]= [identity, ch3s30u08, ch3o4s4QT] # identity, ch3s70u08, unsigned_08expo, ch3_s25_max
#network.layers[8].sfa_args["max_preserved_sfa"]= 5

#noise_addition=0.000004
#network.layers[1].ord_node_class = mdp.nodes.NoiseNode(noise_func=numpy.random.uniform, noise_args=(-(3**0.5)*noise_addition, (3**0.5)*noise_addition), noise_type='additive')

#WARNING, ADDING AN ADDITIONAL SFA NODE IN THE LAST LAYER, 80x80 resolution (Node 9)
double_SFA_top_node = True #and False
if double_SFA_top_node:
    print "adding additional top node (two nodes in layer 8)"
    layer = network.layers[8]
    #NODE 19
    layer.pca_node_class = mdp.nodes.iGSFANode
    layer.pca_out_dim = HiGSFANet_out_dims[8]
    layer.pca_args = dict(layer.sfa_args)
    #NODE 20 / layer[9]
    layer.sfa_node_class = mdp.nodes.iGSFANode
    layer.sfa_out_dim = HiGSFANet_out_dims[9]
    layer.sfa_args = dict(layer.sfa_args)
    layer.sfa_args["expansion_funcs"]= [identity, s40u08ex, o4s15QT] #s50u08ex, o4s18QT
    

for i, layer in enumerate(network.layers):
    if i>0:
        layer.sfa_args["expansion_funcs"]= [identity]
    if i==9:
        layer.pca_args["expansion_funcs"]= [identity]

#********************************************************************************************************
network = HiGSFANetworkU11L_Overlap6x6L0_3Labels = copy.deepcopy(HiGSFANetworkU11L_Overlap6x6L0_1Label)
if LRec_use_RGB_images:
    network.layers[0].pca_out_dim = 56 #50 Problem, more color components also remove higher frequency ones = code image as intensity+color!!!
else:
    network.layers[0].pca_out_dim = 20 #16 #20 #30 #28 #20 #22 more_feats2, 50 for RGB

network.layers[0].red_node_class = mdp.nodes.HeadNode
network.layers[0].red_out_dim = 18 #18

if LRec_use_RGB_images == False:
    HiGSFANet_out_dims = [ 18, 27, 35, 62, 74, 82, 82, 87, 77, 75, 75 ] #increased output dims by 2
#    HiGSFANet_out_dims = [ 16, 24, 33, 60, 72, 80, 80, 85, 75, 75, 75 ] #Experiment with control2: Dt=1.9999
#    HiGSFANet_out_dims = [ 16, 24, 33, 60, 72, 80, 80, 75, 75, 50, 75 ] #Experiment with control2: Dt=1.9999
    print "Number of network features set without RGB:", HiGSFANet_out_dims
else:
    print "unsupported"
    quit()
    HiGSFANet_out_dims = [ 39, 51, 65,70,70,70,70,70,70,70,70 ] #(less features)
    print "Adjusting number of network features due to RGB:", HiGSFANet_out_dims

print "HiGSFANetworkU11L_Overlap6x6L0_3Labels L0-10_SFA_out_dim = ", HiGSFANet_out_dims
for i, layer in enumerate(network.layers):
    layer.sfa_node_class = mdp.nodes.iGSFANode
    layer.sfa_out_dim = HiGSFANet_out_dims[i]

def delta_map(delta):
    return delta
#    return 1.97
#    return 2.0 - (2.0 - delta)*3.0/2.0 #2.5
#    return delta #+0.25 #delta
#return 2.0 - (2.0 - delta)*3.0/3.0 #2.5

network.layers[0].sfa_args["expansion_funcs"]= [s18, s14u08ex, s17Max, s10_d1_Q_N] #identity, s14u08ex, s17Max, s10QT/s10_d1_Q_N
network.layers[0].sfa_args["max_preserved_sfa"]= 3 #3 

network.layers[1].sfa_args["expansion_funcs"]= [ch3s18, ch3s16u08, ch3s10max, ch3_o3_s5_d1_Q_N, ch3_o0_s3_d1_Q_N] #identity, ch3s16u08, ch3s10max, ch3o3s5QT, ch3o0s3QT
network.layers[1].sfa_args["max_preserved_sfa"]= 4 #4

#network.layers[2].sfa_args["expansion_funcs"]= [ch3s26, ch3s24u08, ch3s12max, ch3_o4_s3_d1_Q_N, ch3_o0_s3_d1_Q_N] #identity, ch3s24u08, ch3s12max, ch3o4s3QT, ch3o0s3QT
network.layers[2].sfa_args["expansion_funcs"]= [ch3s27, ch3s27u08, ch3s12max, ch3_o4_s3_d1_Q_N, ch3_o0_s3_d1_Q_N] #Experiment changed o4s5 to o4s3
network.layers[2].sfa_args["max_preserved_sfa"]= delta_map(1.96) #1.97

network.layers[3].sfa_args["expansion_funcs"]= [ch3s35, ch3s33u08, ch3_o5_s3_d1_Q_N, ch3_o0_s3_d1_Q_N] #identity, ch3s33u08, ch3o5s3QT, ch3o0s3QT
#network.layers[3].sfa_args["expansion_funcs"]= [ch3s35, ch3s33u08, ch3o5s5QT, ch3_o0_s3_d1_Q_N] #Experiment
network.layers[3].sfa_args["max_preserved_sfa"]= delta_map(1.96) #1.97

network.layers[4].sfa_args["expansion_funcs"]= [ch3s62, ch3s45u08, ch3_o5_s4_d1_Q_N, ] #identity, ch3s45u08, ch3o5s4QT
network.layers[4].sfa_args["max_preserved_sfa"]= delta_map(1.96) #1.975

network.layers[5].sfa_args["expansion_funcs"]= [ch3s74, ch3s50u08, ch3_o5_s3_d1_Q_N] #identity, ch3s50u08, ch3o5s3QT
network.layers[5].sfa_args["max_preserved_sfa"]= delta_map(1.97) #1.975

network.layers[6].sfa_args["expansion_funcs"]= [ch3s82, ch3s40u08, ch3_o5_s5_d1_Q_N] #identity, ch3s40u08, ch3o5s5QT
network.layers[6].sfa_args["max_preserved_sfa"]= delta_map(1.97) #1.975

network.layers[7].sfa_args["expansion_funcs"]= [ch3s82, ch3s40u08, ch3_o6_s5_d1_Q_N, ] #identity, ch3s40u08, ch3o6s5QT/ch3_o6_s5_Q_N,
network.layers[7].sfa_args["max_preserved_sfa"]= delta_map(1.97) #1.975

network.layers[8].sfa_args["expansion_funcs"]= [ch3s87, ch3s30u08, ch3_o6_s4_d1_Q_N, ch3_o0_s3_d1_Q_N] #identity, ch3s30u08, ch3o6s4QT/ch3_o6_s4_d1_Q_N, ch3o0s3QT/ch3_o0_s3_d1_Q_N
network.layers[8].sfa_args["max_preserved_sfa"]= delta_map(1.97) #1.975

double_SFA_top_node = True #and False
if double_SFA_top_node:
    print "adding additional top node (two nodes in layer 8)"
    layer = network.layers[8]
    #NODE 19
    layer.pca_node_class = mdp.nodes.iGSFANode
    layer.pca_out_dim = HiGSFANet_out_dims[8]
    layer.pca_args = dict(layer.sfa_args)
    #NODE 20 / layer[9]
    layer.sfa_node_class = mdp.nodes.iGSFANode
    layer.sfa_out_dim = HiGSFANet_out_dims[9]
    layer.sfa_args = dict(layer.sfa_args)
    layer.sfa_args["expansion_funcs"]= [ch3s75, s40u08ex, o6_s15_d1_Q_N] #identity, s40u08ex, o6s15QT/o6_s15_d1_Q_N


#********************************************************************************************************
network = HiGSFANetworkU11L_Overlap6x6L0_2Labels = copy.deepcopy(HiGSFANetworkU11L_Overlap6x6L0_1Label)
if LRec_use_RGB_images:
    network.layers[0].pca_out_dim = 56 #50 Problem, more color components also remove higher frequency ones = code image as intensity+color!!!
else:
    network.layers[0].pca_out_dim = 20 #16 #20 #30 #28 #20 #22 more_feats2, 50 for RGB

network.layers[0].red_node_class = mdp.nodes.HeadNode
network.layers[0].red_out_dim = 18 #18

if LRec_use_RGB_images == False:
    HiGSFANet_out_dims = [ 16, 24, 33, 60, 72, 80, 80, 85, 75, 75, 75 ] #Experiment with control2: Dt=1.9999
#    HiGSFANet_out_dims = [ 16, 24, 33, 60, 72, 80, 80, 75, 75, 50, 75 ] #Experiment with control2: Dt=1.9999
    print "Number of network features set without RGB:", HiGSFANet_out_dims
else:
    print "unsupported"
    quit()
    HiGSFANet_out_dims = [ 39, 51, 65,70,70,70,70,70,70,70,70 ] #(less features)
    print "Adjusting number of network features due to RGB:", HiGSFANet_out_dims

print "HiGSFANetworkU11L_Overlap6x6L0_3Labels L0-10_SFA_out_dim = ", HiGSFANet_out_dims

network.layers[0].sfa_args["expansion_funcs"]= [identity, s14u08ex, s17Max, s10_d1_Q_N] #identity, s14u08ex, s17Max, s10QT/s10_d1_Q_N
network.layers[0].sfa_args["max_preserved_sfa"]= 2 #2 

network.layers[1].sfa_args["expansion_funcs"]= [identity, ch3s16u08, ch3s10max, ch3_o2_s5_d1_Q_N, ch3_o0_s2_d1_Q_N] #identity, ch3s16u08, ch3s10max, ch3o3s5QT, ch3o0s3QT
network.layers[1].sfa_args["max_preserved_sfa"]= 3 #3

network.layers[2].sfa_args["expansion_funcs"]= [identity, ch3s24u08, ch3s12max, ch3_o3_s3_d1_Q_N, ch3_o0_s2_d1_Q_N] #identity, ch3s24u08, ch3s12max, ch3o4s3QT, ch3o0s3QT
network.layers[2].sfa_args["max_preserved_sfa"]= 1.95 #1.95

network.layers[3].sfa_args["expansion_funcs"]= [identity, ch3s33u08, ch3_o4_s3_d1_Q_N, ch3_o0_s2_d1_Q_N] #identity, ch3s33u08, ch3o5s3QT, ch3o0s3QT
network.layers[3].sfa_args["max_preserved_sfa"]= 1.96 #1.955

network.layers[4].sfa_args["expansion_funcs"]= [identity, ch3s45u08, ch3_o4_s4_d1_Q_N, ] #identity, ch3s45u08, ch3o5s4QT
network.layers[4].sfa_args["max_preserved_sfa"]= 1.9625 #1.9625 

network.layers[5].sfa_args["expansion_funcs"]= [identity, ch3s50u08, ch3o4s3QT] #identity, ch3s50u08, ch3o5s3QT
network.layers[5].sfa_args["max_preserved_sfa"]= 1.9625 #1.9625 

network.layers[6].sfa_args["expansion_funcs"]= [identity, ch3s40u08, ch3o4s5QT,] #identity, ch3s40u08, ch3o5s5QT
network.layers[6].sfa_args["max_preserved_sfa"]= 1.9625 #1.9625 

network.layers[7].sfa_args["expansion_funcs"]= [identity, ch3s40u08, ch3_o5_s5_d1_Q_N, ] #identity, ch3s40u08, ch3o6s5QT/ch3_o6_s5_Q_N,
network.layers[7].sfa_args["max_preserved_sfa"]= 1.9625 #1.9625 

network.layers[8].sfa_args["expansion_funcs"]= [identity, ch3s30u08, ch3_o5_s4_d1_Q_N, ch3_o0_s2_d1_Q_N] #identity, ch3s30u08, ch3o6s4QT/ch3_o6_s4_d1_Q_N, ch3o0s3QT/ch3_o0_s3_d1_Q_N
network.layers[8].sfa_args["max_preserved_sfa"]= 1.9625 #1.9625 

double_SFA_top_node = True #and False
if double_SFA_top_node:
    print "adding additional top node (two nodes in layer 8)"
    layer = network.layers[8]
    #NODE 19
    layer.pca_node_class = mdp.nodes.iGSFANode
    layer.pca_out_dim = HiGSFANet_out_dims[8]
    layer.pca_args = dict(layer.sfa_args)
    #NODE 20 / layer[9]
    layer.sfa_node_class = mdp.nodes.iGSFANode
    layer.sfa_out_dim = HiGSFANet_out_dims[9]
    layer.sfa_args = dict(layer.sfa_args)
    layer.sfa_args["expansion_funcs"]= [identity, s40u08ex, o5_s15_d1_Q_N] #identity, s40u08ex, o6s15QT/o6_s15_d1_Q_N



##########################################################################################################################
network = HiGSFANetworkU11L_Overlap4x4L0  = copy.deepcopy(HiGSFANetworkU11L_Overlap6x6L0_1Label)
rec_field_size = 4 #6 => 192x192, 5=> 160x160, 4=>128x128
network.layers[0].x_field_channels = rec_field_size
network.layers[0].y_field_channels = rec_field_size
network.layers[0].x_field_spacing = rec_field_size/2
network.layers[0].y_field_spacing = rec_field_size/2
network.layers[0].pca_out_dim = 16

HiGSFANet4x4_out_dims = [ 16, 24, 33, 60, 72, 80, 80, 75, 75, 50, 75 ]
for i, layer in enumerate(network.layers):
    layer.sfa_out_dim = HiGSFANet4x4_out_dims[i]
    layer.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":None,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.9999} 

network = IEMNetworkU11L = copy.deepcopy(linearPCANetworkU11L)
IEMNet_out_dims = [ 13, 20, 35, 60,60,60,60,60,60,60,60 ]
IEMNet_out_dims = [ 30, 45, 60, 70,70,75,75,60,60,60,60 ]
print "IEMNetworkU11L L0-10_SFA_out_dim = ", IEMNet_out_dims

for i, layer in enumerate(network.layers):
    layer.sfa_node_class = mdp.nodes.iGSFANode
    layer.sfa_out_dim = IEMNet_out_dims[i]
 
#    layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "use_pca":True, "max_lenght_slow_part":1, "max_num_samples_for_ev":800, "max_test_samples_for_ev":200, "k":8}
#    layer.sfa_args = {"expansion_funcs":None, "use_pca":True, "max_lenght_slow_part":10, "max_num_samples_for_ev":800, "max_test_samples_for_ev":800, "k":92}
#    layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "use_pca":True, "operation":"average", "max_lenght_slow_part":10, "max_num_samples_for_ev":1200, "max_test_samples_for_ev":1200, "k":92}
#    layer.sfa_args = {"expansion_funcs":None, "use_pca":True, "use_sfa":True, "operation":"average", "max_lenght_slow_part":10, "max_num_samples_for_ev":600, "max_test_samples_for_ev":600, "k":92}
##    layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "use_pca":True, "use_sfa":True, "operation":"average", "max_lenght_slow_part":10, "max_num_samples_for_ev":400, "max_test_samples_for_ev":400, "k":48, "max_preserved_sfa":2.0}
################## Default:
#    layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "use_pca":True, "use_sfa":True, "operation":"average", "max_lenght_slow_part":10, "max_num_samples_for_ev":600, "max_test_samples_for_ev":600, "k":48, "max_preserved_sfa":2.0, "out_sfa_filter":False}
################## Reconstruction only
    layer.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":None,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99} #2.0
#    layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "use_pca":True, "use_sfa":False, "operation":"average", "max_lenght_slow_part":20, "max_num_samples_for_ev":1200, "max_test_samples_for_ev":1200, "k":92, "max_preserved_sfa":1.99, "out_sfa_filter":False}
#    layer.sfa_args = {"expansion_funcs":None, "use_pca":True, "operation":"lin_app", "max_lenght_slow_part":10, "max_num_samples_for_ev":1200, "max_test_samples_for_ev":1200, "k":200}
#    layer.sfa_args = {"expansion_funcs":None, "use_pca":True, "max_lenght_slow_part":6, "max_num_samples_for_ev":600, "max_test_samples_for_ev":600, "k":16}
network.layers[0].pca_node_class = mdp.nodes.PCANode
network.layers[0].pca_out_dim = 30 #adsfa 

print "*******************************************************************"
print "******** Creating Non-Linear Ultra Thin 11L SFA Network ******************"
print "*******************************************************************"
#Warning, this is based on the linear network, thus modifications to the linear 
#network also affect this non linear network
#exp_funcs = [identity, pair_prod_ex, pair_prod_mix1_ex, pair_prod_adj2_ex, pair_prod_adj3_ex]

#layer_exp_funcs = range(11)
#layer_exp_funcs[0] = [identity, pair_prod_mix1_ex]
#layer_exp_funcs[1] = [identity, pair_prod_mix1_ex]
#layer_exp_funcs[2] = [identity, pair_prod_mix1_ex]
#layer_exp_funcs[3] = [identity, pair_prod_mix1_ex]
#layer_exp_funcs[4] = [identity, pair_prod_mix1_ex]
#layer_exp_funcs[5] = [identity, pair_prod_adj2_ex]
#layer_exp_funcs[6] = [identity, pair_prod_adj2_ex]
#layer_exp_funcs[7] = [identity, pair_prod_adj2_ex]
#layer_exp_funcs[8] = [identity, pair_prod_adj2_ex]
#layer_exp_funcs[9] = [identity, pair_prod_adj2_ex]
#layer_exp_funcs[10] = [identity, pair_prod_adj2_ex]


layer = pSFAULayerNL0 = copy.deepcopy(pSFAULayerL0)
layer.exp_funcs = [identity, unsigned_08expo_m1, unsigned_08expo_p1]
#layer.red_node_class = mdp.nodes.WhiteningNode
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros((1,layer.pca_out_dim)))[0])-2

layer = pSFAULayerNL1_H = copy.deepcopy(pSFAULayerL1_H)
#layer.ord_node_class = more_nodes.RandomPermutationNode
#layer.exp_funcs = [identity, pair_prod_adj1_ex, unsigned_08expo, _mix2_ex, weird_sig, signed_sqrt_pair_prod_mix3_ex, e_neg_sqr]
layer.exp_funcs = [identity, unsigned_08expo_m1, unsigned_08expo_p1]
#layer.red_node_class = mdp.nodes.WhiteningNode
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros((1,layer.pca_out_dim)))[0])-2

layer = pSFAULayerNL1_V = copy.deepcopy(pSFAULayerL1_V)
#layer.ord_node_class = more_nodes.RandomPermutationNode
#layer.exp_funcs = [identity, unsigned_08expo]
layer.exp_funcs = [identity, unsigned_08expo_m1, unsigned_08expo_p1]
#layer.red_node_class = mdp.nodes.WhiteningNode
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros((1,layer.pca_out_dim)))[0])-2

layer = pSFAULayerNL2_H = copy.deepcopy(pSFAULayerL2_H)
#layer.ord_node_class = more_nodes.RandomPermutationNode
#layer.exp_funcs = [identity, pair_prod_adj1_ex]
layer.exp_funcs = [identity, unsigned_08expo_m1, unsigned_08expo_p1]
#layer.red_node_class = mdp.nodes.WhiteningNode
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros((1,layer.pca_out_dim)))[0])-2

layer = pSFAULayerNL2_V = copy.deepcopy(pSFAULayerL2_V)
#layer.ord_node_class = more_nodes.RandomPermutationNode
#layer.exp_funcs = [identity, pair_prod_adj1_ex]
layer.exp_funcs = [identity, unsigned_08expo_m1, unsigned_08expo_p1]
#layer.red_node_class = mdp.nodes.WhiteningNode
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros((1,layer.pca_out_dim)))[0])-2

layer = pSFAULayerNL3_H = copy.deepcopy(pSFAULayerL3_H)
#layer.ord_node_class = more_nodes.RandomPermutationNode
#layer.exp_funcs = [identity, pair_prod_adj1_ex]
layer.exp_funcs = [identity, unsigned_08expo_m1, unsigned_08expo_p1]
#layer.red_node_class = mdp.nodes.WhiteningNode
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros((1,layer.pca_out_dim)))[0])-2

layer = pSFAULayerNL3_V = copy.deepcopy(pSFAULayerL3_V)
#layer.ord_node_class = more_nodes.RandomPermutationNode
#layer.exp_funcs = [identity, pair_prod_adj1_ex]
layer.exp_funcs = [identity, unsigned_08expo_m1, unsigned_08expo_p1]
#layer.red_node_class = mdp.nodes.WhiteningNode
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros((1,layer.pca_out_dim)))[0])-2

layer = pSFAULayerNL4_H = copy.deepcopy(pSFAULayerL4_H)
#layer.ord_node_class = more_nodes.RandomPermutationNode
#layer.exp_funcs = [identity, pair_prod_adj1_ex, pair_prod_ex]
layer.exp_funcs = [identity, unsigned_08expo_m1, unsigned_08expo_p1]
#layer.red_node_class = mdp.nodes.WhiteningNode
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros((1,layer.pca_out_dim)))[0])-2

layer = pSFAULayerNL4_V = copy.deepcopy(pSFAULayerL4_V)
#layer.ord_node_class = more_nodes.RandomPermutationNode
#halbs_multiply_ex, pair_prod_adj1_ex
#layer.exp_funcs = [identity, halbs_multiply_ex, e_neg_sqr_exp]
layer.exp_funcs = [identity, unsigned_08expo_m1, unsigned_08expo_p1]
#layer.red_node_class = mdp.nodes.WhiteningNode
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros((1,layer.pca_out_dim)))[0])-2

layer = pSFAULayerNL5_H = copy.deepcopy(pSFAULayerL5_H)
#layer.ord_node_class = more_nodes.RandomPermutationNode
#layer.exp_funcs = [identity, pair_prod_adj1_ex]
layer.exp_funcs = [identity, unsigned_08expo_m1, unsigned_08expo_p1]
#layer.red_node_class = mdp.nodes.WhiteningNode
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros((1,layer.pca_out_dim)))[0])-2

layer = pSFAULayerNL5_V = copy.deepcopy(pSFAULayerL5_V)
#layer.ord_node_class = more_nodes.RandomPermutationNode
#layer.exp_funcs = [identity, pair_prod_adj1_ex]
layer.exp_funcs = [identity, unsigned_08expo_m1, unsigned_08expo_p1]
#layer.red_node_class = mdp.nodes.WhiteningNode
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros((1,layer.pca_out_dim)))[0])-2


                    
####################################################################
###########           THIN Non-LINEAR NETWORK               ############
####################################################################  
network = nonlinearNetworkU11L = system_parameters.ParamsNetwork()
network.name = "Non-Linear Ultra Thin 11 Layer Network"

network.L0 = pSFAULayerNL0
network.L1 = pSFAULayerNL1_H
network.L2 = pSFAULayerNL1_V
network.L3 = pSFAULayerNL2_H
network.L4 = pSFAULayerNL2_V
network.L5 = pSFAULayerNL3_H
network.L6 = pSFAULayerNL3_V
network.L7 = pSFAULayerNL4_H
network.L8 = pSFAULayerNL4_V
network.L9 = pSFAULayerNL5_H
network.L10 = pSFAULayerNL5_V

network.layers = [network.L0, network.L1, network.L2, network.L3, network.L4, network.L5, network.L6, network.L7, \
                  network.L8, network.L9, network.L10]

#TODO: Correct bug in adj_k product, it should also work for very small input channels, like 2x2
network = TestNetworkU11L = system_parameters.ParamsNetwork()
network.name = "Test Non-Linear 11 Layer Thin Network"
network.L0 = pSFAULayerNL0
network.L1 = pSFAULayerNL1_H
network.L2 = pSFAULayerNL1_V
network.L3 = pSFAULayerNL2_H
network.L4 = pSFAULayerNL2_V
network.L5 = pSFAULayerNL3_H
network.L6 = pSFAULayerNL3_V
network.L7 = pSFAULayerNL4_H
network.L8 = pSFAULayerNL4_V
network.L9 = pSFAULayerNL5_H
network.L10 = pSFAULayerNL5_V
network.layers = [network.L0, network.L1, network.L2, network.L3, network.L4, network.L5, network.L6, network.L7, \
                  network.L8, network.L9, network.L10]


#


#u08expoNetworkU11L
u08expoNetworkU11L = NetworkSetExpFuncs([identity, unsigned_08expo], copy.deepcopy(nonlinearNetworkU11L))
u08expoNetworkU11L.L0.pca_node_class = mdp.nodes.PCANode
u08expoNetworkU11L.L0.pca_out_dim = 13 
rec_field_size = 4 #6 => 192x192, 5=> 160x160, 4=>128x128
u08expoNetworkU11L.L0.x_field_channels = rec_field_size
u08expoNetworkU11L.L0.y_field_channels = rec_field_size
u08expoNetworkU11L.L0.x_field_spacing = rec_field_size
u08expoNetworkU11L.L0.y_field_spacing = rec_field_size
u08expoNetworkU11L.L0.sfa_out_dim = 13 # was 13
u08expoNetworkU11L.L1.sfa_out_dim = 20 # was 20
u08expoNetworkU11L.L2.sfa_out_dim = 35 # was 35
u08expoNetworkU11L.L3.sfa_out_dim = 60 # was 60
u08expoNetworkU11L.L4.sfa_out_dim = 60 # was 60
u08expoNetworkU11L.L5.sfa_out_dim = 60 # was 60   
u08expoNetworkU11L.L6.sfa_out_dim = 60 # was 60     
u08expoNetworkU11L.L7.sfa_out_dim = 60 # was 60
u08expoNetworkU11L.L8.sfa_out_dim = 60 # was 60     
u08expoNetworkU11L.L9.sfa_out_dim = 60

u08expoNetworkU11L_128x128 = copy.deepcopy(u08expoNetworkU11L)
u08expoNetworkU11L_64x64 = copy.deepcopy(u08expoNetworkU11L)
u08expoNetworkU11L_64x64.layers = u08expoNetworkU11L_64x64.layers[:-2]
u08expoNetworkU11L_32x32 = copy.deepcopy(u08expoNetworkU11L)
u08expoNetworkU11L_32x32.layers = u08expoNetworkU11L_32x32.layers[:-4]

#******************** GENDER NETWORK FOR ARTICLE ****************
layer = Gender_top_layer = system_parameters.ParamsSFASuperNode()
layer.name = "top layer"
#layer.pca_node_class = None # mdp.nodes.GSFANode
#W
layer.pca_node_class = None
#layer.pca_args = {}
#layer.pca_out_dim = 35 #WARNING: 100 or None
#layer.pca_out_dim = 4000 # 35 #WARNING: 100 or None
#layer.exp_funcs = [identity, QT]
layer.exp_funcs = [identity, unsigned_08expo] #unsigned_08expo, signed_08expo
#layer.exp_funcs = [encode_signal_p9,] #For next experiment: [encode_signal_p9,]
#layer.red_node_class = mdp.nodes.HeadNode
#layer.red_out_dim = int(tuning_parameter)
layer.sfa_node_class =  mdp.nodes.GSFANode #mdp.nodes.GSFANode #mdp.nodes.GSFANode
layer.sfa_out_dim = 10 #3 #49*2 # *3 # None


network = NetworkGender_8x8L0 = copy.deepcopy(u08expoNetworkU11L)
network.L0.pca_out_dim = 50 
rec_field_size = 8 #6 => 192x192, 5=> 160x160, 4=>128x128
network.L0.x_field_channels = rec_field_size
network.L0.y_field_channels = rec_field_size
network.L0.x_field_spacing = rec_field_size
network.L0.y_field_spacing = rec_field_size
network.L0.sfa_out_dim = 40 # was 13
network.L1.sfa_out_dim = 40 # was 20
network.L2.sfa_out_dim = 40 # was 35
network.L3.sfa_out_dim = 40 # was 60
network.L4.sfa_out_dim = 40 # was 60
network.L5.sfa_out_dim = 40 # was 60   
network.L6.sfa_out_dim = 40 # was 60     
network.L7 = Gender_top_layer
network.L8 = None 
network.L9 = None
network.layers = [network.L0, network.L1, network.L2, network.L3, network.L4, network.L5, network.L6, network.L7]
#NetworkGender_8x8L0 = NetworkSetExpFuncs([identity, unsigned_08expo, signed_08expo], copy.deepcopy(NetworkGender_8x8L0))
NetworkGender_8x8L0 = NetworkSetExpFuncs([identity, unsigned_08expo], copy.deepcopy(NetworkGender_8x8L0))
#NetworkGender_8x8L0.L6.exp_funcs = [identity, QT]
#NetworkGender_8x8L0.L7.exp_funcs = [identity, QT] #, CT, P4, P5, P6]

#WARNING, TUNING FOR AGE EXPERIMENTS ONLY, BREAKS NETWORKS DERIVED FROM IT!
network = u08expoNetworkU11L_5x5L0 = copy.deepcopy(u08expoNetworkU11L)
network.L0.pca_out_dim = 20 #was 15?
rec_field_size = 5 #6 => 192x192, 5=> 160x160, 4=>128x128
network.L0.x_field_channels = rec_field_size
network.L0.y_field_channels = rec_field_size
network.L0.x_field_spacing = rec_field_size
network.L0.y_field_spacing = rec_field_size
network.L0.sfa_out_dim = 20 #was 13
network.L1.sfa_out_dim = 30 #was 20
network.L2.sfa_out_dim = 43 #was 35
network.L3.sfa_out_dim = 70 #was 60
network.L4.sfa_out_dim = 82 # 70 or 72 was 60
network.L5.sfa_out_dim = 76 # was 60   
network.L6.sfa_out_dim = 84 # was 60     
network.L7.sfa_out_dim = 82 # was 60
network.L8.sfa_out_dim = 75 # was 60     
network.L9.sfa_out_dim = 70
#Original run for Age experiments: 15, 20, 33, 65, 70, 70, 70, 70, 80
#10 Percent larger:                16, 22, 36, 72, 77, 77, 77, 77, 80
#20 Percent larger:                18, 24, 39, 78, 84, 84, 84, 84, 80
#30 Percent larger:                19, 26, 43, 85, 85, 85, 85, 85, 80
#40 Percent larger:                20, 28, 46, 90, 90, 90, 90, 90, 80
#m10 Percent Smaller:              14, 18, 30, 59, 63, 63, 63, 63, 80
#many1 PF:                          20, 30, 50, 90, 90, 90, 90, 90, 80 
#many2 PF:                          20, 32, 54, 90, 90, 90, 90, 90, 80
#smart try:                        17, 24, 37, 70, 75, 75, 75, 75, 80

u08expoNetwork32x32U11L_NoTop = copy.deepcopy(u08expoNetworkU11L)
network = u08expoNetwork32x32U11L_NoTop
layer = network.L6
layer.exp_funcs = [identity,]
layer.sfa_node_class = mdp.nodes.IdentityNode
layer.sfa_out_dim = 120
layer.sfa_args = {}

network.L7 = None
network.L8 = None
network.L9 = None
network.L10 = None
network.layers = [network.L0,network.L1,network.L2,network.L3,network.L4,network.L5, 
                  network.L6,network.L7,network.L8,network.L9,network.L10]

SFAAdaptiveNLNetwork32x32U11L = copy.deepcopy(u08expoNetworkU11L)
network = SFAAdaptiveNLNetwork32x32U11L
for layer in [network.L6,network.L7,network.L8,network.L9,network.L10]:
    layer.exp_funcs = [identity,]
    layer.sfa_node_class = mdp.nodes.SFAAdaptiveNLNode
    layer.sfa_args = {"pre_expansion_node_class":None, "final_expanded_dim":240, "initial_expansion_size":240, "starting_point":"08Exp", "expansion_size_decrement":150, "expansion_size_increment":155, "number_iterations":6}
    # SFAAdaptiveNLNode( input_dim = 10, output_dim=10, pre_expansion_node_class = None, final_expanded_dim = 20, initial_expansion_size = 15, expansion_size_decrement = 5, expansion_size_increment = 10, number_iterations=3)

HardSFAPCA_u08expoNetworkU11L = copy.deepcopy(u08expoNetworkU11L)
HardSFAPCA_u08expoNetworkU11L.L0.pca_node_class = mdp.nodes.SFAPCANode
NetworkSetSFANodeClass(mdp.nodes.SFAPCANode, HardSFAPCA_u08expoNetworkU11L)

GTSRB_Network2 = copy.deepcopy(HardSFAPCA_u08expoNetworkU11L)
GTSRB_Network2.pca_output_dim = 43


#identity, unsigned_08expo, pair_prod_ex
HeuristicEvaluationExpansionsNetworkU11L = copy.deepcopy(u08expoNetworkU11L)
HeuristicEvaluationExpansionsNetworkU11L.L2.sfa_out_dim=30
HeuristicEvaluationExpansionsNetworkU11L.L3.sfa_out_dim=30
HeuristicEvaluationExpansionsNetworkU11L.L4.sfa_out_dim=30
HeuristicEvaluationExpansionsNetworkU11L.L5.sfa_out_dim=30
HeuristicEvaluationExpansionsNetworkU11L.L6.sfa_out_dim=30

identity
Q_d2_L = [identity, pair_prod_ex]
C_d3_L = C_L(k=nan, d=3.0)
Q_N_k1_d2_L = Q_N_L(k=1.0, d=2.0)
#C_N_k1_d2_L = C_N_L(k=1.0, d=2.0) #WARNING, should change everywhere to d=3.0 ad
C_N_k1_d3_L = C_N_L(k=1.0, d=3.0) 
Q_d08_L = Q_L(k=nan, d=0.8)
C_d09_L = C_L(k=nan, d=0.9)
S_d08_L = [identity, unsigned_08expo]
S_d2_L = S_L(k=nan, d=2.0)

HeuristicEvaluationExpansionsNetworkU11L = NetworkSetExpFuncs(S_d08_L, HeuristicEvaluationExpansionsNetworkU11L)
#
#HeuristicEvaluationExpansionsNetworkU11L.L0.exp_funcs = [identity,]
#HeuristicEvaluationExpansionsNetworkU11L.L1.exp_funcs = [identity,]
#HeuristicEvaluationExpansionsNetworkU11L.L2.exp_funcs = [identity,]
#HeuristicEvaluationExpansionsNetworkU11L.L3.exp_funcs = [identity,]
#HeuristicEvaluationExpansionsNetworkU11L.L4.exp_funcs = [identity,]
#HeuristicEvaluationExpansionsNetworkU11L.L5.exp_funcs = [identity,]
#HeuristicEvaluationExpansionsNetworkU11L.L6.exp_funcs = [identity,]

#HeuristicEvaluationExpansionsNetworkU11L.L0.sfa_out_dim=30
#HeuristicEvaluationExpansionsNetworkU11L.L1.exp_funcs = [identity, pair_prod_ex]

#32x32. L0 normal, L1 pair_prod_ex, rest linear. @8: 19.364, 24.672, 25.183, typical_delta_train= [ 0.25289074  0.32593131, typical delta_newid= [ 0.25795559  0.32364028
#32x32. L0 pair_prod_ex, rest linear. @8: 93.007, 104.805, 95.660, typical_delta_train= [ 0.58488899  0.63026202, typical delta_newid= [ 0.58985147  0.61964849

#32x32, modified, pair_prod_ex: L5lin, @8: 23.342,41.346,40.187, L4lin, @8:24.849,41.552,39.605,
#MSE_Gauss 23.287,43.627,40.215,typical_delta_train= [ 0.21459941  0.29260883, typical delta_newid= [ 0.24974733  0.3316066
#L1lin, @8: 36.657,63.277,61.612,typical_delta_train= [ 0.32495874  0.35448157, typical delta_newid= [ 0.37454736  0.40719719
#L0lin, @8: 218.509, 304.338, 271.035, typical_delta_train= [ 0.80577574  0.82477724, typical delta_newid= [ 1.02751328  1.01734513

#32x32, modified, u08expo: L5lin, @8: 43.508,47.628,46.450 L4lin, @8: 36.729,36.951,36.405,
#MSE_Gauss 32.966, 38.140, 36.703, typical_delta_train= [ 0.27342929  0.36033502, typical delta_newid= [ 0.28085104  0.36441605
#L1lin, @8: 61.620, 59.924, 56.255, typical_delta_train= [ 0.38094301  0.40660367, typical delta_newid= [ 0.3734798   0.39011842
#L0lin, @8: 332.649, 348.934, 315.407, typical_delta_train= [ 0.38094301  0.40660367, typical delta_newid= [ 1.04749017  1.01911393

#32x32, modified, no expo: L5lin, @8: 33.098,36.273,34.123 L4lin, @8: 33.829,34.987,34.537,
#MSE_Gauss 31.616, 36.097, 34.425 L4lin, @8: typical_delta_train= [ 0.28812816  0.3856025, typical delta_newid= [ 0.29035496  0.3779304
#L3lin, @8: 35.585,34.781,34.281, 
#L2lin, @8: 36.345, 36.105, 34.371, typical_delta_train= [ 0.30616905  0.40299957, typical delta_newid= [ 0.30480867  0.38950811
#L1lin, @8: 56.887, 56.034, 50.738, typical_delta_train= [ 0.38377317  0.42976021, typical delta_newid= [ 0.3742282   0.41215822 
#L0lin, @8: 332.513, 344.689, 317.466, typical_delta_train= [ 1.02527302  1.11769401, typical delta_newid= [ 1.05743285  1.11296883 

#32x32, fully original:
#New Id: 0.000 CR_CCC, CR_Gauss 0.000, CR_SVM 0.000, MSE_CCC 46.222, MSE_Gauss 36.007, MSE3_SVM 675.150, MSE2_SVM 675.150, MSE_SVM 675.150, MSE_LR 108.970 

#u08expoNetworkU11L.L0.exp_funcs = [identity]
#u08expoNetworkU11L.L0.sfa_node_class = mdp.nodes.PCANode

#WARNING, TUNING for GTSRB here!!!!
#u08expoNetworkU11L.L0.pca_node_class = None
#u08expoNetworkU11L.L0.exp_funcs = [identity]
#u08expoNetworkU11L.L0.sfa_node_class = mdp.nodes.HeadNode
#u08expoNetworkU11L.L0.sfa_node_class = mdp.nodes.PCANode

#[identity, sel_exp(42, unsigned_08expo)]
u08expoS42NetworkU11L = NetworkSetExpFuncs([identity, sel_exp(42, unsigned_08expo)], copy.deepcopy(nonlinearNetworkU11L))
u08expoS42NetworkU11L.L0.pca_node_class = mdp.nodes.PCANode
u08expoS42NetworkU11L.L1.pca_out_dim = (39)/3
#u08expoS42NetworkU11L.L0.ord_node_class = mdp.nodes.GSFANode
#u08expoS42NetworkU11L.L0.exp_funcs = [identity, unsigned_08expo]
u08expoS42NetworkU11L.L0.sfa_node_class = mdp.nodes.HeadNode #mdp.nodes.GSFANode
u08expoS42NetworkU11L.L0.sfa_out_dim = 78/3 #No dim reduction!

u08expoS42NetworkU11L.L1.pca_node_class = mdp.nodes.GSFANode
u08expoS42NetworkU11L.L1.pca_out_dim = (55)/2
u08expoS42NetworkU11L.L1.sfa_node_class = mdp.nodes.HeadNode
u08expoS42NetworkU11L.L1.sfa_out_dim = (96)/2

u08expoS42NetworkU11L.L2.pca_node_class = mdp.nodes.GSFANode
u08expoS42NetworkU11L.L2.pca_out_dim = (55)/1.5
u08expoS42NetworkU11L.L2.sfa_node_class = mdp.nodes.HeadNode
u08expoS42NetworkU11L.L2.sfa_out_dim = (96)/1.5

u08expoS42NetworkU11L.L3.pca_node_class = mdp.nodes.GSFANode
u08expoS42NetworkU11L.L3.pca_out_dim = (55)
u08expoS42NetworkU11L.L3.sfa_node_class = mdp.nodes.HeadNode
u08expoS42NetworkU11L.L3.sfa_out_dim = (97)

u08expoS42NetworkU11L.L4.pca_node_class = mdp.nodes.GSFANode
u08expoS42NetworkU11L.L4.pca_out_dim = (42)
u08expoS42NetworkU11L.L4.sfa_node_class = mdp.nodes.HeadNode
u08expoS42NetworkU11L.L4.sfa_out_dim = (84)

u08expoS42NetworkU11L.L5.pca_node_class = mdp.nodes.GSFANode
u08expoS42NetworkU11L.L5.pca_out_dim = (42)
u08expoS42NetworkU11L.L5.exp_funcs = [identity]
u08expoS42NetworkU11L.L5.sfa_node_class = mdp.nodes.HeadNode
u08expoS42NetworkU11L.L5.sfa_out_dim = (42)

#W
u08expoS42NetworkU11L.L6.pca_node_class = None
#u08expoS42NetworkU11L.L6.pca_out_dim = (55)
#W
u08expoS42NetworkU11L.L6.exp_funcs = [identity]
u08expoS42NetworkU11L.L6.sfa_node_class = mdp.nodes.GSFANode
u08expoS42NetworkU11L.L6.sfa_out_dim = (84)

#u08expoS42NetworkU11L.L7 = None
#u08expoS42NetworkU11L.L8 = None
#u08expoS42NetworkU11L.L9 = None
#u08expoS42NetworkU11L.L10 = None



u08expoA2NetworkU11L = NetworkSetExpFuncs([identity, pair_prodsigmoid_04_adj2_ex], copy.deepcopy(nonlinearNetworkU11L))
u08expoA3NetworkU11L = NetworkSetExpFuncs([identity, pair_prodsigmoid_04_adj3_ex], copy.deepcopy(nonlinearNetworkU11L))
u08expoA4NetworkU11L = NetworkSetExpFuncs([identity, pair_prodsigmoid_04_adj4_ex], copy.deepcopy(nonlinearNetworkU11L))

u08expo_m1p1_NetworkU11L = NetworkSetExpFuncs([identity, unsigned_08expo_m1, unsigned_08expo_p1], copy.deepcopy(nonlinearNetworkU11L))
u2_08expoNetworkU11L = NetworkSetExpFuncs([identity, unsigned_2_08expo], copy.deepcopy(nonlinearNetworkU11L), include_L0=False)

#1.15
u08expo_pcasfaexpo_NetworkU11L = NetworkSetPCASFAExpo(copy.deepcopy(u08expoNetworkU11L), first_pca_expo=0.0, last_pca_expo=1.0, first_sfa_expo=2.0, last_sfa_expo=1.0, hard_pca_expo=False)
#0.9, 1.2, False
u08expo_pcasfaexpo_NetworkU11L.layers[4].exp_funcs = [identity]
u08expo_pcasfaexpo_NetworkU11L.layers[3].exp_funcs = [identity]
u08expo_pcasfaexpo_NetworkU11L.layers[2].exp_funcs = [identity]
u08expo_pcasfaexpo_NetworkU11L.layers[1].exp_funcs = [identity]
u08expo_pcasfaexpo_NetworkU11L.layers[0].exp_funcs = [identity]

#u08expo_pcasfaexpo_NetworkU11L.layers[4].exp_funcs = [identity, unsigned_08expo]
#u08expo_pcasfaexpo_NetworkU11L.layers[3].exp_funcs = [identity, pair_prodsigmoid_04_adj2_ex]
#u08expo_pcasfaexpo_NetworkU11L.layers[2].exp_funcs = [identity, pair_prodsigmoid_04_adj2_ex]
#u08expo_pcasfaexpo_NetworkU11L.layers[1].exp_funcs = [identity, unsigned_08expo]
#u08expo_pcasfaexpo_NetworkU11L.layers[0].exp_funcs = [identity, unsigned_08expo]


u08expoA2_pcasfaexpo_NetworkU11L = NetworkSetPCASFAExpo(copy.deepcopy(u08expoA2NetworkU11L), first_pca_expo=0.0, last_pca_expo=1.0, first_sfa_expo=1.15, last_sfa_expo=1.0, hard_pca_expo=True)
u08expoA3_pcasfaexpo_NetworkU11L = NetworkSetPCASFAExpo(copy.deepcopy(u08expoA3NetworkU11L), first_pca_expo=0.8, last_pca_expo=1.0, first_sfa_expo=1.05, last_sfa_expo=1.0, hard_pca_expo=False)
u08expoA4_pcasfaexpo_NetworkU11L = NetworkSetPCASFAExpo(copy.deepcopy(u08expoA4NetworkU11L), first_pca_expo=1.0, last_pca_expo=1.0, first_sfa_expo=1.0, last_sfa_expo=1.0, hard_pca_expo=False)

#WARNING, ORIGINAL:
#u08expo_pcasfaexpo_NetworkU11L = NetworkSetPCASFAExpo(copy.deepcopy(u08expoNetworkU11L), first_pca_expo=0.0, last_pca_expo=1.0, first_sfa_expo=1.15, last_sfa_expo=1.0, hard_pca_expo=False)

experimentalNetwork = NetworkSetExpFuncs([identity, unsigned_2_08expo, pair_prodsigmoid_04_02_mix1_ex], copy.deepcopy(nonlinearNetworkU11L), include_L0=False) 


network = TestNetworkPCASFAU11L = copy.deepcopy(linearNetworkU11L)
network.name = "Test Non-Linear 11 Layer PCA/SFA Thin Network"
network.L0 = copy.deepcopy(pSFAULayerL0)
network.L0.pca_node_class = None
network.L0.pca_args = {}
network.L0.ord_node_class = None
network.L0.ord_args = {}     
network.L0.red_node_class = None
network.L0.red_args = {}
network.L0.sfa_node_class = mdp.nodes.GSFANode
network.L0.sfa_args = {}
network.L0.cloneLayer = True

network.L1 = copy.deepcopy(pSFAULayerL1_H)
network.L1.pca_node_class = None
network.L1.pca_args = {}
network.L1.ord_node_class = None
network.L1.ord_args = {}     
network.L1.red_node_class = None
network.L1.red_args = {}
network.L1.sfa_node_class = mdp.nodes.GSFANode
network.L1.sfa_args = {}
network.L1.cloneLayer = True

network.L2 = copy.deepcopy(pSFAULayerL1_V)
network.L2.pca_node_class = None
network.L2.pca_args = {}
network.L2.ord_node_class = None
network.L2.ord_args = {}     
network.L2.red_node_class = None
network.L2.red_args = {}
network.L2.sfa_node_class = mdp.nodes.GSFANode
network.L2.sfa_args = {}
network.L2.cloneLayer = True

network.L3 = copy.deepcopy(pSFAULayerL2_H)
network.L3.pca_node_class = None
network.L3.pca_args = {}
network.L3.ord_node_class = None
network.L3.ord_args = {}     
network.L3.red_node_class = None
network.L3.red_args = {}
network.L3.sfa_node_class = mdp.nodes.GSFANode
network.L3.sfa_args = {}
network.L3.cloneLayer = False

network.L4 = copy.deepcopy(pSFAULayerL2_V)
network.L4.pca_node_class = None
network.L4.pca_args = {}
network.L4.ord_node_class = None
network.L4.ord_args = {}     
network.L4.red_node_class = None
network.L4.red_args = {}
network.L4.sfa_node_class = mdp.nodes.GSFANode
network.L4.sfa_args = {}
network.L4.cloneLayer = False



#Network Useful for the face detection experiments
network = HiGSFANetworkU11L_NoOverlap_4x4L0 = copy.deepcopy(u08expoNetworkU11L)
rec_field_size = 4 #6 => 192x192, 5=> 160x160, 4=>128x128
LRec_use_RGB_images = False #or True
network.layers[0].x_field_channels = rec_field_size
network.layers[0].y_field_channels = rec_field_size
network.layers[0].x_field_spacing = rec_field_size
network.layers[0].y_field_spacing = rec_field_size
network.layers[0].pca_node_class = mdp.nodes.PCANode
#if LRec_use_RGB_images:
#    network.layers[0].pca_out_dim = 50 #50 Problem, more color components also remove higher frequency ones = code image as intensity+color!!!
#else:
#    network.layers[0].pca_out_dim = 20 #20 #30 #28 #20 #22 more_feats2, 50 for RGB

#if LRec_use_RGB_images == False:
#    HiGSFANet_out_dims = [ 28, 52, 75, 85, 90, 90, 85, 100, 75,75,75 ] #Experiment with control2: Dt=1.9999
#    print "Number of network features set without RGB:", HiGSFANet_out_dims
#else:
#    HiGSFANet_out_dims = [ 39, 51, 65,70,70,70,70,70,70,70,70 ] #(less features)
#    print "Adjusting number of network features due to RGB:", HiGSFANet_out_dims

print "HiGSFANetworkU11L_NoOverlap_4x4L0 L0-10_SFA_out_dim = ", HiGSFANet_out_dims

for i, layer in enumerate(network.layers):
    layer.exp_funcs = [identity]
    layer.sfa_node_class = mdp.nodes.iGSFANode
#    layer.sfa_out_dim = iGSFANet_out_dims[i]
    #QT_AP_L(k=nan, d=0.8), Q_AN_exp
    #layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":10,   "max_preserved_sfa":2.0}
#    layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":1,   "offsetting_mode":"sensitivity_based", "max_preserved_sfa":2.0}
#    layer.sfa_args = {"pre_expansion_node_class":mdp.nodes.GSFANode, "expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":2.0}
    layer.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo], "max_lenght_slow_part":None,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.999} 



delta_thresholds = [1.998, 1.996, 1.995, 1.99, 1.9855, 1.976, 1.97, 1.95, 1.94, 1.93, 1.92, 1.91]
output_dims =      [   16,    28,    50,    75,    75,    75,   75,   75,   75,   75,   80,   80]
network = HiGSFANetworkU11L_NoOverlap_4x4L0_mod = copy.deepcopy(HiGSFANetworkU11L_NoOverlap_4x4L0)
for i, layer in enumerate(network.layers):
    layer.sfa_args["max_preserved_sfa"] = delta_thresholds[i]
    layer.sfa_out_dim = output_dims[i]

delta_thresholds = [2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
output_dims =      [   15,    28,    50,    75,    75,    75,   75,   75,   75,   75,   80,   80]
network = HiGSFANetworkU11L_NoOverlap_4x4L0_mod2 = copy.deepcopy(HiGSFANetworkU11L_NoOverlap_4x4L0)
for i, layer in enumerate(network.layers):
    layer.sfa_args["max_preserved_sfa"] = delta_thresholds[i]
    layer.sfa_out_dim = output_dims[i]

delta_thresholds = [3, 5, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9]
output_dims =      [   15,    28,    50,    75,    75,    75,   75,   75,   75,   75,   80,   80]
network = HiGSFANetworkU11L_NoOverlap_4x4L0_mod4 = copy.deepcopy(HiGSFANetworkU11L_NoOverlap_4x4L0)
for i, layer in enumerate(network.layers):
    layer.sfa_args["max_preserved_sfa"] = delta_thresholds[i]
    layer.sfa_out_dim = output_dims[i]

####################### Networks FOR POS X, POS Y, SCALE, ANGLE WITHOUT OVERLAP #######################
delta_thresholds = [4, 6, 10, 14, 20, 20, 20, 20, 20, 20, 20, 20] * 12
output_dims =      [   16,    28,    50,    75,    75,    75,   75,   75,   75,   75,   80,   80]
network = HiGSFANetworkU11L_NoOverlap_4x4L0_PosXPosYScaleAngle = copy.deepcopy(HiGSFANetworkU11L_NoOverlap_4x4L0)
for i, layer in enumerate(network.layers):
    layer.sfa_args["max_preserved_sfa"] = delta_thresholds[i]
    layer.sfa_out_dim = output_dims[i]

HiGSFANetworkU11L_NoOverlap_4x4L0_PosXPosYScaleAngle_128x128 = copy.deepcopy(HiGSFANetworkU11L_NoOverlap_4x4L0_PosXPosYScaleAngle)

HiGSFANetworkU11L_NoOverlap_4x4L0_PosXPosYScaleAngle_64x64 = copy.deepcopy(HiGSFANetworkU11L_NoOverlap_4x4L0_PosXPosYScaleAngle)
HiGSFANetworkU11L_NoOverlap_4x4L0_PosXPosYScaleAngle_64x64.layers = HiGSFANetworkU11L_NoOverlap_4x4L0_PosXPosYScaleAngle_64x64.layers[0:-2]

HiGSFANetworkU11L_NoOverlap_4x4L0_PosXPosYScaleAngle_32x32 = copy.deepcopy(HiGSFANetworkU11L_NoOverlap_4x4L0_PosXPosYScaleAngle)
HiGSFANetworkU11L_NoOverlap_4x4L0_PosXPosYScaleAngle_32x32.layers = HiGSFANetworkU11L_NoOverlap_4x4L0_PosXPosYScaleAngle_32x32.layers[0:-2]

####################### NETWORKS FOR EYE DETECTION WITHOUT OVERLAP ####################################
delta_thresholds = [3, 8, 14, 24, 28, 34, 40, 40, 40, 40, 40, 40]
output_dims =      [   16,    28,    50,    75,    75,    75,   75,   75,   75,   75,   80,   80]
network = HiGSFANetworkU11L_NoOverlap_4x4L0_EyeL = copy.deepcopy(HiGSFANetworkU11L_NoOverlap_4x4L0)
for i, layer in enumerate(network.layers):
    layer.sfa_args["max_preserved_sfa"] = delta_thresholds[i]
    layer.sfa_out_dim = output_dims[i]

HiGSFANetworkU11L_NoOverlap_4x4L0_EyeL_128x128 = copy.deepcopy(HiGSFANetworkU11L_NoOverlap_4x4L0_EyeL)

HiGSFANetworkU11L_NoOverlap_4x4L0_EyeL_64x64 = copy.deepcopy(HiGSFANetworkU11L_NoOverlap_4x4L0_EyeL_128x128)
HiGSFANetworkU11L_NoOverlap_4x4L0_EyeL_64x64.layers = HiGSFANetworkU11L_NoOverlap_4x4L0_EyeL_64x64.layers[0:-2]

HiGSFANetworkU11L_NoOverlap_4x4L0_EyeL_32x32 = copy.deepcopy(HiGSFANetworkU11L_NoOverlap_4x4L0_EyeL_64x64)
HiGSFANetworkU11L_NoOverlap_4x4L0_EyeL_32x32.layers = HiGSFANetworkU11L_NoOverlap_4x4L0_EyeL_32x32.layers[0:-2]

####################### NETWORKS FOR EYE DETECTION WITH OVERLAP ####################################
network = HiGSFANetworkU11L_Overlap_4x4L0_EyeL_128x128 = copy.deepcopy(HiGSFANetworkU11L_NoOverlap_4x4L0_EyeL_128x128)
rec_field_size = 4
network.layers[0].x_field_channels = rec_field_size
network.layers[0].y_field_channels = rec_field_size
network.layers[0].x_field_spacing = rec_field_size/2
network.layers[0].y_field_spacing = rec_field_size/2
for i in range(1, len(network.layers)):
    if i%2 == 0: #A vertical layer
        network.layers[i].x_field_channels = 1
        network.layers[i].y_field_channels = 3
        network.layers[i].x_field_spacing = 1
        network.layers[i].y_field_spacing = 2        
    else: #An horizontal layer
        network.layers[i].x_field_channels = 3
        network.layers[i].y_field_channels = 1
        network.layers[i].x_field_spacing = 2
        network.layers[i].y_field_spacing = 1
    
HiGSFANetworkU11L_Overlap_4x4L0_EyeL_64x64 = copy.deepcopy(HiGSFANetworkU11L_Overlap_4x4L0_EyeL_128x128)
HiGSFANetworkU11L_Overlap_4x4L0_EyeL_64x64.layers = HiGSFANetworkU11L_Overlap_4x4L0_EyeL_64x64.layers[0:-2]

HiGSFANetworkU11L_Overlap_4x4L0_EyeL_32x32 = copy.deepcopy(HiGSFANetworkU11L_Overlap_4x4L0_EyeL_64x64)
HiGSFANetworkU11L_Overlap_4x4L0_EyeL_32x32.layers = HiGSFANetworkU11L_Overlap_4x4L0_EyeL_32x32.layers[0:-2]


#WARNING, ADDING AN ADDITIONAL SFA NODE IN THE LAST LAYER, 80x80 resolution (Node 9)
#double_SFA_top_node = True and False
#if double_SFA_top_node:
#    layer = network.layers[8]
#    layer.pca_node_class = mdp.nodes.iGSFANode
#    layer.pca_out_dim = HiGSFANet_out_dims[8]
#    layer.pca_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, div2_sel75_unsigned_08expo], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} #2.0
#    layer.sfa_node_class = mdp.nodes.iGSFANode
#    layer.sfa_out_dim = HiGSFANet_out_dims[8]
#    layer.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, sel8_04QT], "max_lenght_slow_part":10,   "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.99999} #2.0




#
#print "******** Setting Layer L3 k-adj-prod Parameters *********************"
#pSFAL3_L3KadjProd = system_parameters.ParamsSFASuperNode()
##pSFAL3_L3KadjProd.in_channel_dim = pSFALayerL2.sfa_out_dim
#
##pca_node_L3 = mdp.nodes.WhiteningNode(output_dim=pca_out_dim_L3) 
#pSFAL3_L3KadjProd.pca_node_class = mdp.nodes.GSFANode
##pca_out_dim_L3 = 210
##pca_out_dim_L3 = 0.999HiGSFANetworkU11L_NoOverlap_4x4L0
##WARNING!!! CHANGED PCA TO SFA
#pSFAL3_L3KadjProd.pca_out_dim = 300
##pSFALayerL1.pca_args = {"block_size": block_size}
#pSFAL3_L3KadjProd.pca_args = {"block_size": -1, "train_mode": -1}
#
##exp_funcs_L3 = [identity, pair_prod_ex, pair_prod_adj1_ex, pair_prod_adj2_ex, pair_prod_adj3_ex]
#pSFAL3_L3KadjProd.exp_funcs = [identity, pair_prod_adj2_ex]
#pSFAL3_L3KadjProd.inv_use_hint = True
#pSFAL3_L3KadjProd.max_steady_factor=0.35
#pSFAL3_L3KadjProd.delta_factor=0.6
#pSFAL3_L3KadjProd.min_delta=0.0001
#
#pSFAL3_L3KadjProd.red_node_class = mdp.nodes.WhiteningNode
#pSFAL3_L3KadjProd.red_out_dim = 0.999999
#pSFAL3_L3KadjProd.red_args = {}
#
#pSFAL3_L3KadjProd.sfa_node_class = mdp.nodes.GSFANode
##sfa_out_dim_L1 = 12
#pSFAL3_L3KadjProd.sfa_out_dim = 40
#pSFAL3_L3KadjProd.sfa_args = {"block_size": -1, "train_mode": -1}
##Default: cloneLayerL1 = False
#pSFAL3_L3KadjProd.cloneLayer = False
#system_parameters.test_object_contents(pSFAL3_L3KadjProd)
#
#####################################################################
############           NON-LINEAR NETWORK                ############
#####################################################################  
#Network4L = system_parameters.ParamsNetwork()
#Network4L.name = "Linear 4 Layer Network"
#Network4L.L0 = pSFALayerL0
#Network4L.L1 = pSFALayerL1
#Network4L.L2 = pSFALayerL2
#Network4L.L3 = pSFAL3_L3KadjProd

################# Reproduction of a network developed by Fabian Schoenfeld for his Ratlab experiment ############
layer = pSFARatlab_L0 = system_parameters.ParamsSFALayer()
layer.name = "Layer 0 for the Ratlab experiment"
layer.x_field_channels=10
layer.y_field_channels=8
layer.x_field_spacing=5
layer.y_field_spacing=4
layer.pca_node_class = mdp.nodes.iGSFANode
layer.pca_out_dim = 32
layer.pca_args = {} # Specific parameters are defined later
layer.ord_node_class = None
layer.exp_funcs = [identity, ]
layer.red_node_class = None
layer.sfa_node_class = mdp.nodes.iGSFANode
layer.sfa_out_dim = 32
layer.sfa_args = {} # Specific parameters are defined later
layer.cloneLayer = True
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)

layer = pSFARatlab_L1 = system_parameters.ParamsSFALayer()
layer.name = "Layer 1 for the Ratlab experiment"
layer.x_field_channels=14
layer.y_field_channels=6
layer.x_field_spacing=7
layer.y_field_spacing=3
layer.pca_node_class = mdp.nodes.iGSFANode
layer.pca_out_dim = 32
layer.pca_args = {} # Specific parameters are defined later
layer.ord_node_class = None
layer.exp_funcs = [identity, ]
layer.red_node_class = None
layer.sfa_node_class = mdp.nodes.iGSFANode
layer.sfa_out_dim = 32
layer.sfa_args = {} # Specific parameters are defined later
layer.cloneLayer = True
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)

layer = pSFARatlab_L2 = system_parameters.ParamsSFALayer()
layer.name = "Layer 2 for the Ratlab experiment"
layer.x_field_channels=8
layer.y_field_channels=2
layer.x_field_spacing=8
layer.y_field_spacing=2
layer.pca_node_class = mdp.nodes.iGSFANode
layer.pca_out_dim = 32
layer.pca_args = {} # Specific parameters are defined later
layer.ord_node_class = None
layer.exp_funcs = [identity,]
layer.red_node_class = None
layer.sfa_node_class = mdp.nodes.iGSFANode
layer.sfa_out_dim = 32
layer.sfa_args = {} # Specific parameters are defined later
layer.cloneLayer = False
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)

network = RatlabNetwork3L = system_parameters.ParamsNetwork()
network.name = "Ratlab Network, 3 Layers of SFA & QSFA"
network.L0 = pSFARatlab_L0
network.L1 = pSFARatlab_L1
network.L2 = pSFARatlab_L2
network.layers = [network.L0, network.L1, network.L2]

for i, layer in enumerate(network.layers):
    #layer.sfa_node_class = mdp.nodes.SFANode
    layer.pca_args["offsetting_mode"]=None
    layer.pca_args["reconstruct_with_sfa"]=False    
    layer.pca_args["expansion_funcs"] = None
    layer.pca_args["max_preserved_sfa"]= 12 # 4.0 #4.0 => preserves only SFA components; 16 => half of the computed features are slow, and the remaining reconstructive

    layer.sfa_args["offsetting_mode"]=None
    layer.sfa_args["reconstruct_with_sfa"]=False    
    layer.sfa_args["expansion_funcs"] = [identity, QT]
    layer.sfa_args["max_preserved_sfa"]= 12 # 4.0 #4.0 => preserves only SFA components; 16 => half of the computed features are slow, and the remaining reconstructive

import SystemParameters
from SystemParameters import load_data_from_sSeq
import numpy
import mdp
import sfa_libs 
import more_nodes
import patch_mdp
import imageLoader
from nonlinear_expansion import *
import lattice
import Image
import copy
import os
import string

#MEGAWARNING, remember that initialization overwrites these values!!!!!!!!!!
#Therefore, the next section is useless unless class variables are read!!!!!!!!!!
activate_random_permutation = False
activate_sfa_ordering = False
if activate_random_permutation:
    print "Random Permutation Activated!!!"
    SystemParameters.ParamsSFALayer.ord_node_class = more_nodes.RandomPermutationNode
    SystemParameters.ParamsSFALayer.ord_args = {}
    SystemParameters.ParamsSFASuperNode.ord_node_class = more_nodes.RandomPermutationNode
    SystemParameters.ParamsSFASuperNode.ord_args = {}
elif activate_sfa_ordering:
    SystemParameters.ParamsSFALayer.ord_node_class = mdp.nodes.SFANode
    SystemParameters.ParamsSFASuperNode.ord_node_class = mdp.nodes.SFANode
    SystemParameters.ParamsSFALayer.ord_args = {}
    SystemParameters.ParamsSFASuperNode.ord_args = {}
else:
    SystemParameters.ParamsSFALayer.ord_node_class = None
    SystemParameters.ParamsSFASuperNode.ord_node_class = None
    SystemParameters.ParamsSFALayer.ord_args = {}
    SystemParameters.ParamsSFASuperNode.ord_args = {}
    
    

print "SystemParameters.ParamsSFALayer.ord_node_class is:", SystemParameters.ParamsSFALayer.ord_node_class
print "SystemParameters.ParamsSFALayer.ord_args is:", SystemParameters.ParamsSFALayer.ord_args

def comp_layer_name(cloneLayer, exp_funcs, x_field_channels, y_field_channels, pca_out_dim, sfa_out_dim):
    name = ""
    if cloneLayer == False:
        name += "Homogeneous "
    else:
        name += "Inhomogeneous "
    if exp_funcs == [identity,]:
        name += "Linear "
    else:
        name += "Non-Linear ("
        for fun in exp_funcs:
            name += fun.__name__ + ","
        name += ") "
    name += "Layer: %dx%d => %d => %d"% (y_field_channels, x_field_channels, pca_out_dim, sfa_out_dim)
    return name

def comp_supernode_name(exp_funcs, pca_out_dim, sfa_out_dim):
    name = ""
    if exp_funcs == [identity,]:
        name += "Linear "
    else:
        name += "Non-Linear ("
        for fun in exp_funcs:
            name += fun.__name__ + ","
        name += ") "
    name += "SFA Super Node:  all => %d => %d"% (pca_out_dim, sfa_out_dim)
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


print "*******************************************************************"
print "********    Creating Void Network            ******************"
print "*******************************************************************"
print "******** Setting Layer L0 Parameters          *********************"
layer = pVoidLayer = SystemParameters.ParamsSFASuperNode()
layer.name = "Void Layer"
layer.pca_node_class = None
layer.exp_funcs = [identity,]
layer.red_node_class = None
layer.sfa_node_class = mdp.nodes.IdentityNode
layer.sfa_args = {}
layer.sfa_out_dim = None

####################################################################
###########               Void NETWORK                ############
####################################################################  
network = voidNetwork1L = SystemParameters.ParamsNetwork()
network.name = "Void 1 Layer Network"
network.L0 = pVoidLayer
network.L1 = None
network.L2 = None
network.L3 = None
network.L4 = None
network.layers = [network.L0]

print "*******************************************************************"
print "********    Creating One-Layer Linear SFA Network            ******************"
print "*******************************************************************"
print "******** Setting Layer L0 Parameters          *********************"
layer = pSFAOneLayer = SystemParameters.ParamsSFASuperNode()
layer.name = "One-Node SFA Layer"
layer.pca_node_class = None # mdp.nodes.SFANode
#W
layer.pca_node_class = mdp.nodes.PCANode
layer.pca_args = {}
layer.pca_out_dim = 100 #W None
layer.exp_funcs = [identity,]
layer.red_node_class = None
layer.sfa_node_class = mdp.nodes.SFANode
layer.sfa_args = {}
#W
#layer.sfa_out_dim = None
layer.sfa_out_dim = 49*2 # *3

####################################################################
######        One-Layer Linear SFA NETWORK              ############
####################################################################  
network = SFANetwork1L = SystemParameters.ParamsNetwork()
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
network.L0.sfa_out_dim = 49 * 2 # *3
network.L0.sfa_args = {}

network = HeuristicPaperNetwork = copy.deepcopy(SFANetwork1L)
network.L0.pca_node_class = mdp.nodes.SFANode
network.L0.pca_out_dim = 60
network.exp_funcs = [identity]
network.L0.sfa_node_class = mdp.nodes.SFANode
network.L0.sfa_out_dim = 60 # *3


####################################################################
######        2-Layer Linear SFA NETWORK TUBE           ############
####################################################################  
network = SFANetwork2T = SystemParameters.ParamsNetwork()
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
network = SFANetwork3T = SystemParameters.ParamsNetwork()
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
#SFANetwork1L.layers[0].pca_node_class = mdp.nodes.SFANode
#unsigned_08expo, pair_prodsigmoid_04_adj2_ex, unsigned_2_08expo, sel_exp(42, unsigned_08expo)
u08expoNetwork1L = NetworkSetExpFuncs([identity, sel_exp(42, unsigned_2_08expo), ], copy.deepcopy(SFANetwork1L))
#W
u08expoNetwork1L.layers[0].pca_node_class = mdp.nodes.PCANode
u08expoNetwork1L.layers[0].pca_out_dim = 500/3 #49
u08expoNetwork1L.layers[0].ord_node_class = mdp.nodes.SFANode
u08expoNetwork1L.layers[0].ord_args = {"output_dim": 400}
u08expoNetwork1L.layers[0].sfa_out_dim = 100 #49

#W for 1L network
##u08expoNetwork1L.layers[0].pca_node_class = None
##u08expoNetwork1L.layers[0].ord_node_class = None
##u08expoNetwork1L.layers[0].sfa_out_dim = 49


####################################################################
######        Two-Layer NON-Linear SFA NETWORK TUBE     ############
####################################################################  
#SFANetwork1L.layers[0].pca_node_class = mdp.nodes.SFANode
u08expoNetwork2T = NetworkSetExpFuncs([identity, unsigned_08expo], copy.deepcopy(SFANetwork2T))
#u08expoNetwork2T.layers[0].pca_node_class = mdp.nodes.SFANode
u08expoNetwork2T.layers[0].pca_node_class = None
u08expoNetwork2T.layers[0].ord_node_class = mdp.nodes.SFANode
u08expoNetwork2T.layers[0].ord_args = {"output_dim": 49}
u08expoNetwork2T.layers[1].pca_node_class = None
u08expoNetwork2T.layers[1].ord_node_class = None
u08expoNetwork2T.layers[1].sfa_node_class = mdp.nodes.SFANode
u08expoNetwork2T.layers[1].sfa_out_dim = 49

####################################################################
######        One-Layer Quadratic SFA NETWORK          ############
####################################################################  
quadraticNetwork1L = NetworkSetExpFuncs([identity, pair_prod_ex], copy.deepcopy(SFANetwork1L))
quadraticNetwork1L.layers[0].pca_node_class = mdp.nodes.SFANode
quadraticNetwork1L.layers[0].pca_out_dim = 16


#### 40, 65, 26, 35, 40
#### 60*3=180, 150, 50*3, 55*2 
##GTSRBNetwork = copy.deepcopy(SFANetwork3T)
##GTSRBNetwork.L0.pca_node_class = mdp.nodes.PCANode
##GTSRBNetwork.L0.pca_out_dim = 60 #40*3=120, 32x32=1024
##GTSRBNetwork.L0.ord_node_class = mdp.nodes.SFANode
##GTSRBNetwork.L0.ord_args = {"output_dim": 150} #65....75/3 = 25, This number of dimensions are not expanded!
##GTSRBNetwork.L0.exp_funcs = [identity, sel_exp(42, unsigned_08expo)] #pair_prodsigmoid_04_adj2_ex
##GTSRBNetwork.L0.sfa_node_class = mdp.nodes.SFANode     
##GTSRBNetwork.L0.sfa_out_dim = 50 # 17*3 = 51  26*3 = 78
##
##GTSRBNetwork.L1.exp_funcs = [identity, sel_exp(42, unsigned_08expo)] #pair_prodsigmoid_04_adj2_ex
##GTSRBNetwork.L1.sfa_node_class = mdp.nodes.SFANode     
##GTSRBNetwork.L1.sfa_out_dim = 55 # 35 * 2 = 70
##
##GTSRBNetwork.L2.exp_funcs = [identity, sel_exp(42, unsigned_08expo)] #pair_prodsigmoid_04_adj2_ex
##GTSRBNetwork.L2.sfa_node_class = mdp.nodes.SFANode     
##GTSRBNetwork.L2.sfa_out_dim = 40 # 40 *1.5 = 60

GTSRBNetwork = copy.deepcopy(SFANetwork1L)
GTSRBNetwork.L0.pca_node_class = mdp.nodes.SFANode
#GTSRBNetwork.L0.pca_node_class = None
#W 150
GTSRBNetwork.L0.pca_out_dim = 300/3 # 200 #WW 50  #32x32x3=1024x3 
#GTSRBNetwork.L0.ord_node_class = mdp.nodes.SFANode
#GTSRBNetwork.L0.ord_args = {"output_dim": 120} #75/3 = 25, This number of dimensions are not expanded! sel_exp(42, unsigned_08expo)
#GTSRBNetwork.L0.exp_funcs = [identity, unsigned_08expo] #pair_prodsigmoid_04_adj2_ex, unsigned_08expo, unsigned_2_08expo
GTSRBNetwork.L0.exp_funcs = [identity, unsigned_08expo]
#For SFA features on img: unsigned_08expo, for final system img+hog: unsigned_08expo also
GTSRBNetwork.L0.sfa_node_class = mdp.nodes.SFANode     #SFANode
GTSRBNetwork.L0.sfa_out_dim = 42/3 # WW 26*3 # 17*3 = 51 ## FOR RGB 26, for L/HOG/SFA
GTSRBNetwork.layers=[GTSRBNetwork.L0]

Q_N_k1_d2_L = Q_N_L(k=1.0, d=2.0)


##GTSRBNetwork = copy.deepcopy(SFANetwork2T)
##GTSRBNetwork.L0.pca_node_class = mdp.nodes.SFANode #SFAPCANode
##GTSRBNetwork.L0.pca_out_dim = 150 # 40 #32x32=1024
###GTSRBNetwork.L0.ord_node_class = mdp.nodes.SFAPCANode
###GTSRBNetwork.L0.ord_args = {"output_dim": 65} #75/3 = 25, This number of dimensions are not expanded!
##GTSRBNetwork.L0.exp_funcs = [identity, unsigned_08expo] #sel_exp(42, unsigned_08expo)] #pair_prodsigmoid_04_adj2_ex
##GTSRBNetwork.L0.sfa_node_class = mdp.nodes.SFANode     #SFAPCANode
##GTSRBNetwork.L0.sfa_out_dim = 10 #200 # 17*3 = 51 
##
##
##GTSRBNetwork.L1.exp_funcs =  [identity, unsigned_08expo] # Q_N_k1_d2_L # [identity, unsigned_08expo] # sel_exp(42, unsigned_08expo)] #pair_prodsigmoid_04_adj2_ex
##GTSRBNetwork.L1.pca_node_class = None
##GTSRBNetwork.L1.sfa_node_class = mdp.nodes.SFANode     #SFAPCANode
##GTSRBNetwork.L1.sfa_out_dim = 50


##GTSRBNetwork.L2.exp_funcs = [identity, unsigned_08expo] # sel_exp(42, unsigned_08expo)] #pair_prodsigmoid_04_adj2_ex
##GTSRBNetwork.L2.pca_node_class = None
##GTSRBNetwork.L2.sfa_node_class = mdp.nodes.SFAPCANode     
##GTSRBNetwork.L2.sfa_out_dim = 200

#GTSRBNetwork = copy.deepcopy(SFANetwork2T)
#GTSRBNetwork.L0.pca_node_class = mdp.nodes.PCANode
#GTSRBNetwork.L0.pca_out_dim = 75 #32x32=1024
#GTSRBNetwork.L0.ord_node_class = mdp.nodes.SFANode
#GTSRBNetwork.L0.ord_args = {"output_dim": 50} #75/3 = 25, This number of dimensions are not expanded!
#GTSRBNetwork.L0.exp_funcs = [identity, unsigned_08expo] #pair_prodsigmoid_04_adj2_ex
#GTSRBNetwork.L0.sfa_node_class = mdp.nodes.SFANode     
#GTSRBNetwork.L0.sfa_out_dim = 20
#
##GTSRBNetwork.L0.ord_node_class = mdp.nodes.SFANode
##GTSRBNetwork.L0.ord_args = {"output_dim": 45} #75/3 = 25, This number of dimensions are not expanded!
#GTSRBNetwork.L1.exp_funcs = [identity, unsigned_08expo] #pair_prodsigmoid_04_adj2_ex
#GTSRBNetwork.L1.sfa_node_class = mdp.nodes.SFANode     
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
##layer = pSFAOneNLayer = SystemParameters.ParamsSFASuperNode()
##layer.name = "One-Node SFA NL Layer"
##layer.pca_node_class = None
##layer.exp_funcs = [identity,]
##layer.red_node_class = None
##layer.sfa_node_class = mdp.nodes.SFANode
##layer.sfa_args = {}
##layer.sfa_out_dim = None
##
######################################################################
########        One-Layer Linear SFA NETWORK              ############
######################################################################  
##network = SFANetwork1L = SystemParameters.ParamsNetwork()
##network.name = "SFA 1 Layer Linear Network"
##network.L0 = pSFAOneLayer
##network.L1 = None
##network.L2 = None
##network.L3 = None
##network.L4 = None
##network.layers = [network.L0]


print "*******************************************************************"
print "******** Creating Linear 4L SFA Network          ******************"
print "*******************************************************************"
print "******** Setting Layer L0 Parameters          *********************"
pSFALayerL0 = SystemParameters.ParamsSFALayer()
pSFALayerL0.name = "Homogeneous Linear Layer L0 5x5 => 15"
pSFALayerL0.x_field_channels=5
pSFALayerL0.y_field_channels=5
pSFALayerL0.x_field_spacing=5
pSFALayerL0.y_field_spacing=5
#pSFALayerL0.in_channel_dim=1

pSFALayerL0.pca_node_class = mdp.nodes.SFANode
pSFALayerL0.pca_out_dim = 16
#pSFALayerL0.pca_args = {"block_size": block_size}
pSFALayerL0.pca_args = {"block_size": -1, "train_mode": -1}

pSFALayerL0.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class

pSFALayerL0.exp_funcs = [identity,]

pSFALayerL0.red_node_class = mdp.nodes.WhiteningNode
pSFALayerL0.red_out_dim = 0.9999999
pSFALayerL0.red_args = {}


pSFALayerL0.sfa_node_class = mdp.nodes.SFANode
pSFALayerL0.sfa_out_dim = 16
#pSFALayerL0.sfa_args = {"block_size": -1, "train_mode": -1}

pSFALayerL0.cloneLayer = False
pSFALayerL0.name = comp_layer_name(pSFALayerL0.cloneLayer, pSFALayerL0.exp_funcs, pSFALayerL0.x_field_channels, pSFALayerL0.y_field_channels, pSFALayerL0.pca_out_dim, pSFALayerL0.sfa_out_dim)
SystemParameters.test_object_contents(pSFALayerL0)

print "******** Setting Layer L1 Parameters *********************"
pSFALayerL1 = SystemParameters.ParamsSFALayer()
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

pSFALayerL1.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class

pSFALayerL1.exp_funcs = [identity,]

pSFALayerL1.red_node_class = mdp.nodes.WhiteningNode
pSFALayerL1.red_out_dim = 125
pSFALayerL1.red_args = {}

pSFALayerL1.sfa_node_class = mdp.nodes.SFANode
#sfa_out_dim_L1 = 12
pSFALayerL1.sfa_out_dim = 30
#pSFALayerL1.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL1 = False
#WARNING, DEFAULT IS: pSFALayerL1.cloneLayer = True
pSFALayerL1.cloneLayer = False
pSFALayerL1.name = comp_layer_name(pSFALayerL1.cloneLayer, pSFALayerL1.exp_funcs, pSFALayerL1.x_field_channels, pSFALayerL1.y_field_channels, pSFALayerL1.pca_out_dim, pSFALayerL1.sfa_out_dim)
SystemParameters.test_object_contents(pSFALayerL1)

print "******** Setting Layer L2 Parameters *********************"
pSFALayerL2 = SystemParameters.ParamsSFALayer()
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

pSFALayerL2.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class

pSFALayerL2.exp_funcs = [identity,]

pSFALayerL2.red_node_class = mdp.nodes.WhiteningNode
pSFALayerL2.red_out_dim = 200
pSFALayerL2.red_args = {}

pSFALayerL2.sfa_node_class = mdp.nodes.SFANode
#sfa_out_dim_L2 = 12
pSFALayerL2.sfa_out_dim = 40
#pSFALayerL2.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL2 = False
pSFALayerL2.cloneLayer = False
pSFALayerL2.name = comp_layer_name(pSFALayerL2.cloneLayer, pSFALayerL2.exp_funcs, pSFALayerL2.x_field_channels, pSFALayerL2.y_field_channels, pSFALayerL2.pca_out_dim, pSFALayerL2.sfa_out_dim)
SystemParameters.test_object_contents(pSFALayerL2)


print "******** Setting Layer L3 Parameters *********************"
pSFAL3 = SystemParameters.ParamsSFASuperNode()
pSFAL3.name = "SFA Linear Super Node L3  all =>  300 => 40"
#pSFAL3.in_channel_dim = pSFALayerL2.sfa_out_dim

#pca_node_L3 = mdp.nodes.WhiteningNode(output_dim=pca_out_dim_L3) 
pSFAL3.pca_node_class = mdp.nodes.SFANode
#pca_out_dim_L3 = 210
#pca_out_dim_L3 = 0.999
#WARNING!!! CHANGED PCA TO SFA
pSFAL3.pca_out_dim = 300
#pSFALayerL1.pca_args = {"block_size": block_size}
pSFAL3.pca_args = {"block_size": -1, "train_mode": -1}

pSFAL3.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class

#exp_funcs_L3 = [identity, pair_prod_ex, pair_prod_adj1_ex, pair_prod_adj2_ex, pair_prod_adj3_ex]
pSFAL3.exp_funcs = [identity,]
pSFAL3.inv_use_hint = True
pSFAL3.max_steady_factor=0.35
pSFAL3.delta_factor=0.6
pSFAL3.min_delta=0.0001

pSFAL3.red_node_class = mdp.nodes.WhiteningNode
pSFAL3.red_out_dim = 0.999999
pSFAL3.red_args = {}

pSFAL3.sfa_node_class = mdp.nodes.SFANode
#sfa_out_dim_L1 = 12
pSFAL3.sfa_out_dim = 40
#pSFAL3.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL1 = False
pSFAL3.cloneLayer = False
pSFAL3.name = comp_supernode_name(pSFAL3.exp_funcs, pSFAL3.pca_out_dim, pSFAL3.sfa_out_dim)
SystemParameters.test_object_contents(pSFAL3)


print "******** Setting Layer L4 Parameters *********************"
pSFAL4 = SystemParameters.ParamsSFASuperNode()
pSFAL4.name = "SFA Linear Super Node L4  all => 40 => 40"
#pSFAL4.in_channel_dim = pSFAL3.sfa_out_dim

#pca_node_L3 = mdp.nodes.WhiteningNode(output_dim=pca_out_dim_L3) 
pSFAL4.pca_node_class = mdp.nodes.SFANode
#pca_out_dim_L3 = 210
#pca_out_dim_L3 = 0.999
#WARNING!!! CHANGED PCA TO SFA
pSFAL4.pca_out_dim = 40
#pSFALayerL1.pca_args = {"block_size": block_size}
pSFAL4.pca_args = {"block_size": -1, "train_mode": -1}

pSFAL4.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class

#exp_funcs_L3 = [identity, pair_prod_ex, pair_prod_adj1_ex, pair_prod_adj2_ex, pair_prod_adj3_ex]
pSFAL4.exp_funcs = [identity,]
pSFAL4.inv_use_hint = True
pSFAL4.max_steady_factor=0.35
pSFAL4.delta_factor=0.6
pSFAL4.min_delta=0.0001

pSFAL4.red_node_class = mdp.nodes.WhiteningNode
pSFAL4.red_out_dim = 0.999999
pSFAL4.red_args = {}

pSFAL4.sfa_node_class = mdp.nodes.SFANode
#sfa_out_dim_L1 = 12
pSFAL4.sfa_out_dim = 40
#pSFAL4.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL1 = False
pSFAL4.cloneLayer = False
pSFAL4.name = comp_supernode_name(pSFAL4.exp_funcs, pSFAL4.pca_out_dim, pSFAL4.sfa_out_dim)
SystemParameters.test_object_contents(pSFAL4)


####################################################################
###########               LINEAR NETWORK                ############
####################################################################  
linearNetwork4L = SystemParameters.ParamsNetwork()
linearNetwork4L.name = "Linear 4 Layer Network"
linearNetwork4L.L0 = pSFALayerL0
linearNetwork4L.L1 = pSFALayerL1
linearNetwork4L.L2 = pSFALayerL2
linearNetwork4L.L3 = pSFAL3
network = linearNetwork4L 
network.layers = [network.L0, network.L1, network.L2, network.L3]

linearNetwork5L = SystemParameters.ParamsNetwork()
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
pSFALayerNL0 = SystemParameters.ParamsSFALayer()
pSFALayerNL0.name = "Homogeneous Non-Linear Layer L0 3x3 => 15"
pSFALayerNL0.x_field_channels=5
pSFALayerNL0.y_field_channels=5
pSFALayerNL0.x_field_spacing=5
pSFALayerNL0.y_field_spacing=5
#pSFALayerL0.in_channel_dim=1

pSFALayerNL0.pca_node_class = mdp.nodes.SFANode
pSFALayerNL0.pca_out_dim = 16
#pSFALayerL0.pca_args = {"block_size": block_size}
pSFALayerNL0.pca_args = {"block_size": -1, "train_mode": -1}

pSFALayerNL0.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class

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

pSFALayerNL0.sfa_node_class = mdp.nodes.SFANode
pSFALayerNL0.sfa_out_dim = 16
#pSFALayerNL0.sfa_args = {"block_size": -1, "train_mode": -1}

pSFALayerNL0.cloneLayer = True
pSFALayerNL0.name = comp_layer_name(pSFALayerNL0.cloneLayer, pSFALayerNL0.exp_funcs, pSFALayerNL0.x_field_channels, pSFALayerNL0.y_field_channels, pSFALayerNL0.pca_out_dim, pSFALayerNL0.sfa_out_dim)
SystemParameters.test_object_contents(pSFALayerNL0)

print "******** Setting Layer NL1 Parameters *********************"
pSFALayerNL1 = SystemParameters.ParamsSFALayer()
pSFALayerNL1.name = "Homogeneous Non-Linear Layer L1 3x3 => 30"
pSFALayerNL1.x_field_channels=3
pSFALayerNL1.y_field_channels=3
pSFALayerNL1.x_field_spacing=3
pSFALayerNL1.y_field_spacing=3
#pSFALayerL1.in_channel_dim = pSFALayerL0.sfa_out_dim

pSFALayerNL1.pca_node_class = mdp.nodes.SFANode
#pSFALayerL0.pca_args = {"block_size": block_size}
pSFALayerNL1.pca_args = {"block_size": -1, "train_mode": -1}

#pca_out_dim_L1 = 90
#pca_out_dim_L1 = sfa_out_dim_L0 x_field_channels_L1 * x_field_channels_L1 * 0.75 
#pSFALayerNL1.pca_out_dim = 125
pSFALayerNL1.pca_out_dim = 125
#pSFALayerL1.pca_args = {"block_size": block_size}
#pSFALayerNL1.pca_args = {}

pSFALayerNL1.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class

pSFALayerNL1.exp_funcs = [identity, pair_prod_mix1_ex ]
pSFALayerNL1.inv_use_hint = True
pSFALayerNL1.max_steady_factor=6.5
pSFALayerNL1.delta_factor=0.8
pSFALayerNL1.min_delta=0.0000001

pSFALayerNL1.red_node_class = mdp.nodes.WhiteningNode
pSFALayerNL1.red_out_dim = 0.99999
pSFALayerNL1.red_args = {}

pSFALayerNL1.sfa_node_class = mdp.nodes.SFANode
#sfa_out_dim_L1 = 12
pSFALayerNL1.sfa_out_dim = 30
#pSFALayerNL1.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL1 = False
pSFALayerNL1.cloneLayer = True
pSFALayerNL1.name = comp_layer_name(pSFALayerNL1.cloneLayer, pSFALayerNL1.exp_funcs, pSFALayerNL1.x_field_channels, pSFALayerNL1.y_field_channels, pSFALayerNL1.pca_out_dim, pSFALayerNL1.sfa_out_dim)
SystemParameters.test_object_contents(pSFALayerNL1)

print "******** Setting Layer NL2 Parameters *********************"
pSFALayerNL2 = SystemParameters.ParamsSFALayer()
pSFALayerNL2.name = "Inhomogeneous Non-Linear Layer L2 3x3 => 300 => 40"
pSFALayerNL2.x_field_channels=3
pSFALayerNL2.y_field_channels=3
pSFALayerNL2.x_field_spacing=3
pSFALayerNL2.y_field_spacing=3
#pSFALayerL2.in_channel_dim = pSFALayerL1.sfa_out_dim

pSFALayerNL2.pca_node_class = mdp.nodes.SFANode
#pca_out_dim_L2 = 90
#pca_out_dim_L2 = sfa_out_dim_L0 x_field_channels_L2 * x_field_channels_L2 * 0.75 
pSFALayerNL2.pca_out_dim = 270
pSFALayerNL2.pca_args = {"block_size": -1, "train_mode": -1}

pSFALayerNL2.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class

pSFALayerNL2.exp_funcs = [identity, pair_prod_mix1_ex]
pSFALayerNL2.inv_use_hint = True
pSFALayerNL2.max_steady_factor=6.5
pSFALayerNL2.delta_factor=0.8
pSFALayerNL2.min_delta=0.0000001

pSFALayerNL2.red_node_class = mdp.nodes.WhiteningNode
pSFALayerNL2.red_out_dim = 0.99999
pSFALayerNL2.red_args = {}

pSFALayerNL2.sfa_node_class = mdp.nodes.SFANode
#sfa_out_dim_L2 = 12
pSFALayerNL2.sfa_out_dim = 40
#pSFALayerNL2.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL2 = False
pSFALayerNL2.cloneLayer = False
pSFALayerNL2.name = comp_layer_name(pSFALayerNL2.cloneLayer, pSFALayerNL2.exp_funcs, pSFALayerNL2.x_field_channels, pSFALayerNL2.y_field_channels, pSFALayerNL2.pca_out_dim, pSFALayerNL2.sfa_out_dim)
SystemParameters.test_object_contents(pSFALayerNL2)


print "******** Setting Layer NL3 Parameters *********************"
pSFANL3 = SystemParameters.ParamsSFASuperNode()
pSFANL3.name = "SFA Non-Linear Super Node L3  all => 300 => 40"
#pSFAL3.in_channel_dim = pSFALayerL2.sfa_out_dim

#pca_node_L3 = mdp.nodes.WhiteningNode(output_dim=pca_out_dim_L3) 
pSFANL3.pca_node_class = mdp.nodes.SFANode
#pca_out_dim_L3 = 210
#pca_out_dim_L3 = 0.999
#WARNING!!! CHANGED PCA TO SFA
pSFANL3.pca_out_dim = 300
#pSFALayerL1.pca_args = {"block_size": block_size}
pSFANL3.pca_args = {"block_size": -1, "train_mode": -1}

pSFANL3.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class

#exp_funcs_L3 = [identity, pair_prod_ex, pair_prod_mix1_ex, pair_prod_adj2_ex, pair_prod_adj3_ex]
pSFANL3.exp_funcs = [identity, pair_prod_mix1_ex]
pSFANL3.inv_use_hint = True
pSFANL3.max_steady_factor=6.5
pSFANL3.delta_factor=0.8
pSFANL3.min_delta=0.0000001

pSFANL3.red_node_class = mdp.nodes.WhiteningNode
pSFANL3.red_out_dim = 0.999999
pSFANL3.red_args = {}

pSFANL3.sfa_node_class = mdp.nodes.SFANode
#sfa_out_dim_L1 = 12
pSFANL3.sfa_out_dim = 40
#pSFANL3.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL1 = False
pSFANL3.cloneLayer = False
pSFANL3.name = comp_supernode_name(pSFANL3.exp_funcs, pSFANL3.pca_out_dim, pSFANL3.sfa_out_dim)
SystemParameters.test_object_contents(pSFANL3)


print "******** Setting Layer NL4 Parameters *********************"
pSFANL4 = SystemParameters.ParamsSFASuperNode()
pSFANL4.name = "SFA Linear Super Node L4  all => 40 => 40"
#pSFAL4.in_channel_dim = pSFAL3.sfa_out_dim

#pca_node_L3 = mdp.nodes.WhiteningNode(output_dim=pca_out_dim_L3) 
pSFANL4.pca_node_class = mdp.nodes.SFANode
#pca_out_dim_L3 = 210
#pca_out_dim_L3 = 0.999
#WARNING!!! CHANGED PCA TO SFA
pSFANL4.pca_out_dim = 40
#pSFALayerL1.pca_args = {"block_size": block_size}
pSFANL4.pca_args = {"block_size": -1, "train_mode": -1}

pSFANL4.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class

#exp_funcs_L3 = [identity, pair_prod_ex, pair_prod_adj1_ex, pair_prod_adj2_ex, pair_prod_adj3_ex]
pSFANL4.exp_funcs = [identity, pair_prod_mix1_ex]
pSFANL4.inv_use_hint = True
pSFANL4.max_steady_factor=6.5
pSFANL4.delta_factor=0.8
pSFANL4.min_delta=0.0000001

pSFANL4.red_node_class = mdp.nodes.WhiteningNode
pSFANL4.red_out_dim = 0.99999
pSFANL4.red_args = {}

pSFANL4.sfa_node_class = mdp.nodes.SFANode
#sfa_out_dim_L1 = 12
pSFANL4.sfa_out_dim = 40
#pSFANL4.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL1 = False
pSFANL4.cloneLayer = False
pSFANL4.name = comp_supernode_name(pSFANL4.exp_funcs, pSFANL4.pca_out_dim, pSFANL4.sfa_out_dim)
SystemParameters.test_object_contents(pSFANL4)


####################################################################
###########            NON-LINEAR NETWORKS              ############
####################################################################  
NL_Network4L = SystemParameters.ParamsNetwork()
NL_Network4L.name = "Fully Non-Linear 4 Layer Network"
NL_Network4L.L0 = pSFALayerNL0
NL_Network4L.L1 = pSFALayerNL1
NL_Network4L.L2 = pSFALayerNL2
NL_Network4L.L3 = pSFANL3

NL_Network5L = SystemParameters.ParamsNetwork()
NL_Network5L.name = "Fully Non-Linear 5 Layer Network"
NL_Network5L.L0 = pSFALayerNL0
NL_Network5L.L1 = pSFALayerNL1
NL_Network5L.L2 = pSFALayerNL2
NL_Network5L.L3 = pSFANL3
NL_Network5L.L4 = pSFANL4

Test_Network = SystemParameters.ParamsNetwork()
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
layer = pSFATLayerL0 = SystemParameters.ParamsSFALayer()
layer.name = "Homogeneous Thin Linear Layer L0 4x4 => 13 => x => 13"
layer.x_field_channels=4
layer.y_field_channels=4
layer.x_field_spacing=4
layer.y_field_spacing=4
#layer.in_channel_dim=1

#Warning!!!
layer.pca_node_class = mdp.nodes.PCANode
#layer.pca_node_class = mdp.nodes.SFANode
layer.pca_out_dim = 13
#layer.pca_args = {"block_size": block_size}
layer.pca_args = {}

layer.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class

layer.exp_funcs = [identity,]

#WARNING!
#layer.red_node_class = mdp.nodes.WhiteningNode
layer.red_node_class = None
layer.red_out_dim = 0.9999999
layer.red_args = {}


layer.sfa_node_class = mdp.nodes.SFANode
layer.sfa_out_dim = 13
layer.sfa_args = {}

#Warning, default: layer.cloneLayer = True
layer.cloneLayer = False
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
SystemParameters.test_object_contents(layer)

print "******** Setting Layer L1 Parameters *********************"
layer=None
layer = pSFATLayerL1 = SystemParameters.ParamsSFALayer()

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

layer.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class

layer.exp_funcs = [identity,]

layer.red_node_class = mdp.nodes.WhiteningNode
layer.red_out_dim = 0.99999
layer.red_args = {}

layer.sfa_node_class = mdp.nodes.SFANode
#sfa_out_dim_L1 = 12
#30/(9*15) = 0.222, (4*12)
layer.sfa_out_dim = 40
layer.sfa_args = {}
#Default: cloneLayerL1 = False
layer.cloneLayer = False
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
SystemParameters.test_object_contents(layer)

print "******** Setting Layer L2 Parameters *********************"
layer = None
layer = pSFATLayerL2 = SystemParameters.ParamsSFALayer()
layer.name = "Inhomogeneous Thin Linear Layer L2 2x2 => 158 => 158 => 70"
layer.x_field_channels=2
layer.y_field_channels=2
layer.x_field_spacing=2
layer.y_field_spacing=2
#layer.in_channel_dim = layer.sfa_out_dim

layer.pca_node_class = mdp.nodes.SFANode
#pca_out_dim_L2 = 90
#pca_out_dim_L2 = sfa_out_dim_L1 x_field_channels_L2 * x_field_channels_L2 * 0.75 
layer.pca_out_dim = 100
#layer.pca_args = {"block_size": block_size}
layer.pca_args = {}

layer.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class

layer.exp_funcs = [identity,]

layer.red_node_class = mdp.nodes.WhiteningNode
#Note: number in (0,1) might potentially cause problems, if cells have different output_dim
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros(layer.pca_out_dim)))-2
layer.red_args = {}

layer.sfa_node_class = mdp.nodes.SFANode
#sfa_out_dim_L2 = 12
layer.sfa_out_dim = 70
layer.sfa_args = {}
#Default: cloneLayerL2 = False
layer.cloneLayer = False
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
SystemParameters.test_object_contents(layer)


print "******** Setting Layer L3 Parameters *********************"
layer = None
layer = pSFATLayerL3 = copy.deepcopy(pSFATLayerL2)
layer.name = "Inhomogeneous Thin Linear Layer L3 2x2 => 100 => I => w - 2 => 70"
layer.x_field_channels=2
layer.y_field_channels=2
layer.x_field_spacing=2
layer.y_field_spacing=2
#layer.in_channel_dim = layer.sfa_out_dim

layer.pca_node_class = mdp.nodes.SFANode
#pca_out_dim_L2 = 90
#pca_out_dim_L2 = sfa_out_dim_L1 x_field_channels_L2 * x_field_channels_L2 * 0.75 
layer.pca_out_dim = 100
#layer.pca_args = {"block_size": block_size}
layer.pca_args = {}

layer.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class

layer.exp_funcs = [identity,]

layer.red_node_class = mdp.nodes.WhiteningNode
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros(layer.pca_out_dim)))-2
layer.red_args = {}

layer.sfa_node_class = mdp.nodes.SFANode
#sfa_out_dim_L2 = 12
layer.sfa_out_dim = 70
layer.sfa_args = {}
#Default: cloneLayerL2 = False
layer.cloneLayer = False
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
SystemParameters.test_object_contents(layer)


print "******** Setting Layer L4 Parameters *********************"
layer = None
layer = pSFATLayerL4 = copy.deepcopy(pSFATLayerL3)
layer.name = "Inhomogeneous Thin Linear Layer L4 2x2 => 278 => 278 => 70"
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
SystemParameters.test_object_contents(layer)

print "******** Setting Layer L5 Parameters *********************"
layer = None
layer = pSFATLayerL5 = copy.deepcopy(pSFATLayerL4)
layer.name = "Inhomogeneous Thin Linear Layer L5 2x2 => 278 => 278 => 70"
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
SystemParameters.test_object_contents(layer)


####################################################################
###########           THIN LINEAR NETWORK               ############
####################################################################  
network = linearNetworkT6L = SystemParameters.ParamsNetwork()
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
network = nonlinearNetworkT6L = SystemParameters.ParamsNetwork()
network.name = "Non-Linear 6 Layer Network"
network.L0 = pSFATLayerNL0
network.L1 = pSFATLayerNL1
network.L2 = pSFATLayerNL2
network.L3 = pSFATLayerNL3
network.L4 = pSFATLayerNL4
network.L5 = pSFATLayerNL5


network = TestNetworkT6L = SystemParameters.ParamsNetwork()
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
layer = pSFAULayerL1_H = SystemParameters.ParamsSFALayer()

layer.name = "Homogeneous Ultra Thin Linear Layer L1 1x2 (>= 26) => 26 x => 26, => 20"
layer.x_field_channels=2
layer.y_field_channels=1
layer.x_field_spacing=2
layer.y_field_spacing=1

#WARNING!!!!! mdp.nodes.SFANode
layer.pca_node_class = mdp.nodes.SFANode
layer.pca_out_dim = 26
#layer.pca_args = {"block_size": block_size, "train_mode": -1}
layer.pca_args = {}

layer.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class

layer.exp_funcs = [identity,]

layer.red_node_class = None
layer.red_out_dim = 0.99999
layer.red_args = {}

layer.sfa_node_class = mdp.nodes.SFANode
layer.sfa_out_dim = 20
layer.sfa_args = {}
#Warning!!! layer.cloneLayer = True
layer.cloneLayer = False
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
SystemParameters.test_object_contents(layer)



print "******** Setting Ultra Thin Layer L1 V Parameters *********************"
layer=None
layer = pSFAULayerL1_V = SystemParameters.ParamsSFALayer()

layer.name = "Homogeneous Ultra Thin Linear Layer L1 2x1 (>= 40) => 40 x => 40, => 35"
layer.x_field_channels=1
layer.y_field_channels=2
layer.x_field_spacing=1
layer.y_field_spacing=2

#layer.in_channel_dim = pSFALayerL0.sfa_out_dim
layer.pca_node_class = mdp.nodes.SFANode #WhiteningNode
layer.pca_out_dim = 40
#layer.pca_args = {"block_size": block_size}
layer.pca_args = {}

layer.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class

layer.exp_funcs = [identity]

layer.red_node_class = None
layer.red_out_dim = 0.99999
layer.red_args = {}

layer.sfa_node_class = mdp.nodes.SFANode
layer.sfa_out_dim = 35
layer.sfa_args = {}
#Warning!!!! layer.cloneLayer = True
layer.cloneLayer = False
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
SystemParameters.test_object_contents(layer)


print "******** Setting Layer L2 H Parameters *********************"
layer = None
layer = pSFAULayerL2_H = SystemParameters.ParamsSFALayer()
layer.name = "Inhomogeneous Ultra Thin Linear Layer L2 1x2 (>=70) => 70 => x => (x-2) => 60"
layer.x_field_channels=2
layer.y_field_channels=1
layer.x_field_spacing=2
layer.y_field_spacing=1

layer.pca_node_class = mdp.nodes.SFANode
#pca_out_dim_L2 = 90
#pca_out_dim_L2 = sfa_out_dim_L1 x_field_channels_L2 * x_field_channels_L2 * 0.75 
layer.pca_out_dim = 70
#layer.pca_args = {"block_size": block_size}
layer.pca_args = {}

layer.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class
layer.ord_args = SystemParameters.ParamsSFALayer.ord_args

layer.exp_funcs = [identity,]

layer.red_node_class = None 
#mdp.nodes.PCANode
#Note: number in (0,1) might potentially cause problems, if cells have different output_dim
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros(layer.pca_out_dim)))-2
layer.red_args = {}

layer.sfa_node_class = mdp.nodes.SFANode
#sfa_out_dim_L2 = 12
layer.sfa_out_dim = 60
#layer.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL2 = False
layer.cloneLayer = False
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
SystemParameters.test_object_contents(layer)


print "******** Setting Layer L2 V Parameters *********************"
layer = None
layer = pSFAULayerL2_V = SystemParameters.ParamsSFALayer()
layer.name = "Inhomogeneous Ultra Thin Linear Layer L2 2x1 (>=120) => 120 => x => (x-2) => 60"
layer.x_field_channels=1
layer.y_field_channels=2
layer.x_field_spacing=1
layer.y_field_spacing=2

layer.pca_node_class = mdp.nodes.SFANode
#pca_out_dim_L2 = 90
#pca_out_dim_L2 = sfa_out_dim_L1 x_field_channels_L2 * x_field_channels_L2 * 0.75 
layer.pca_out_dim = 120
#layer.pca_args = {"block_size": block_size}
layer.pca_args = {"block_size": -1, "train_mode": -1}

layer.ord_node_class = SystemParameters.ParamsSFALayer.ord_node_class

layer.exp_funcs = [identity,]

layer.red_node_class = None
#mdp.nodes.WhiteningNode
#Note: number in (0,1) might potentially cause problems, if cells have different output_dim
layer.red_out_dim = len(sfa_libs.apply_funcs_to_signal(layer.exp_funcs, numpy.zeros(layer.pca_out_dim)))-2
layer.red_args = {}

layer.sfa_node_class = mdp.nodes.SFANode
#sfa_out_dim_L2 = 12
layer.sfa_out_dim = 60
#layer.sfa_args = {"block_size": -1, "train_mode": -1}
#Default: cloneLayerL2 = False
layer.cloneLayer = False
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
SystemParameters.test_object_contents(layer)


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
SystemParameters.test_object_contents(layer)

print "******** Setting Layer L3 V Parameters *********************"
layer = None
layer = pSFAULayerL3_V = copy.deepcopy(pSFAULayerL2_V)
layer.name = "Inhomogeneous Ultra Linear Layer L3 2x1 (>=120) =>  120 => x => x-2 => 60"
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
SystemParameters.test_object_contents(layer)

print "******** Setting Layer L4 H Parameters *********************"
layer = None
layer = pSFAULayerL4_H = copy.deepcopy(pSFAULayerL3_H)
layer.name = "Inhomogeneous Ultra Linear Layer L4 1x2 (>=120) =>  120 => x => x-2 => 60"
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
SystemParameters.test_object_contents(layer)

print "******** Setting Layer L4 V Parameters *********************"
layer = None
layer = pSFAULayerL4_V = copy.deepcopy(pSFAULayerL3_V)
layer.name = "Inhomogeneous Ultra Linear Layer L3 2x1 (>=120) =>  120 => x => x-2 => 60"
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
SystemParameters.test_object_contents(layer)


print "******** Setting Layer L5 H Parameters *********************"
layer = None
layer = pSFAULayerL5_H = copy.deepcopy(pSFAULayerL3_H)
layer.name = "Inhomogeneous Ultra Linear Layer L4 1x2 (>=120) =>  120 => x => x-2 => 60"
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
SystemParameters.test_object_contents(layer)

print "******** Setting Layer L5 V Parameters *********************"
layer = None
layer = pSFAULayerL5_V = copy.deepcopy(pSFAULayerL3_V)
layer.name = "Inhomogeneous Ultra Linear Layer L3 2x1 (>=120) =>  120 => x => x-2 => 60"
layer.name = comp_layer_name(layer.cloneLayer, layer.exp_funcs, layer.x_field_channels, layer.y_field_channels, layer.pca_out_dim, layer.sfa_out_dim)
SystemParameters.test_object_contents(layer)

####################################################################
###########           THIN LINEAR NETWORK               ############
####################################################################  
network = linearNetworkU11L = SystemParameters.ParamsNetwork()
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
            layer.pca_node_class = mdp.nodes.SFANode
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
pca_in_dim=(4*1*2*1*2*1*2*1*2*1*2)**2
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



network = linearWhiteningNetwork11L = copy.deepcopy(linearPCANetworkU11L)
for i, layer in enumerate(network.layers):
    layer.sfa_node_class = mdp.nodes.WhiteningNode


network = IEVMLRecNetworkU11L = copy.deepcopy(linearPCANetworkU11L)
# Original:
#IEVMLRecNet_out_dims = [ 13, 20,35,60,60,60,60,60,60,60,60 ]
# Accomodating more space for principal components
#IEVMLRecNet_out_dims = [ 22, 35, 45,60,60,60,60,60,60,60,60 ]
#IEVMLRecNet_out_dims = [ 25, 35,45,60,60,60,60,60,60,60,60 ]

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

if LRec_use_RGB_images == False:
#25+14=39 => 30 (16 PCA/25); 16+13=29 => 22 (9 PCA/16)
#First 80x80 Network:
#IEVMLRecNet_out_dims = [ 30, 42, 52,60,60,60,60,60,60,60,60 ] 
#Improved 80x80 Network: (more feats 2) ******* TOP PARAMS for grayscale
    IEVMLRecNet_out_dims = [ 31, 42, 60,70,70,70,70,70,70,70,70 ]
#Even more features:
#IEVMLRecNet_out_dims = [ 32, 46, 65,75,75,75,75,75,75,75,75 ]
#Base 96x96 Network: (more feats 4)
#IEVMLRecNet_out_dims = [ 38, 48, 65,70,70,70,70,70,70,70,70 ]
#Base 96x96 Network: (more feats 5)
#IEVMLRecNet_out_dims = [ 40, 50, 70,70,70,70,70,70,70,70,70 ]
else:
#Improved 80x80 Network, RGB version (more feats 2)
#    IEVMLRecNet_out_dims = [ 48, 60, 70,70,70,70,70,70,70,70,70 ] #(Orig)
    IEVMLRecNet_out_dims = [ 39, 51, 65,70,70,70,70,70,70,70,70 ] #(less features)


print "IEVMLRecNetworkU11L L0-10_SFA_out_dim = ", IEVMLRecNet_out_dims

for i, layer in enumerate(network.layers):
    layer.sfa_node_class = mdp.nodes.IEVMLRecNode
    layer.sfa_out_dim = IEVMLRecNet_out_dims[i]
    #Q_AP_L(k=nan, d=0.8), Q_AN_exp
    #layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "max_comp":10, "max_num_samples_for_ev":None, "max_test_samples_for_ev":None, "max_preserved_sfa":2.0}
#    layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "max_comp":1, "max_num_samples_for_ev":None, "max_test_samples_for_ev":None, "offsetting_mode":"sensitivity_based", "max_preserved_sfa":2.0}
#    layer.sfa_args = {"pre_expansion_node_class":mdp.nodes.SFANode, "expansion_funcs":[identity, unsigned_08expo], "max_comp":10, "max_num_samples_for_ev":None, "max_test_samples_for_ev":None, "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":2.0}
    layer.sfa_args = {"pre_expansion_node_class":None, "expansion_funcs":[identity, unsigned_08expo], "max_comp":10, "max_num_samples_for_ev":None, "max_test_samples_for_ev":None, "offsetting_mode":"sensitivity_based_pure", "max_preserved_sfa":1.999999} #2.0

    #"offsetting_mode": "QR_decomposition", "sensitivity_based_pure", "sensitivity_based_normalized", "max_comp features"
#    layer.sfa_args = {"expansion_funcs":None, "use_pca":True, "operation":"lin_app", "max_comp":10, "max_num_samples_for_ev":1200, "max_test_samples_for_ev":1200, "k":200}
#    layer.sfa_args = {"expansion_funcs":None, "use_pca":True, "max_comp":6, "max_num_samples_for_ev":600, "max_test_samples_for_ev":600, "k":16}
#network.layers[0].pca_node_class = mdp.nodes.PCANode
#network.layers[0].pca_out_dim = 13

#WARNING, EXPERIMENTAL CODE TO TEST OTHER EXPANSIONS
#network.layers[10].sfa_node_class = mdp.nodes.SFANode
#network.layers[6].sfa_args = {"expansion_funcs":[Q_exp], "max_comp":10, "max_num_samples_for_ev":None, "max_test_samples_for_ev":None, "max_preserved_sfa":2.0}


network = IEMNetworkU11L = copy.deepcopy(linearPCANetworkU11L)
IEMNet_out_dims = [ 13, 20,35,60,60,60,60,60,60,60,60 ]
print "IEMNetworkU11L L0-10_SFA_out_dim = ", IEMNet_out_dims

for i, layer in enumerate(network.layers):
    layer.sfa_node_class = mdp.nodes.IEVMNode
    layer.sfa_out_dim = IEMNet_out_dims[i]
 
#    layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "use_pca":True, "max_comp":1, "max_num_samples_for_ev":800, "max_test_samples_for_ev":200, "k":8}
#    layer.sfa_args = {"expansion_funcs":None, "use_pca":True, "max_comp":10, "max_num_samples_for_ev":800, "max_test_samples_for_ev":800, "k":92}
#    layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "use_pca":True, "operation":"average", "max_comp":10, "max_num_samples_for_ev":1200, "max_test_samples_for_ev":1200, "k":92}
#    layer.sfa_args = {"expansion_funcs":None, "use_pca":True, "use_sfa":True, "operation":"average", "max_comp":10, "max_num_samples_for_ev":600, "max_test_samples_for_ev":600, "k":92}
##    layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "use_pca":True, "use_sfa":True, "operation":"average", "max_comp":10, "max_num_samples_for_ev":400, "max_test_samples_for_ev":400, "k":48, "max_preserved_sfa":2.0}
################## Default:
#    layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "use_pca":True, "use_sfa":True, "operation":"average", "max_comp":10, "max_num_samples_for_ev":600, "max_test_samples_for_ev":600, "k":48, "max_preserved_sfa":2.0, "out_sfa_filter":False}
################## Reconstruction only
    layer.sfa_args = {"expansion_funcs":[identity, unsigned_08expo], "use_pca":True, "use_sfa":False, "operation":"average", "max_comp":20, "max_num_samples_for_ev":1200, "max_test_samples_for_ev":1200, "k":92, "max_preserved_sfa":2.0, "out_sfa_filter":False}
#    layer.sfa_args = {"expansion_funcs":None, "use_pca":True, "operation":"lin_app", "max_comp":10, "max_num_samples_for_ev":1200, "max_test_samples_for_ev":1200, "k":200}
#    layer.sfa_args = {"expansion_funcs":None, "use_pca":True, "max_comp":6, "max_num_samples_for_ev":600, "max_test_samples_for_ev":600, "k":16}
network.layers[0].pca_node_class = mdp.nodes.PCANode
network.layers[0].pca_out_dim = 13

print "*******************************************************************"
print "******** Creating Non-Linear Ultra Thin 11L SFA Network ******************"
print "*******************************************************************"
#Warning, this is based on the linear network, thus modifications to the linear 
#network also afect this non linear network
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
network = nonlinearNetworkU11L = SystemParameters.ParamsNetwork()
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
network = TestNetworkU11L = SystemParameters.ParamsNetwork()
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
T_d3_L = T_L(k=nan, d=3.0)
Q_N_k1_d2_L = Q_N_L(k=1.0, d=2.0)
#T_N_k1_d2_L = T_N_L(k=1.0, d=2.0) #WARNING, should change everywhere to d=3.0 ad
T_N_k1_d3_L = T_N_L(k=1.0, d=3.0) 
Q_d08_L = Q_L(k=nan, d=0.8)
T_d09_L = T_L(k=nan, d=0.9)
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
#u08expoS42NetworkU11L.L0.ord_node_class = mdp.nodes.SFANode
#u08expoS42NetworkU11L.L0.exp_funcs = [identity, unsigned_08expo]
u08expoS42NetworkU11L.L0.sfa_node_class = mdp.nodes.HeadNode #mdp.nodes.SFANode
u08expoS42NetworkU11L.L0.sfa_out_dim = 78/3 #No dim reduction!

u08expoS42NetworkU11L.L1.pca_node_class = mdp.nodes.SFANode
u08expoS42NetworkU11L.L1.pca_out_dim = (55)/2
u08expoS42NetworkU11L.L1.sfa_node_class = mdp.nodes.HeadNode
u08expoS42NetworkU11L.L1.sfa_out_dim = (96)/2

u08expoS42NetworkU11L.L2.pca_node_class = mdp.nodes.SFANode
u08expoS42NetworkU11L.L2.pca_out_dim = (55)/1.5
u08expoS42NetworkU11L.L2.sfa_node_class = mdp.nodes.HeadNode
u08expoS42NetworkU11L.L2.sfa_out_dim = (96)/1.5

u08expoS42NetworkU11L.L3.pca_node_class = mdp.nodes.SFANode
u08expoS42NetworkU11L.L3.pca_out_dim = (55)
u08expoS42NetworkU11L.L3.sfa_node_class = mdp.nodes.HeadNode
u08expoS42NetworkU11L.L3.sfa_out_dim = (97)

u08expoS42NetworkU11L.L4.pca_node_class = mdp.nodes.SFANode
u08expoS42NetworkU11L.L4.pca_out_dim = (42)
u08expoS42NetworkU11L.L4.sfa_node_class = mdp.nodes.HeadNode
u08expoS42NetworkU11L.L4.sfa_out_dim = (84)

u08expoS42NetworkU11L.L5.pca_node_class = mdp.nodes.SFANode
u08expoS42NetworkU11L.L5.pca_out_dim = (42)
u08expoS42NetworkU11L.L5.exp_funcs = [identity]
u08expoS42NetworkU11L.L5.sfa_node_class = mdp.nodes.HeadNode
u08expoS42NetworkU11L.L5.sfa_out_dim = (42)

#W
u08expoS42NetworkU11L.L6.pca_node_class = None
#u08expoS42NetworkU11L.L6.pca_out_dim = (55)
#W
u08expoS42NetworkU11L.L6.exp_funcs = [identity]
u08expoS42NetworkU11L.L6.sfa_node_class = mdp.nodes.SFANode
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
network.L0.sfa_node_class = mdp.nodes.SFANode
network.L0.sfa_args = {}
network.L0.cloneLayer = True

network.L1 = copy.deepcopy(pSFAULayerL1_H)
network.L1.pca_node_class = None
network.L1.pca_args = {}
network.L1.ord_node_class = None
network.L1.ord_args = {}     
network.L1.red_node_class = None
network.L1.red_args = {}
network.L1.sfa_node_class = mdp.nodes.SFANode
network.L1.sfa_args = {}
network.L1.cloneLayer = True

network.L2 = copy.deepcopy(pSFAULayerL1_V)
network.L2.pca_node_class = None
network.L2.pca_args = {}
network.L2.ord_node_class = None
network.L2.ord_args = {}     
network.L2.red_node_class = None
network.L2.red_args = {}
network.L2.sfa_node_class = mdp.nodes.SFANode
network.L2.sfa_args = {}
network.L2.cloneLayer = True

network.L3 = copy.deepcopy(pSFAULayerL2_H)
network.L3.pca_node_class = None
network.L3.pca_args = {}
network.L3.ord_node_class = None
network.L3.ord_args = {}     
network.L3.red_node_class = None
network.L3.red_args = {}
network.L3.sfa_node_class = mdp.nodes.SFANode
network.L3.sfa_args = {}
network.L3.cloneLayer = False

network.L4 = copy.deepcopy(pSFAULayerL2_V)
network.L4.pca_node_class = None
network.L4.pca_args = {}
network.L4.ord_node_class = None
network.L4.ord_args = {}     
network.L4.red_node_class = None
network.L4.red_args = {}
network.L4.sfa_node_class = mdp.nodes.SFANode
network.L4.sfa_args = {}
network.L4.cloneLayer = False






#
#print "******** Setting Layer L3 k-adj-prod Parameters *********************"
#pSFAL3_L3KadjProd = SystemParameters.ParamsSFASuperNode()
##pSFAL3_L3KadjProd.in_channel_dim = pSFALayerL2.sfa_out_dim
#
##pca_node_L3 = mdp.nodes.WhiteningNode(output_dim=pca_out_dim_L3) 
#pSFAL3_L3KadjProd.pca_node_class = mdp.nodes.SFANode
##pca_out_dim_L3 = 210
##pca_out_dim_L3 = 0.999
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
#pSFAL3_L3KadjProd.sfa_node_class = mdp.nodes.SFANode
##sfa_out_dim_L1 = 12
#pSFAL3_L3KadjProd.sfa_out_dim = 40
#pSFAL3_L3KadjProd.sfa_args = {"block_size": -1, "train_mode": -1}
##Default: cloneLayerL1 = False
#pSFAL3_L3KadjProd.cloneLayer = False
#SystemParameters.test_object_contents(pSFAL3_L3KadjProd)
#
#####################################################################
############           NON-LINEAR NETWORK                ############
#####################################################################  
#Network4L = SystemParameters.ParamsNetwork()
#Network4L.name = "Linear 4 Layer Network"
#Network4L.L0 = pSFALayerL0
#Network4L.L1 = pSFALayerL1
#Network4L.L2 = pSFALayerL2
#Network4L.L3 = pSFAL3_L3KadjProd

import os

on_lok21 = os.path.lexists("/local2/tmp/escalafl/")
on_zappa01 = os.path.lexists("/local/escalafl/on_zappa01")
on_lok09 = os.path.lexists("/local/escalafl/on_lok09/")
on_lok10 = os.path.lexists("/local/escalafl/on_lok10")
local_available = os.path.lexists("/local/escalafl/")
if on_lok21:
    user_base_dir = "/local2/tmp/escalafl/"
    frgc_normalized_base_dir = "/local2/tmp/escalafl/Alberto/FRGC_Normalized"
    frgc_noface_base_dir = "/local/escalafl/Alberto/FRGC_NoFace"
    alldb_noface_base_dir = "/local/escalafl/Alberto/AllDB_NoFace"
    frgc_eyeL_normalized_base_dir = "/local/escalafl/Alberto/FRGC_EyeL"
    print "Running on Lok21"
elif on_lok09 or on_lok10:
    user_base_dir = "/local/escalafl/"
    frgc_normalized_base_dir = "/local/escalafl/Alberto/FRGC_Normalized"
    frgc_noface_base_dir = "/local/escalafl/Alberto/FRGC_NoFace"
    alldb_noface_base_dir = "/local/escalafl/Alberto/AllDB_NoFace"
    alldbnormalized_base_dir = "/local/escalafl/Alberto/AllDBNormalized"    
    frgc_eyeL_normalized_base_dir = "/local/escalafl/Alberto/FRGC_EyeL"
    #age_eyes_normalized_base_dir =  "/local/escalafl/Alberto/MORPH_normalizedEyesZByAge"
    print "Running on Lok09 or Lok10"
elif local_available:
    user_base_dir = "/local/escalafl/"
    frgc_normalized_base_dir = "/local/escalafl/Alberto/FRGC_Normalized"
    frgc_noface_base_dir = "/local/escalafl/Alberto/FRGC_NoFace"
    alldb_noface_base_dir = "/local/escalafl/Alberto/AllDB_NoFace"
    alldbnormalized_base_dir = "/local/escalafl/Alberto/AllDBNormalized"
    frgc_eyeL_normalized_base_dir = "/local/escalafl/Alberto/FRGC_EyeL"
    #age_eyes_normalized_base_dir =  "/local/escalafl/Alberto/MORPH_normalizedEyesZByAge"
    print "Unknown host, but /local/escalafl available"
else:
    user_base_dir = "/local/tmp/escalafl/"
    frgc_normalized_base_dir ="/local/tmp/escalafl/Alberto/FRGC_Normalized"    
    frgc_noface_base_dir = "/local/escalafl/Alberto/FRGC_NoFace"
    alldb_noface_base_dir = "/local/escalafl/Alberto/AllDB_NoFace"
    alldbnormalized_base_dir = "/local/escalafl/Alberto/AllDBNormalized"
    frgc_eyeL_normalized_base_dir = "/local/escalafl/Alberto/FRGC_EyeL"
    print "Running on unknown host"
    quit()



    
print "******** Setting Training Information Parameters for Gender **********"
gender_continuous = True and False

iTrainGender = SystemParameters.ParamsInput()
iTrainGender.name = "Gender60x200"
iTrainGender.data_base_dir = user_base_dir + "Alberto/RenderingsGender60x200"
iTrainGender.ids = numpy.arange(0,180) #160, but 180 for paper!
iTrainGender.ages = [999]
iTrainGender.MIN_GENDER = -3
iTrainGender.MAX_GENDER = 3
iTrainGender.GENDER_STEP = 0.10000 #01. 0.20025 default. 0.4 fails, use 0.4005, 0.80075, 0.9005
#iTrainGender.GENDER_STEP = 0.80075 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
iTrainGender.real_genders = numpy.arange(iTrainGender.MIN_GENDER, iTrainGender.MAX_GENDER, iTrainGender.GENDER_STEP)
iTrainGender.genders = map(imageLoader.code_gender, iTrainGender.real_genders)
iTrainGender.racetweens = [999]
iTrainGender.expressions = [0]
iTrainGender.morphs = [0]
iTrainGender.poses = [0]
iTrainGender.lightings = [0]
iTrainGender.slow_signal = 2
iTrainGender.step = 1
iTrainGender.offset = 0
iTrainGender.input_files = imageLoader.create_image_filenames2(iTrainGender.data_base_dir, iTrainGender.slow_signal, iTrainGender.ids, iTrainGender.ages, \
                                            iTrainGender.genders, iTrainGender.racetweens, iTrainGender.expressions, iTrainGender.morphs, \
                                            iTrainGender.poses, iTrainGender.lightings, iTrainGender.step, iTrainGender.offset)
#MEGAWARNING!!!!
#numpy.random.shuffle(iTrainGender.input_files)  
#numpy.random.shuffle(iTrainGender.input_files)  

iTrainGender.num_images = len(iTrainGender.input_files)
#iTrainGender.params = [ids, expressions, morphs, poses, lightings]
iTrainGender.params = [iTrainGender.ids, iTrainGender.ages, iTrainGender.genders, iTrainGender.racetweens, iTrainGender.expressions, \
                  iTrainGender.morphs, iTrainGender.poses, iTrainGender.lightings]

if gender_continuous:
    iTrainGender.block_size = iTrainGender.num_images / len(iTrainGender.params[iTrainGender.slow_signal])  
    iTrainGender.correct_labels = sfa_libs.wider_1Darray(iTrainGender.real_genders, iTrainGender.block_size)
    iTrainGender.correct_classes = sfa_libs.wider_1Darray(iTrainGender.real_genders, iTrainGender.block_size)
else:
    bs = iTrainGender.num_images / len(iTrainGender.params[iTrainGender.slow_signal])
    bs1 = len(iTrainGender.real_genders[iTrainGender.real_genders<0]) * bs
    bs2 = len(iTrainGender.real_genders[iTrainGender.real_genders>=0]) * bs
    
    iTrainGender.block_size = numpy.array([bs1, bs2])
    iTrainGender.correct_labels = numpy.array([-1]*bs1 + [1]*bs2)
    #iTrainGender.correct_classes = sfa_libs.wider_1Darray(iTrainGender.real_genders, iTrainGender.block_size)
    iTrainGender.correct_classes = numpy.array([-1]*bs1 + [1]*bs2)
    
SystemParameters.test_object_contents(iTrainGender)

#iTrainGender.correct_classes = sfa_libs.wider_1Darray(numpy.arange(iTrainGender.num_images / iTrainGender.block_size), iTrainGender.block_size)

print "******** Setting Training Data Parameters for Gender  ****************"
sTrainGender = SystemParameters.ParamsDataLoading()
sTrainGender.input_files = iTrainGender.input_files
sTrainGender.num_images = iTrainGender.num_images
sTrainGender.block_size = iTrainGender.block_size
sTrainGender.image_width = 256
sTrainGender.image_height = 192
sTrainGender.subimage_width = 135
sTrainGender.subimage_height = 135 
sTrainGender.pixelsampling_x = 1
sTrainGender.pixelsampling_y =  1
sTrainGender.subimage_pixelsampling = 2
sTrainGender.subimage_first_row =  sTrainGender.image_height/2-sTrainGender.subimage_height*sTrainGender.pixelsampling_y/2
sTrainGender.subimage_first_column = sTrainGender.image_width/2-sTrainGender.subimage_width*sTrainGender.pixelsampling_x/2+ 5*sTrainGender.pixelsampling_x
sTrainGender.add_noise_L0 = True
sTrainGender.convert_format = "L"
sTrainGender.background_type = "black"
sTrainGender.translation = 2
sTrainGender.translations_x = numpy.random.random_integers(-sTrainGender.translation, sTrainGender.translation, sTrainGender.num_images)
sTrainGender.translations_y = numpy.random.random_integers(-sTrainGender.translation, sTrainGender.translation, sTrainGender.num_images)
sTrainGender.trans_sampled = False
if gender_continuous:
    sTrainGender.train_mode = 'mixed'
else:
    sTrainGender.train_mode = 'clustered'
SystemParameters.test_object_contents(sTrainGender)

print "****** Setting Seen Id Test Information Parameters for Gender ********"
iSeenidGender = SystemParameters.ParamsInput()
iSeenidGender.name = "Gender60x200Seenid"
iSeenidGender.data_base_dir =user_base_dir + "Alberto/RenderingsGender60x200"
iSeenidGender.ids = numpy.arange(0,180)#160, (0,180) for paper!
iSeenidGender.ages = [999]
iSeenidGender.MIN_GENDER = -3
iSeenidGender.MAX_GENDER = 3
iSeenidGender.GENDER_STEP = 0.10000 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
#iSeenidGender.GENDER_STEP = 0.80075 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
iSeenidGender.real_genders = numpy.arange(iSeenidGender.MIN_GENDER, iSeenidGender.MAX_GENDER, iSeenidGender.GENDER_STEP)
iSeenidGender.genders = map(imageLoader.code_gender, iSeenidGender.real_genders)
iSeenidGender.racetweens = [999]
iSeenidGender.expressions = [0]
iSeenidGender.morphs = [0]
iSeenidGender.poses = [0]
iSeenidGender.lightings = [0]
iSeenidGender.slow_signal = 2
iSeenidGender.step = 1
iSeenidGender.offset = 0                             
iSeenidGender.input_files = imageLoader.create_image_filenames2(iSeenidGender.data_base_dir, iSeenidGender.slow_signal, iSeenidGender.ids, iSeenidGender.ages, \
                                            iSeenidGender.genders, iSeenidGender.racetweens, iSeenidGender.expressions, iSeenidGender.morphs, \
                                            iSeenidGender.poses, iSeenidGender.lightings, iSeenidGender.step, iSeenidGender.offset)  
iSeenidGender.num_images = len(iSeenidGender.input_files)
#iSeenidGender.params = [ids, expressions, morphs, poses, lightings]
iSeenidGender.params = [iSeenidGender.ids, iSeenidGender.ages, iSeenidGender.genders, iSeenidGender.racetweens, iSeenidGender.expressions, \
                  iSeenidGender.morphs, iSeenidGender.poses, iSeenidGender.lightings]
if gender_continuous:
    iSeenidGender.block_size = iSeenidGender.num_images / len(iSeenidGender.params[iSeenidGender.slow_signal])
    iSeenidGender.correct_labels = sfa_libs.wider_1Darray(iSeenidGender.real_genders, iSeenidGender.block_size)
    iSeenidGender.correct_classes = sfa_libs.wider_1Darray(iSeenidGender.real_genders, iSeenidGender.block_size)
else:
    bs = iSeenidGender.num_images / len(iSeenidGender.params[iSeenidGender.slow_signal])
    bs1 = len(iSeenidGender.real_genders[iSeenidGender.real_genders<0]) * bs
    bs2 = len(iSeenidGender.real_genders[iSeenidGender.real_genders>=0]) * bs
    iSeenidGender.block_size = numpy.array([bs1, bs2])
    iSeenidGender.correct_labels = numpy.array([-1]*bs1 + [1]*bs2)
    iSeenidGender.correct_classes = numpy.array([-1]*bs1 + [1]*bs2)

SystemParameters.test_object_contents(iSeenidGender)


print "***** Setting Seen Id Sequence Parameters for Gender ****************"
sSeenidGender = SystemParameters.ParamsDataLoading()
sSeenidGender.input_files = iSeenidGender.input_files
sSeenidGender.num_images = iSeenidGender.num_images
sSeenidGender.image_width = 256
sSeenidGender.image_height = 192
sSeenidGender.subimage_width = 135
sSeenidGender.subimage_height = 135 
sSeenidGender.pixelsampling_x = 1
sSeenidGender.pixelsampling_y =  1
sSeenidGender.subimage_pixelsampling = 2
sSeenidGender.subimage_first_row =  sSeenidGender.image_height/2-sSeenidGender.subimage_height*sSeenidGender.pixelsampling_y/2
sSeenidGender.subimage_first_column = sSeenidGender.image_width/2-sSeenidGender.subimage_width*sSeenidGender.pixelsampling_x/2+ 5*sSeenidGender.pixelsampling_x
sSeenidGender.add_noise_L0 = True
sSeenidGender.convert_format = "L"
sSeenidGender.background_type = "black"
sSeenidGender.translation = 2
sSeenidGender.translations_x = numpy.random.random_integers(-sSeenidGender.translation, sSeenidGender.translation, sSeenidGender.num_images)
sSeenidGender.translations_y = numpy.random.random_integers(-sSeenidGender.translation, sSeenidGender.translation, sSeenidGender.num_images)
sSeenidGender.trans_sampled = False
sSeenidGender.load_data = load_data_from_sSeq
SystemParameters.test_object_contents(sSeenidGender)

0
print "** Setting New Id Test Information Parameters for Gender **********"
iNewidGender = SystemParameters.ParamsInput()
iNewidGender.name = "Gender60x200Newid"
iNewidGender.data_base_dir =user_base_dir + "Alberto/RenderingsGender60x200"
iNewidGender.ids = range(180,200)#160,200, 180-200 for paper!
iNewidGender.ages = [999]
iNewidGender.MIN_GENDER = -3
iNewidGender.MAX_GENDER = 3
iNewidGender.GENDER_STEP = 0.10000 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
#iSeenidGender.GENDER_STEP = 0.80075 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
iNewidGender.real_genders = numpy.arange(iNewidGender.MIN_GENDER, iNewidGender.MAX_GENDER, iNewidGender.GENDER_STEP)
iNewidGender.genders = map(imageLoader.code_gender, iNewidGender.real_genders)
iNewidGender.racetweens = [999]
iNewidGender.expressions = [0]
iNewidGender.morphs = [0]
iNewidGender.poses = [0]
iNewidGender.lightings = [0]
iNewidGender.slow_signal = 2
iNewidGender.step = 1
iNewidGender.offset = 0                             
iNewidGender.input_files = imageLoader.create_image_filenames2(iNewidGender.data_base_dir, iNewidGender.slow_signal, iNewidGender.ids, iNewidGender.ages, \
                                            iNewidGender.genders, iNewidGender.racetweens, iNewidGender.expressions, iNewidGender.morphs, \
                                            iNewidGender.poses, iNewidGender.lightings, iNewidGender.step, iNewidGender.offset)  
iNewidGender.num_images = len(iNewidGender.input_files)
#iNewidGender.params = [ids, expressions, morphs, poses, lightings]
iNewidGender.params = [iNewidGender.ids, iNewidGender.ages, iNewidGender.genders, iNewidGender.racetweens, iNewidGender.expressions, \
                  iNewidGender.morphs, iNewidGender.poses, iNewidGender.lightings]
iNewidGender.block_size = iNewidGender.num_images / len(iNewidGender.params[iNewidGender.slow_signal])

iNewidGender.correct_labels = sfa_libs.wider_1Darray(iNewidGender.real_genders, iNewidGender.block_size)
iNewidGender.correct_classes = sfa_libs.wider_1Darray(iNewidGender.real_genders, iNewidGender.block_size)
#iNewidGender.correct_classes = sfa_libs.wider_1Darray(numpy.arange(iNewidGender.num_images / iNewidGender.block_size), iNewidGender.block_size)

if gender_continuous:
    iNewidGender.block_size = iNewidGender.num_images / len(iNewidGender.params[iNewidGender.slow_signal])
    iNewidGender.correct_labels = sfa_libs.wider_1Darray(iNewidGender.real_genders, iNewidGender.block_size)
    iNewidGender.correct_classes = sfa_libs.wider_1Darray(iNewidGender.real_genders, iNewidGender.block_size)
else:
    bs = iNewidGender.num_images / len(iNewidGender.params[iNewidGender.slow_signal])
    bs1 = len(iNewidGender.real_genders[iNewidGender.real_genders<0]) * bs
    bs2 = len(iNewidGender.real_genders[iNewidGender.real_genders>=0]) * bs
    iNewidGender.block_size = numpy.array([bs1, bs2])
    iNewidGender.correct_labels = numpy.array([-1]*bs1 + [1]*bs2)
    iNewidGender.correct_classes = numpy.array( [-1]*bs1 + [1]*bs2)

SystemParameters.test_object_contents(iNewidGender)


print "******** Setting New Id Data Parameters ******************************"
sNewidGender = SystemParameters.ParamsDataLoading()
sNewidGender.input_files = iNewidGender.input_files
sNewidGender.num_images = iNewidGender.num_images
sNewidGender.image_width = 256
sNewidGender.image_height = 192
sNewidGender.subimage_width = 135
sNewidGender.subimage_height = 135 
sNewidGender.pixelsampling_x = 1
sNewidGender.pixelsampling_y =  1
sNewidGender.subimage_pixelsampling = 2
sNewidGender.subimage_first_row =  sNewidGender.image_height/2-sNewidGender.subimage_height*sNewidGender.pixelsampling_y/2
sNewidGender.subimage_first_column = sNewidGender.image_width/2-sNewidGender.subimage_width*sNewidGender.pixelsampling_x/2+ 5*sNewidGender.pixelsampling_x
sNewidGender.add_noise_L0 = True
sNewidGender.convert_format = "L"
sNewidGender.background_type = "black"
sNewidGender.translation = 2
sNewidGender.translations_x = numpy.random.random_integers(-sNewidGender.translation, sNewidGender.translation, sNewidGender.num_images)
sNewidGender.translations_y = numpy.random.random_integers(-sNewidGender.translation, sNewidGender.translation, sNewidGender.num_images)
sNewidGender.trans_sampled = False
sNewidGender.load_data = load_data_from_sSeq
SystemParameters.test_object_contents(sNewidGender)


####################################################################
###########        SYSTEM FOR GENDER ESTIMATION         ############
####################################################################  
ParamsGender = SystemParameters.ParamsSystem()
ParamsGender.name = "Network that extracts gender information"
ParamsGender.network = None
ParamsGender.iTrain = [[iTrainGender]] #[]
ParamsGender.sTrain = [[sTrainGender]] #[]
ParamsGender.iSeenid = iSeenidGender
ParamsGender.sSeenid = sSeenidGender
ParamsGender.iNewid = [[iNewidGender]] #[]
ParamsGender.sNewid = [[sNewidGender]] #[]
ParamsGender.block_size = iTrainGender.block_size
if gender_continuous:
    ParamsGender.train_mode = 'mixed'
else:
    ParamsGender.train_mode = 'clustered'
    
ParamsGender.analysis = None
ParamsGender.enable_reduced_image_sizes = False
ParamsGender.reduction_factor = 1.0
ParamsGender.hack_image_size = 128
ParamsGender.enable_hack_image_size = True

print "******** Setting Train Information Parameters for Identity ***********"
iTrainIdentity = SystemParameters.ParamsInput()
iTrainIdentity.name = "Identities20x500"
iTrainIdentity.data_base_dir =user_base_dir + "Alberto/Renderings20x500"
iTrainIdentity.ids = numpy.arange(0,18)
iTrainIdentity.ages = [999]
#iTrainIdentity.MIN_GENDER = -3
#iTrainIdentity.MAX_GENDER = 3
#iTrainIdentity.GENDER_STEP = 0.10000 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
#iTrainIdentity.GENDER_STEP = 0.80075 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
#iTrainIdentity.genders = map(imageLoader.code_gender, numpy.arange(iTrainIdentity.MIN_GENDER, iTrainIdentity.MAX_GENDER, iTrainIdentity.GENDER_STEP))
iTrainIdentity.genders = [999]
iTrainIdentity.racetweens = [999]
iTrainIdentity.expressions = [0]
iTrainIdentity.morphs = [0]
iTrainIdentity.poses = range(0,500)
iTrainIdentity.lightings = [0]
iTrainIdentity.slow_signal = 0
iTrainIdentity.step = 2
iTrainIdentity.offset = 0     

iTrainIdentity.input_files = imageLoader.create_image_filenames(iTrainIdentity.data_base_dir, iTrainIdentity.slow_signal, iTrainIdentity.ids, iTrainIdentity.expressions, iTrainIdentity.morphs, iTrainIdentity.poses, iTrainIdentity.lightings, iTrainIdentity.step, iTrainIdentity.offset)
iTrainIdentity.num_images = len(iTrainIdentity.input_files)
iTrainIdentity.params = [iTrainIdentity.ids, iTrainIdentity.expressions, iTrainIdentity.morphs, iTrainIdentity.poses, iTrainIdentity.lightings]
iTrainIdentity.block_size= iTrainIdentity.num_images / len(iTrainIdentity.params[iTrainIdentity.slow_signal])

iTrainIdentity.correct_classes = sfa_libs.wider_1Darray(iTrainIdentity.ids, iTrainIdentity.block_size)
iTrainIdentity.correct_labels = sfa_libs.wider_1Darray(iTrainIdentity.ids, iTrainIdentity.block_size)

SystemParameters.test_object_contents(iTrainIdentity)

print "***** Setting Train Sequence Parameters for Identity *****************"
sTrainIdentity = SystemParameters.ParamsDataLoading()
sTrainIdentity.input_files = iTrainIdentity.input_files
sTrainIdentity.num_images = iTrainIdentity.num_images
sTrainIdentity.image_width = 640
sTrainIdentity.image_height = 480
sTrainIdentity.subimage_width = 135
sTrainIdentity.subimage_height = 135 
sTrainIdentity.pixelsampling_x = 2
sTrainIdentity.pixelsampling_y =  2
sTrainIdentity.subimage_pixelsampling = 2
sTrainIdentity.subimage_first_row =  sTrainIdentity.image_height/2-sTrainIdentity.subimage_height*sTrainIdentity.pixelsampling_y/2
sTrainIdentity.subimage_first_column = sTrainIdentity.image_width/2-sTrainIdentity.subimage_width*sTrainIdentity.pixelsampling_x/2+ 5*sTrainIdentity.pixelsampling_x
sTrainIdentity.add_noise_L0 = False
sTrainIdentity.convert_format = "L"
sTrainIdentity.background_type = "black"
sTrainIdentity.translation = 0
sTrainIdentity.translations_x = numpy.random.random_integers(-sTrainIdentity.translation, sTrainIdentity.translation, sTrainIdentity.num_images)
sTrainIdentity.translations_y = numpy.random.random_integers(-sTrainIdentity.translation, sTrainIdentity.translation, sTrainIdentity.num_images)
sTrainIdentity.trans_sampled = False
SystemParameters.test_object_contents(sTrainIdentity)


print "******** Setting Seen Id Information Parameters for Identity *********"
iSeenidIdentity = SystemParameters.ParamsInput()
iSeenidIdentity.name = "Identities20x500"
iSeenidIdentity.data_base_dir =user_base_dir + "Alberto/Renderings20x500"
iSeenidIdentity.ids = numpy.arange(0,18)
iSeenidIdentity.ages = [999]
iSeenidIdentity.MIN_GENDER = -3
iSeenidIdentity.MAX_GENDER = 3
iSeenidIdentity.GENDER_STEP = 0.10000 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
#iSeenidIdentity.GENDER_STEP = 0.80075 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
iSeenidIdentity.genders = map(imageLoader.code_gender, numpy.arange(iSeenidIdentity.MIN_GENDER, iSeenidIdentity.MAX_GENDER, iSeenidIdentity.GENDER_STEP))
iSeenidIdentity.racetweens = [999]
iSeenidIdentity.expressions = [0]
iSeenidIdentity.morphs = [0]
iSeenidIdentity.poses = range(0,500)
iSeenidIdentity.lightings = [0]
iSeenidIdentity.slow_signal = 0
iSeenidIdentity.step = 2
iSeenidIdentity.offset = 1    

iSeenidIdentity.input_files = imageLoader.create_image_filenames(iSeenidIdentity.data_base_dir, iSeenidIdentity.slow_signal, iSeenidIdentity.ids, iSeenidIdentity.expressions, iSeenidIdentity.morphs, iSeenidIdentity.poses, iSeenidIdentity.lightings, iSeenidIdentity.step, iSeenidIdentity.offset)
iSeenidIdentity.num_images = len(iSeenidIdentity.input_files)
iSeenidIdentity.params = [iSeenidIdentity.ids, iSeenidIdentity.expressions, iSeenidIdentity.morphs, iSeenidIdentity.poses, iSeenidIdentity.lightings]
iSeenidIdentity.block_size= iSeenidIdentity.num_images / len(iSeenidIdentity.params[iSeenidIdentity.slow_signal])

iSeenidIdentity.correct_classes = sfa_libs.wider_1Darray(iSeenidIdentity.ids, iSeenidIdentity.block_size)
iSeenidIdentity.correct_labels = sfa_libs.wider_1Darray(iSeenidIdentity.ids, iSeenidIdentity.block_size)

SystemParameters.test_object_contents(iSeenidIdentity)

print "******** Setting Seen Id Sequence Parameters for Identity ************"
sSeenidIdentity = SystemParameters.ParamsDataLoading()
sSeenidIdentity.input_files = iSeenidIdentity.input_files
sSeenidIdentity.num_images = iSeenidIdentity.num_images
sSeenidIdentity.image_width = 640
sSeenidIdentity.image_height = 480
sSeenidIdentity.subimage_width = 135
sSeenidIdentity.subimage_height = 135 
sSeenidIdentity.pixelsampling_x = 2
sSeenidIdentity.pixelsampling_y =  2
sSeenidIdentity.subimage_pixelsampling = 2
sSeenidIdentity.subimage_first_row =  sSeenidIdentity.image_height/2-sSeenidIdentity.subimage_height*sSeenidIdentity.pixelsampling_y/2
sSeenidIdentity.subimage_first_column = sSeenidIdentity.image_width/2-sSeenidIdentity.subimage_width*sSeenidIdentity.pixelsampling_x/2+ 5*sSeenidIdentity.pixelsampling_x
sSeenidIdentity.add_noise_L0 = False
sSeenidIdentity.convert_format = "L"
sSeenidIdentity.background_type = "black"
sSeenidIdentity.translation = 0
sSeenidIdentity.translations_x = numpy.random.random_integers(-sSeenidIdentity.translation, sSeenidIdentity.translation, sSeenidIdentity.num_images)
sSeenidIdentity.translations_y = numpy.random.random_integers(-sSeenidIdentity.translation, sSeenidIdentity.translation, sSeenidIdentity.num_images)
sSeenidIdentity.trans_sampled = False
SystemParameters.test_object_contents(sSeenidIdentity)


print "******** Setting New Id Information Parameters for Identity **********"
iNewidIdentity = SystemParameters.ParamsInput()
iNewidIdentity.name = "Identities20x500"
iNewidIdentity.data_base_dir =user_base_dir + "Alberto/Renderings20x500"
iNewidIdentity.ids = numpy.arange(18,20, dtype="int")
iNewidIdentity.ages = [999]
iNewidIdentity.MIN_GENDER = -3
iNewidIdentity.MAX_GENDER = 3
iNewidIdentity.GENDER_STEP = 0.10000 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
#iNewidIdentity.GENDER_STEP = 0.80075 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
iNewidIdentity.genders = map(imageLoader.code_gender, numpy.arange(iNewidIdentity.MIN_GENDER, iNewidIdentity.MAX_GENDER, iNewidIdentity.GENDER_STEP))
iNewidIdentity.racetweens = [999]
iNewidIdentity.expressions = [0]
iNewidIdentity.morphs = [0]
iNewidIdentity.poses = range(0,500)
iNewidIdentity.lightings = [0]
iNewidIdentity.slow_signal = 0
iNewidIdentity.step = 1
iNewidIdentity.offset = 0     

iNewidIdentity.input_files = imageLoader.create_image_filenames(iNewidIdentity.data_base_dir, iNewidIdentity.slow_signal, iNewidIdentity.ids, iNewidIdentity.expressions, iNewidIdentity.morphs, iNewidIdentity.poses, iNewidIdentity.lightings, iNewidIdentity.step, iNewidIdentity.offset)
iNewidIdentity.num_images = len(iNewidIdentity.input_files)
iNewidIdentity.params = [iNewidIdentity.ids, iNewidIdentity.expressions, iNewidIdentity.morphs, iNewidIdentity.poses, iNewidIdentity.lightings]
iNewidIdentity.block_size= iNewidIdentity.num_images / len(iNewidIdentity.params[iNewidIdentity.slow_signal])

iNewidIdentity.correct_classes = sfa_libs.wider_1Darray(iNewidIdentity.ids, iNewidIdentity.block_size)
iNewidIdentity.correct_labels = sfa_libs.wider_1Darray(iNewidIdentity.ids, iNewidIdentity.block_size)

SystemParameters.test_object_contents(iNewidIdentity)

print "******** Setting New Id Sequence Parameters for Identity *************"
sNewidIdentity = SystemParameters.ParamsDataLoading()
sNewidIdentity.input_files = iNewidIdentity.input_files
sNewidIdentity.num_images = iNewidIdentity.num_images
sNewidIdentity.image_width = 640
sNewidIdentity.image_height = 480
sNewidIdentity.subimage_width = 135
sNewidIdentity.subimage_height = 135 
sNewidIdentity.pixelsampling_x = 2
sNewidIdentity.pixelsampling_y =  2
sNewidIdentity.subimage_pixelsampling = 2
sNewidIdentity.subimage_first_row =  sNewidIdentity.image_height/2-sNewidIdentity.subimage_height*sNewidIdentity.pixelsampling_y/2
sNewidIdentity.subimage_first_column = sNewidIdentity.image_width/2-sNewidIdentity.subimage_width*sNewidIdentity.pixelsampling_x/2+ 5*sNewidIdentity.pixelsampling_x
sNewidIdentity.add_noise_L0 = False
sNewidIdentity.convert_format = "L"
sNewidIdentity.background_type = "black"
sNewidIdentity.translation = 0
sNewidIdentity.translations_x = numpy.random.random_integers(-sNewidIdentity.translation, sNewidIdentity.translation, sNewidIdentity.num_images)
sNewidIdentity.translations_y = numpy.random.random_integers(-sNewidIdentity.translation, sNewidIdentity.translation, sNewidIdentity.num_images)
sNewidIdentity.trans_sampled = False
SystemParameters.test_object_contents(sNewidIdentity)



####################################################################
###########        SYSTEM FOR IDENTITY RECOGNITION      ############
####################################################################  
ParamsIdentity = SystemParameters.ParamsSystem()
ParamsIdentity.name = "Network that extracts identity information"
ParamsIdentity.network = linearNetwork4L
ParamsIdentity.iTrain = iTrainIdentity
ParamsIdentity.sTrain = sTrainIdentity
ParamsIdentity.iSeenid = iSeenidIdentity
ParamsIdentity.sSeenid = sSeenidIdentity
ParamsIdentity.iNewid = iNewidIdentity
ParamsIdentity.sNewid = sNewidIdentity
ParamsIdentity.block_size = iTrainIdentity.block_size
ParamsIdentity.train_mode = 'clustered'
ParamsIdentity.analysis = None
        

print "******** Setting Train Information Parameters for Angle **************"
iTrainAngle = SystemParameters.ParamsInput()
iTrainAngle.name = "Angle20x500"
iTrainAngle.data_base_dir =user_base_dir + "Alberto/Renderings20x500"
iTrainAngle.ids = numpy.arange(0,18)
iTrainAngle.ages = [999]
#iTrainAngle.MIN_GENDER = -3
#iTrainAngle.MAX_GENDER = 3
#iTrainAngle.GENDER_STEP = 0.10000 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
#iTrainAngle.GENDER_STEP = 0.80075 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
#iTrainAngle.genders = map(imageLoader.code_gender, numpy.arange(iTrainAngle.MIN_GENDER, iTrainAngle.MAX_GENDER, iTrainAngle.GENDER_STEP))
iTrainAngle.genders = [999]
iTrainAngle.racetweens = [999]
iTrainAngle.expressions = [0]
iTrainAngle.morphs = [0]
iTrainAngle.real_poses = numpy.linspace(0, 90.0, 125) #0,90,500
iTrainAngle.poses = numpy.arange(0,500,4) #0,500
iTrainAngle.lightings = [0]
iTrainAngle.slow_signal = 3
iTrainAngle.step = 1 # 1
iTrainAngle.offset = 0     

iTrainAngle.input_files = imageLoader.create_image_filenames(iTrainAngle.data_base_dir, iTrainAngle.slow_signal, iTrainAngle.ids, iTrainAngle.expressions, iTrainAngle.morphs, iTrainAngle.poses, iTrainAngle.lightings, iTrainAngle.step, iTrainAngle.offset)
iTrainAngle.num_images = len(iTrainAngle.input_files)
iTrainAngle.params = [iTrainAngle.ids, iTrainAngle.expressions, iTrainAngle.morphs, iTrainAngle.poses, iTrainAngle.lightings]
iTrainAngle.block_size= iTrainAngle.num_images / len(iTrainAngle.params[iTrainAngle.slow_signal])

iTrainAngle.correct_classes = sfa_libs.wider_1Darray(iTrainAngle.poses, iTrainAngle.block_size)
iTrainAngle.correct_labels = sfa_libs.wider_1Darray(iTrainAngle.real_poses, iTrainAngle.block_size)

SystemParameters.test_object_contents(iTrainAngle)

print "***** Setting Train Sequence Parameters for Angle ********************"
sTrainAngle = SystemParameters.ParamsDataLoading()
sTrainAngle.input_files = iTrainAngle.input_files
sTrainAngle.num_images = iTrainAngle.num_images
sTrainAngle.image_width = 640
sTrainAngle.image_height = 480
sTrainAngle.subimage_width = 135
sTrainAngle.subimage_height = 135 
sTrainAngle.pixelsampling_x = 2
sTrainAngle.pixelsampling_y =  2
sTrainAngle.subimage_pixelsampling = 2
sTrainAngle.subimage_first_row =  sTrainAngle.image_height/2-sTrainAngle.subimage_height*sTrainAngle.pixelsampling_y/2
sTrainAngle.subimage_first_column = sTrainAngle.image_width/2-sTrainAngle.subimage_width*sTrainAngle.pixelsampling_x/2+ 5*sTrainAngle.pixelsampling_x
sTrainAngle.add_noise_L0 = False
sTrainAngle.convert_format = "L"
sTrainAngle.background_type = "black"
sTrainAngle.translation = 1
sTrainAngle.translations_x = numpy.random.random_integers(-sTrainAngle.translation, sTrainAngle.translation, sTrainAngle.num_images)
sTrainAngle.translations_y = numpy.random.random_integers(-sTrainAngle.translation, sTrainAngle.translation, sTrainAngle.num_images)
sTrainAngle.trans_sampled = False
SystemParameters.test_object_contents(sTrainAngle)


print "******** Setting Seen Id Information Parameters for Angle ************"
iSeenidAngle = SystemParameters.ParamsInput()
iSeenidAngle.name = "Angle20x500"
iSeenidAngle.data_base_dir =user_base_dir + "Alberto/Renderings20x500"
iSeenidAngle.ids = numpy.arange(0,18)
iSeenidAngle.ages = [999]
iSeenidAngle.MIN_GENDER = -3
iSeenidAngle.MAX_GENDER = 3
iSeenidAngle.GENDER_STEP = 0.10000 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
#iSeenidAngle.GENDER_STEP = 0.80075 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
iSeenidAngle.genders = map(imageLoader.code_gender, numpy.arange(iSeenidAngle.MIN_GENDER, iSeenidAngle.MAX_GENDER, iSeenidAngle.GENDER_STEP))
iSeenidAngle.racetweens = [999]
iSeenidAngle.expressions = [0]
iSeenidAngle.morphs = [0]
iSeenidAngle.real_poses = numpy.linspace(0,90.0,125) #0,90,500
iSeenidAngle.poses = numpy.arange(0,500, 4) #0,500
iSeenidAngle.lightings = [0]
iSeenidAngle.slow_signal = 3
iSeenidAngle.step = 1 # 1
iSeenidAngle.offset = 0

iSeenidAngle.input_files = imageLoader.create_image_filenames(iSeenidAngle.data_base_dir, iSeenidAngle.slow_signal, iSeenidAngle.ids, iSeenidAngle.expressions, iSeenidAngle.morphs, iSeenidAngle.poses, iSeenidAngle.lightings, iSeenidAngle.step, iSeenidAngle.offset)
iSeenidAngle.num_images = len(iSeenidAngle.input_files)
iSeenidAngle.params = [iSeenidAngle.ids, iSeenidAngle.expressions, iSeenidAngle.morphs, iSeenidAngle.poses, iSeenidAngle.lightings]
iSeenidAngle.block_size= iSeenidAngle.num_images / len(iSeenidAngle.params[iSeenidAngle.slow_signal])

iSeenidAngle.correct_classes = sfa_libs.wider_1Darray(iSeenidAngle.poses, iSeenidAngle.block_size)
iSeenidAngle.correct_labels = sfa_libs.wider_1Darray(iSeenidAngle.real_poses, iSeenidAngle.block_size)

SystemParameters.test_object_contents(iSeenidAngle)

print "******** Setting Seen Id Sequence Parameters for Angle ***************"
sSeenidAngle = SystemParameters.ParamsDataLoading()
sSeenidAngle.input_files = iSeenidAngle.input_files
sSeenidAngle.num_images = iSeenidAngle.num_images
sSeenidAngle.image_width = 640
sSeenidAngle.image_height = 480
sSeenidAngle.subimage_width = 135
sSeenidAngle.subimage_height = 135 
sSeenidAngle.pixelsampling_x = 2
sSeenidAngle.pixelsampling_y =  2
sSeenidAngle.subimage_pixelsampling = 2
sSeenidAngle.subimage_first_row =  sSeenidAngle.image_height/2-sSeenidAngle.subimage_height*sSeenidAngle.pixelsampling_y/2
sSeenidAngle.subimage_first_column = sSeenidAngle.image_width/2-sSeenidAngle.subimage_width*sSeenidAngle.pixelsampling_x/2+ 5*sSeenidAngle.pixelsampling_x
sSeenidAngle.add_noise_L0 = False
sSeenidAngle.convert_format = "L"
sSeenidAngle.background_type = "black"
sSeenidAngle.translation = 1
sSeenidAngle.translations_x = numpy.random.random_integers(-sSeenidAngle.translation, sSeenidAngle.translation, sSeenidAngle.num_images)
sSeenidAngle.translations_y = numpy.random.random_integers(-sSeenidAngle.translation, sSeenidAngle.translation, sSeenidAngle.num_images)
sSeenidAngle.trans_sampled = False
SystemParameters.test_object_contents(sSeenidAngle)


print "******** Setting New Id Information Parameters for Angle *************"
iNewidAngle = SystemParameters.ParamsInput()
iNewidAngle.name = "Angle20x500"
iNewidAngle.data_base_dir =user_base_dir + "Alberto/Renderings20x500"
iNewidAngle.ids = numpy.arange(18,20)
iNewidAngle.ages = [999]
iNewidAngle.MIN_GENDER = -3
iNewidAngle.MAX_GENDER = 3
iNewidAngle.GENDER_STEP = 0.10000 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
#iNewidAngle.GENDER_STEP = 0.80075 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
iNewidAngle.genders = map(imageLoader.code_gender, numpy.arange(iNewidAngle.MIN_GENDER, iNewidAngle.MAX_GENDER, iNewidAngle.GENDER_STEP))
iNewidAngle.racetweens = [999]
iNewidAngle.expressions = [0]
iNewidAngle.morphs = [0]
iNewidAngle.real_poses = numpy.linspace(0,90.0,500)
iNewidAngle.poses = numpy.arange(0,500)
iNewidAngle.lightings = [0]
iNewidAngle.slow_signal = 3
iNewidAngle.step = 1
iNewidAngle.offset = 0     

iNewidAngle.input_files = imageLoader.create_image_filenames(iNewidAngle.data_base_dir, iNewidAngle.slow_signal, iNewidAngle.ids, iNewidAngle.expressions, iNewidAngle.morphs, iNewidAngle.poses, iNewidAngle.lightings, iNewidAngle.step, iNewidAngle.offset)
iNewidAngle.num_images = len(iNewidAngle.input_files)
iNewidAngle.params = [iNewidAngle.ids, iNewidAngle.expressions, iNewidAngle.morphs, iNewidAngle.poses, iNewidAngle.lightings]
iNewidAngle.block_size= iNewidAngle.num_images / len(iNewidAngle.params[iNewidAngle.slow_signal])

iNewidAngle.correct_classes = sfa_libs.wider_1Darray(iNewidAngle.poses, iNewidAngle.block_size)
iNewidAngle.correct_labels = sfa_libs.wider_1Darray(iNewidAngle.real_poses, iNewidAngle.block_size)

SystemParameters.test_object_contents(iNewidAngle)

print "******** Setting New Id Sequence Parameters for Angle ****************"
sNewidAngle = SystemParameters.ParamsDataLoading()
sNewidAngle.input_files = iNewidAngle.input_files
sNewidAngle.num_images = iNewidAngle.num_images
sNewidAngle.image_width = 640
sNewidAngle.image_height = 480
sNewidAngle.subimage_width = 135
sNewidAngle.subimage_height = 135 
sNewidAngle.pixelsampling_x = 2
sNewidAngle.pixelsampling_y =  2
sNewidAngle.subimage_pixelsampling = 2
sNewidAngle.subimage_first_row =  sNewidAngle.image_height/2-sNewidAngle.subimage_height*sNewidAngle.pixelsampling_y/2
sNewidAngle.subimage_first_column = sNewidAngle.image_width/2-sNewidAngle.subimage_width*sNewidAngle.pixelsampling_x/2+ 5*sNewidAngle.pixelsampling_x
sNewidAngle.add_noise_L0 = False
sNewidAngle.convert_format = "L"
sNewidAngle.background_type = "black"
sNewidAngle.translation = 1
sNewidAngle.translations_x = numpy.random.random_integers(-sNewidAngle.translation, sNewidAngle.translation, sNewidAngle.num_images)
sNewidAngle.translations_y = numpy.random.random_integers(-sNewidAngle.translation, sNewidAngle.translation, sNewidAngle.num_images)
sNewidAngle.trans_sampled = False
SystemParameters.test_object_contents(sNewidAngle)



####################################################################
###########        SYSTEM FOR ANGLE RECOGNITION      ############
####################################################################  
ParamsAngle = SystemParameters.ParamsSystem()
ParamsAngle.name = "Network that extracts Angle information"
ParamsAngle.network = linearNetwork4L
ParamsAngle.iTrain = iTrainAngle
ParamsAngle.sTrain = sTrainAngle
ParamsAngle.iSeenid = iSeenidAngle
ParamsAngle.sSeenid = sSeenidAngle
ParamsAngle.iNewid = iNewidAngle
ParamsAngle.sNewid = sNewidAngle
ParamsAngle.block_size = iTrainAngle.block_size
ParamsAngle.train_mode = 'mixed'
ParamsAngle.analysis = None





print "***** Setting Training Information Parameters for Translation X ******"
iTrainTransX = SystemParameters.ParamsInput()
iTrainTransX.name = "Translation X: 60Genders x 200 identities"
iTrainTransX.data_base_dir = user_base_dir + "Alberto/RenderingsGender60x200"
iTrainTransX.ids = numpy.arange(0,150) # 160
iTrainTransX.trans = numpy.arange(-50, 50, 2)
if len(iTrainTransX.ids) % len(iTrainTransX.trans) != 0:
    ex="Here the number of translations must be a divisor of the number of identities"
    raise Exception(ex)
iTrainTransX.ages = [999]
iTrainTransX.MIN_GENDER= -3
iTrainTransX.MAX_GENDER = 3
iTrainTransX.GENDER_STEP = 0.10000 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
#iTrainTransX.TransX_STEP = 0.80075 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
iTrainTransX.real_genders = numpy.arange(iTrainTransX.MIN_GENDER, iTrainTransX.MAX_GENDER, iTrainTransX.GENDER_STEP)
iTrainTransX.genders = map(imageLoader.code_gender, iTrainTransX.real_genders)
iTrainTransX.racetweens = [999]
iTrainTransX.expressions = [0]
iTrainTransX.morphs = [0]
iTrainTransX.poses = [0]
iTrainTransX.lightings = [0]
iTrainTransX.slow_signal = 0 #real slow signal is the translation in the x axis, added during image loading
iTrainTransX.step = 1
iTrainTransX.offset = 0
iTrainTransX.input_files = imageLoader.create_image_filenames2(iTrainTransX.data_base_dir, iTrainTransX.slow_signal, iTrainTransX.ids, iTrainTransX.ages, \
                                            iTrainTransX.genders, iTrainTransX.racetweens, iTrainTransX.expressions, iTrainTransX.morphs, \
                                            iTrainTransX.poses, iTrainTransX.lightings, iTrainTransX.step, iTrainTransX.offset)
#MEGAWARNING!!!!
#iTrainTransX.input_files = iTrainTransX.input_files
#numpy.random.shuffle(iTrainTransX.input_files)  
#numpy.random.shuffle(iTrainTransX.input_files)  

iTrainTransX.num_images = len(iTrainTransX.input_files)
#iTrainTransX.params = [ids, expressions, morphs, poses, lightings]
iTrainTransX.params = [iTrainTransX.ids, iTrainTransX.ages, iTrainTransX.genders, iTrainTransX.racetweens, iTrainTransX.expressions, \
                  iTrainTransX.morphs, iTrainTransX.poses, iTrainTransX.lightings]
iTrainTransX.block_size = iTrainTransX.num_images / len (iTrainTransX.trans)
#print "Blocksize = ", iTrainTransX.block_size
#quit()

iTrainTransX.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iTrainTransX.trans)), iTrainTransX.block_size)
iTrainTransX.correct_labels = sfa_libs.wider_1Darray(iTrainTransX.trans, iTrainTransX.block_size)

SystemParameters.test_object_contents(iTrainTransX)

print "******** Setting Training Data Parameters for TransX  ****************"
sTrainTransX = SystemParameters.ParamsDataLoading()
sTrainTransX.input_files = iTrainTransX.input_files
sTrainTransX.num_images = iTrainTransX.num_images
sTrainTransX.block_size = iTrainTransX.block_size
sTrainTransX.image_width = 256
sTrainTransX.image_height = 192
sTrainTransX.subimage_width = 135
sTrainTransX.subimage_height = 135 
sTrainTransX.pixelsampling_x = 1
sTrainTransX.pixelsampling_y =  1
sTrainTransX.subimage_pixelsampling = 2
sTrainTransX.subimage_first_row =  sTrainTransX.image_height/2-sTrainTransX.subimage_height*sTrainTransX.pixelsampling_y/2
sTrainTransX.subimage_first_column = sTrainTransX.image_width/2-sTrainTransX.subimage_width*sTrainTransX.pixelsampling_x/2
#sTrainTransX.subimage_first_column = sTrainTransX.image_width/2-sTrainTransX.subimage_width*sTrainTransX.pixelsampling_x/2+ 5*sTrainTransX.pixelsampling_x
sTrainTransX.add_noise_L0 = True
sTrainTransX.convert_format = "L"
sTrainTransX.background_type = "black"
sTrainTransX.translation = 25
#sTrainTransX.translations_x = numpy.random.random_integers(-sTrainTransX.translation, sTrainTransX.translation, sTrainTransX.num_images)                                                           
sTrainTransX.translations_x = sfa_libs.wider_1Darray(iTrainTransX.trans, iTrainTransX.block_size)
sTrainTransX.translations_y = numpy.random.random_integers(-sTrainTransX.translation, sTrainTransX.translation, sTrainTransX.num_images)
sTrainTransX.trans_sampled = False
SystemParameters.test_object_contents(sTrainTransX)

print "***** Setting Seen ID Information Parameters for Translation X *******"
iSeenidTransX = SystemParameters.ParamsInput()
iSeenidTransX.name = "Test Translation X: 60Genders x 200 identities, dx = 1 pixel"
iSeenidTransX.data_base_dir = user_base_dir + "Alberto/RenderingsGender60x200"
iSeenidTransX.ids = numpy.arange(0,50) # 160
iSeenidTransX.trans = iTrainTransX.trans + 1
if len(iSeenidTransX.ids) % len(iSeenidTransX.trans) != 0:
    ex="Here the number of translations must be a divisor of the number of identities"
    raise Exception(ex)
iSeenidTransX.ages = [999]
iSeenidTransX.MIN_GENDER= -3
iSeenidTransX.MAX_GENDER = 3
iSeenidTransX.GENDER_STEP = 0.10000 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
#iSeenidTransX.TransX_STEP = 0.80075 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
iSeenidTransX.real_genders = numpy.arange(iSeenidTransX.MIN_GENDER, iSeenidTransX.MAX_GENDER, iSeenidTransX.GENDER_STEP)
iSeenidTransX.genders = map(imageLoader.code_gender, iSeenidTransX.real_genders)
iSeenidTransX.racetweens = [999]
iSeenidTransX.expressions = [0]
iSeenidTransX.morphs = [0]
iSeenidTransX.poses = [0]
iSeenidTransX.lightings = [0]
iSeenidTransX.slow_signal = 0 #real slow signal is the translation in the x axis, added during image loading
iSeenidTransX.step = 1
iSeenidTransX.offset = 0
iSeenidTransX.input_files = imageLoader.create_image_filenames2(iSeenidTransX.data_base_dir, iSeenidTransX.slow_signal, iSeenidTransX.ids, iSeenidTransX.ages, \
                                            iSeenidTransX.genders, iSeenidTransX.racetweens, iSeenidTransX.expressions, iSeenidTransX.morphs, \
                                            iSeenidTransX.poses, iSeenidTransX.lightings, iSeenidTransX.step, iSeenidTransX.offset)
#MEGAWARNING!!!!
#numpy.random.shuffle(iSeenidTransX.input_files)  
#numpy.random.shuffle(iSeenidTransX.input_files)  

iSeenidTransX.num_images = len(iSeenidTransX.input_files)
#iSeenidTransX.params = [ids, expressions, morphs, poses, lightings]
iSeenidTransX.params = [iSeenidTransX.ids, iSeenidTransX.ages, iSeenidTransX.genders, iSeenidTransX.racetweens, iSeenidTransX.expressions, \
                  iSeenidTransX.morphs, iSeenidTransX.poses, iSeenidTransX.lightings]
iSeenidTransX.block_size = iSeenidTransX.num_images / len (iSeenidTransX.trans)

iSeenidTransX.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeenidTransX.trans)), iSeenidTransX.block_size)
iSeenidTransX.correct_labels = sfa_libs.wider_1Darray(iSeenidTransX.trans, iSeenidTransX.block_size)

SystemParameters.test_object_contents(iSeenidTransX)

print "******** Setting Seen Id Data Parameters for TransX  *****************"
sSeenidTransX = SystemParameters.ParamsDataLoading()
sSeenidTransX.input_files = iSeenidTransX.input_files
sSeenidTransX.num_images = iSeenidTransX.num_images
sSeenidTransX.block_size = iSeenidTransX.block_size
sSeenidTransX.image_width = 256
sSeenidTransX.image_height = 192
sSeenidTransX.subimage_width = 135
sSeenidTransX.subimage_height = 135 
sSeenidTransX.pixelsampling_x = 1
sSeenidTransX.pixelsampling_y =  1
sSeenidTransX.subimage_pixelsampling = 2
sSeenidTransX.subimage_first_row =  sSeenidTransX.image_height/2-sSeenidTransX.subimage_height*sSeenidTransX.pixelsampling_y/2
sSeenidTransX.subimage_first_column = sSeenidTransX.image_width/2-sSeenidTransX.subimage_width*sSeenidTransX.pixelsampling_x/2
#sSeenidTransX.subimage_first_column = sSeenidTransX.image_width/2-sSeenidTransX.subimage_width*sSeenidTransX.pixelsampling_x/2+ 5*sSeenidTransX.pixelsampling_x
sSeenidTransX.add_noise_L0 = True
sSeenidTransX.convert_format = "L"
sSeenidTransX.background_type = "black"
sSeenidTransX.translation = 20
#sSeenidTransX.translations_x = numpy.random.random_integers(-sSeenidTransX.translation, sSeenidTransX.translation, sSeenidTransX.num_images)                                                           
sSeenidTransX.translations_x = sfa_libs.wider_1Darray(iSeenidTransX.trans, iSeenidTransX.block_size)
sSeenidTransX.translations_y = numpy.random.random_integers(-sSeenidTransX.translation, sSeenidTransX.translation, sSeenidTransX.num_images)
sSeenidTransX.trans_sampled = False
SystemParameters.test_object_contents(sSeenidTransX)


print "******** Setting New Id Information Parameters for Translation X *****"
iNewidTransX = SystemParameters.ParamsInput()
iNewidTransX.name = "New ID Translation X: 60Genders x 200 identities, dx = 1 pixel"
iNewidTransX.data_base_dir =user_base_dir + "Alberto/RenderingsGender60x200"
iNewidTransX.ids = numpy.arange(150,200) # 160
iNewidTransX.trans = numpy.arange(-50,50,2)
if len(iNewidTransX.ids) % len(iNewidTransX.trans) != 0:
    ex="Here the number of translations must be a divisor of the number of identities"
    raise Exception(ex)
iNewidTransX.ages = [999]
iNewidTransX.MIN_GENDER= -3
iNewidTransX.MAX_GENDER = 3
iNewidTransX.GENDER_STEP = 0.10000 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
#iNewidTransX.TransX_STEP = 0.80075 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
iNewidTransX.real_genders = numpy.arange(iNewidTransX.MIN_GENDER, iNewidTransX.MAX_GENDER, iNewidTransX.GENDER_STEP)
iNewidTransX.genders = map(imageLoader.code_gender, iNewidTransX.real_genders)
iNewidTransX.racetweens = [999]
iNewidTransX.expressions = [0]
iNewidTransX.morphs = [0]
iNewidTransX.poses = [0]
iNewidTransX.lightings = [0]
iNewidTransX.slow_signal = 0 #real slow signal is the translation in the x axis, added during image loading
iNewidTransX.step = 1
iNewidTransX.offset = 0
iNewidTransX.input_files = imageLoader.create_image_filenames2(iNewidTransX.data_base_dir, iNewidTransX.slow_signal, iNewidTransX.ids, iNewidTransX.ages, \
                                            iNewidTransX.genders, iNewidTransX.racetweens, iNewidTransX.expressions, iNewidTransX.morphs, \
                                            iNewidTransX.poses, iNewidTransX.lightings, iNewidTransX.step, iNewidTransX.offset)
#MEGAWARNING!!!!
#iNewidTransX.input_files = iNewidTransX.input_files
#numpy.random.shuffle(iNewidTransX.input_files)  
#numpy.random.shuffle(iNewidTransX.input_files)  

iNewidTransX.num_images = len(iNewidTransX.input_files)
#iNewidTransX.params = [ids, expressions, morphs, poses, lightings]
iNewidTransX.params = [iNewidTransX.ids, iNewidTransX.ages, iNewidTransX.genders, iNewidTransX.racetweens, iNewidTransX.expressions, \
                  iNewidTransX.morphs, iNewidTransX.poses, iNewidTransX.lightings]
iNewidTransX.block_size = iNewidTransX.num_images / len (iNewidTransX.trans)

iNewidTransX.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iNewidTransX.trans)), iNewidTransX.block_size)
iNewidTransX.correct_labels = sfa_libs.wider_1Darray(iNewidTransX.trans, iNewidTransX.block_size)

SystemParameters.test_object_contents(iNewidTransX)

print "******** Setting Seen Id Data Parameters for TransX  *****************"
sNewidTransX = SystemParameters.ParamsDataLoading()
sNewidTransX.input_files = iNewidTransX.input_files
sNewidTransX.num_images = iNewidTransX.num_images
sNewidTransX.block_size = iNewidTransX.block_size
sNewidTransX.image_width = 256
sNewidTransX.image_height = 192
sNewidTransX.subimage_width = 135
sNewidTransX.subimage_height = 135 
sNewidTransX.pixelsampling_x = 1
sNewidTransX.pixelsampling_y =  1
sNewidTransX.subimage_pixelsampling = 2
sNewidTransX.subimage_first_row =  sNewidTransX.image_height/2-sNewidTransX.subimage_height*sNewidTransX.pixelsampling_y/2
sNewidTransX.subimage_first_column = sNewidTransX.image_width/2-sNewidTransX.subimage_width*sNewidTransX.pixelsampling_x/2
#sNewidTransX.subimage_first_column = sNewidTransX.image_width/2-sNewidTransX.subimage_width*sNewidTransX.pixelsampling_x/2+ 5*sNewidTransX.pixelsampling_x
sNewidTransX.add_noise_L0 = True
sNewidTransX.convert_format = "L"
sNewidTransX.background_type = "black"
sNewidTransX.translation = 25 #20
#sNewidTransX.translations_x = numpy.random.random_integers(-sNewidTransX.translation, sNewidTransX.translation, sNewidTransX.num_images)                                                           
sNewidTransX.translations_x = sfa_libs.wider_1Darray(iNewidTransX.trans, iNewidTransX.block_size)
sNewidTransX.translations_y = numpy.random.random_integers(-sNewidTransX.translation, sNewidTransX.translation, sNewidTransX.num_images)
sNewidTransX.trans_sampled = False
SystemParameters.test_object_contents(sNewidTransX)


####################################################################
###########    SYSTEM FOR TRANSLATION_X EXTRACTION      ############
####################################################################  
ParamsTransX = SystemParameters.ParamsSystem()
ParamsTransX.name = "Network that extracts TransX information"
ParamsTransX.network = linearNetwork4L
ParamsTransX.iTrain = iTrainTransX
ParamsTransX.sTrain = sTrainTransX
ParamsTransX.iSeenid = iSeenidTransX
ParamsTransX.sSeenid = sSeenidTransX
ParamsTransX.iNewid = iNewidTransX
ParamsTransX.sNewid = sNewidTransX
ParamsTransX.block_size = iTrainTransX.block_size
ParamsTransX.train_mode = 'mixed'
ParamsTransX.analysis = None




print "***** Setting Training Information Parameters for Age ******"
iTrainAge = SystemParameters.ParamsInput()
iTrainAge.name = "Age: 23 Ages x 200 identities"
iTrainAge.data_base_dir =user_base_dir + "Alberto/RendersAge200x23"
iTrainAge.im_base_name = "age"
iTrainAge.ids = numpy.arange(0,180) # 180, warning speeding up
#Available ages: iTrainAge.ages = [15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 35, 36, 40, 42, 44, 45, 46, 48, 50, 55, 60, 65]
iTrainAge.ages = numpy.array([15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 35, 36, 40, 42, 44, 45, 46, 48, 50, 55, 60, 65])
#iTrainAge.ages = numpy.array([15, 20, 24, 30, 35, 40, 45, 50, 55, 60, 65])
iTrainAge.genders = [None]
iTrainAge.racetweens = [None]
iTrainAge.expressions = [0]
iTrainAge.morphs = [0]
iTrainAge.poses = [0]
iTrainAge.lightings = [0]
iTrainAge.slow_signal = 1 
iTrainAge.step = 1 
iTrainAge.offset = 0
iTrainAge.input_files = imageLoader.create_image_filenames3(iTrainAge.data_base_dir, iTrainAge.im_base_name, iTrainAge.slow_signal, iTrainAge.ids, iTrainAge.ages, \
                                            iTrainAge.genders, iTrainAge.racetweens, iTrainAge.expressions, iTrainAge.morphs, \
                                            iTrainAge.poses, iTrainAge.lightings, iTrainAge.step, iTrainAge.offset, verbose=False)

#print "Filenames = ", iTrainAge.input_files
iTrainAge.num_images = len(iTrainAge.input_files)
#print "Num Images = ", iTrainAge.num_images
#iTrainAge.params = [ids, expressions, morphs, poses, lightings]
iTrainAge.params = [iTrainAge.ids, iTrainAge.ages, iTrainAge.genders, iTrainAge.racetweens, iTrainAge.expressions, \
                  iTrainAge.morphs, iTrainAge.poses, iTrainAge.lightings]
iTrainAge.block_size = iTrainAge.num_images / len (iTrainAge.ages)

iTrainAge.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iTrainAge.ages)), iTrainAge.block_size)
iTrainAge.correct_labels = sfa_libs.wider_1Darray(iTrainAge.ages, iTrainAge.block_size)

SystemParameters.test_object_contents(iTrainAge)

print "******** Setting Training Data Parameters for Age  ****************"
sTrainAge = SystemParameters.ParamsDataLoading()
sTrainAge.input_files = iTrainAge.input_files
sTrainAge.num_images = iTrainAge.num_images
sTrainAge.block_size = iTrainAge.block_size
sTrainAge.image_width = 256
sTrainAge.image_height = 192
sTrainAge.subimage_width = 135
sTrainAge.subimage_height = 135 
sTrainAge.pixelsampling_x = 1
sTrainAge.pixelsampling_y =  1
sTrainAge.subimage_pixelsampling = 2
sTrainAge.subimage_first_row =  sTrainAge.image_height/2-sTrainAge.subimage_height*sTrainAge.pixelsampling_y/2
sTrainAge.subimage_first_column = sTrainAge.image_width/2-sTrainAge.subimage_width*sTrainAge.pixelsampling_x/2
#sTrainAge.subimage_first_column = sTrainAge.image_width/2-sTrainAge.subimage_width*sTrainAge.pixelsampling_x/2+ 5*sTrainAge.pixelsampling_x
sTrainAge.add_noise_L0 = True
sTrainAge.convert_format = "L"
sTrainAge.background_type = "blue"
sTrainAge.translation = 1
#sTrainAge.translations_x = numpy.random.random_integers(-sTrainAge.translation, sTrainAge.translation, sTrainAge.num_images)                                                           
sTrainAge.translations_x = numpy.random.random_integers(-sTrainAge.translation, sTrainAge.translation, sTrainAge.num_images)
sTrainAge.translations_y = numpy.random.random_integers(-sTrainAge.translation, sTrainAge.translation, sTrainAge.num_images)
sTrainAge.trans_sampled = False
sTrainAge.train_mode = 'mixed'
sTrainAge.name = iTrainAge.name
sTrainAge.load_data = load_data_from_sSeq
SystemParameters.test_object_contents(sTrainAge)


print "***** Setting Seen Id Test Information Parameters for Age ******"
iSeenidAge = SystemParameters.ParamsInput()
iSeenidAge.name = "Age: 23 Ages x 200 identities"
iSeenidAge.data_base_dir =user_base_dir + "Alberto/RendersAge200x23"
iSeenidAge.im_base_name = "age"
iSeenidAge.ids = numpy.arange(0,180) # 180
#Available ages: iSeenidAge.ages = numpy.array([15, 16, 18, 20, 22, 24, 25, 26, 28, 30, 32, 34, 35, 36, 40, 42, 44, 45, 46, 48, 50, 55, 60, 65])
iSeenidAge.ages = numpy.array([15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 35, 36, 40, 42, 44, 45, 46, 48, 50, 55, 60, 65])
#iSeenidAge.ages = numpy.array([15, 20, 24, 30, 35, 40, 45, 50, 55, 60, 65])
iSeenidAge.genders = [None]
iSeenidAge.racetweens = [None]
iSeenidAge.expressions = [0]
iSeenidAge.morphs = [0]
iSeenidAge.poses = [0]
iSeenidAge.lightings = [0]
iSeenidAge.slow_signal = 1 
iSeenidAge.step = 1
iSeenidAge.offset = 0
iSeenidAge.input_files = imageLoader.create_image_filenames3(iSeenidAge.data_base_dir, iSeenidAge.im_base_name, iSeenidAge.slow_signal, iSeenidAge.ids, iSeenidAge.ages, \
                                            iSeenidAge.genders, iSeenidAge.racetweens, iSeenidAge.expressions, iSeenidAge.morphs, \
                                            iSeenidAge.poses, iSeenidAge.lightings, iSeenidAge.step, iSeenidAge.offset)

iSeenidAge.num_images = len(iSeenidAge.input_files)
#iSeenidAge.params = [ids, expressions, morphs, poses, lightings]
iSeenidAge.params = [iSeenidAge.ids, iSeenidAge.ages, iSeenidAge.genders, iSeenidAge.racetweens, iSeenidAge.expressions, \
                  iSeenidAge.morphs, iSeenidAge.poses, iSeenidAge.lightings]
iSeenidAge.block_size = iSeenidAge.num_images / len (iSeenidAge.ages)

iSeenidAge.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeenidAge.ages)), iSeenidAge.block_size)
iSeenidAge.correct_labels = sfa_libs.wider_1Darray(iSeenidAge.ages, iSeenidAge.block_size)

SystemParameters.test_object_contents(iSeenidAge)

print "******** Setting Seen Id Data Parameters for Age  ****************"
sSeenidAge = SystemParameters.ParamsDataLoading()
sSeenidAge.input_files = iSeenidAge.input_files
sSeenidAge.num_images = iSeenidAge.num_images
sSeenidAge.block_size = iSeenidAge.block_size
sSeenidAge.image_width = 256
sSeenidAge.image_height = 192
sSeenidAge.subimage_width = 135
sSeenidAge.subimage_height = 135 
sSeenidAge.pixelsampling_x = 1
sSeenidAge.pixelsampling_y =  1
sSeenidAge.subimage_pixelsampling = 2
sSeenidAge.subimage_first_row =  sSeenidAge.image_height/2-sSeenidAge.subimage_height*sSeenidAge.pixelsampling_y/2
sSeenidAge.subimage_first_column = sSeenidAge.image_width/2-sSeenidAge.subimage_width*sSeenidAge.pixelsampling_x/2
#sSeenidAge.subimage_first_column = sSeenidAge.image_width/2-sSeenidAge.subimage_width*sSeenidAge.pixelsampling_x/2+ 5*sSeenidAge.pixelsampling_x
sSeenidAge.add_noise_L0 = True
sSeenidAge.convert_format = "L"
sSeenidAge.background_type = "blue"
sSeenidAge.translation = 1
#sSeenidAge.translations_x = numpy.random.random_integers(-sSeenidAge.translation, sSeenidAge.translation, sSeenidAge.num_images)                                                           
sSeenidAge.translations_x = numpy.random.random_integers(-sSeenidAge.translation, sSeenidAge.translation, sSeenidAge.num_images)
sSeenidAge.translations_y = numpy.random.random_integers(-sSeenidAge.translation, sSeenidAge.translation, sSeenidAge.num_images)
sSeenidAge.trans_sampled = False
sSeenidAge.name = iSeenidAge.name
sSeenidAge.load_data = load_data_from_sSeq
SystemParameters.test_object_contents(sSeenidAge)


print "***** Setting New Id Test Information Parameters for Age ******"
iNewidAge = SystemParameters.ParamsInput()
iNewidAge.name = "Age: 23 Ages x 200 identities"
iNewidAge.data_base_dir =user_base_dir + "Alberto/RendersAge200x23"
iNewidAge.im_base_name = "age"
iNewidAge.ids = numpy.arange(180,200) # 180,200
#Available ages: iNewidAge.ages = numpy.array([15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 35, 36, 40, 42, 44, 45, 46, 48, 50, 55, 60, 65])
iNewidAge.ages = numpy.array([15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 35, 36, 40, 42, 44, 45, 46, 48, 50, 55, 60, 65])
#iNewidAge.ages = numpy.array([15, 20, 24, 30, 35, 40, 45, 50, 55, 60, 65])
iNewidAge.genders = [None]
iNewidAge.racetweens = [None]
iNewidAge.expressions = [0]
iNewidAge.morphs = [0]
iNewidAge.poses = [0]
iNewidAge.lightings = [0]
iNewidAge.slow_signal = 1 
iNewidAge.step = 1
iNewidAge.offset = 0
iNewidAge.input_files = imageLoader.create_image_filenames3(iNewidAge.data_base_dir, iNewidAge.im_base_name, iNewidAge.slow_signal, iNewidAge.ids, iNewidAge.ages, \
                                            iNewidAge.genders, iNewidAge.racetweens, iNewidAge.expressions, iNewidAge.morphs, \
                                            iNewidAge.poses, iNewidAge.lightings, iNewidAge.step, iNewidAge.offset)

iNewidAge.num_images = len(iNewidAge.input_files)
#iNewidAge.params = [ids, expressions, morphs, poses, lightings]
iNewidAge.params = [iNewidAge.ids, iNewidAge.ages, iNewidAge.genders, iNewidAge.racetweens, iNewidAge.expressions, \
                  iNewidAge.morphs, iNewidAge.poses, iNewidAge.lightings]
iNewidAge.block_size = iNewidAge.num_images / len (iNewidAge.ages)

iNewidAge.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iNewidAge.ages)), iNewidAge.block_size)
iNewidAge.correct_labels = sfa_libs.wider_1Darray(iNewidAge.ages, iNewidAge.block_size)

SystemParameters.test_object_contents(iNewidAge)

print "******** Setting Training Data Parameters for Age  ****************"
sNewidAge = SystemParameters.ParamsDataLoading()
sNewidAge.input_files = iNewidAge.input_files
sNewidAge.num_images = iNewidAge.num_images
sNewidAge.block_size = iNewidAge.block_size
sNewidAge.image_width = 256
sNewidAge.image_height = 192
sNewidAge.subimage_width = 135
sNewidAge.subimage_height = 135 
sNewidAge.pixelsampling_x = 1
sNewidAge.pixelsampling_y =  1
sNewidAge.subimage_pixelsampling = 2
sNewidAge.subimage_first_row =  sNewidAge.image_height/2-sNewidAge.subimage_height*sNewidAge.pixelsampling_y/2
sNewidAge.subimage_first_column = sNewidAge.image_width/2-sNewidAge.subimage_width*sNewidAge.pixelsampling_x/2
#sNewidAge.subimage_first_column = sNewidAge.image_width/2-sNewidAge.subimage_width*sNewidAge.pixelsampling_x/2+ 5*sNewidAge.pixelsampling_x
sNewidAge.add_noise_L0 = True
sNewidAge.convert_format = "L"
sNewidAge.background_type = "blue"
sNewidAge.translation = 1
#sNewidAge.translations_x = numpy.random.random_integers(-sNewidAge.translation, sNewidAge.translation, sNewidAge.num_images)                                                           
sNewidAge.translations_x = numpy.random.random_integers(-sNewidAge.translation, sNewidAge.translation, sNewidAge.num_images)
sNewidAge.translations_y = numpy.random.random_integers(-sNewidAge.translation, sNewidAge.translation, sNewidAge.num_images)
sNewidAge.trans_sampled = False
sNewidAge.name = iNewidAge.name
sNewidAge.load_data = load_data_from_sSeq
SystemParameters.test_object_contents(sNewidAge)


####################################################################
###########    SYSTEM FOR AGE EXTRACTION      ############
####################################################################  
ParamsAge = SystemParameters.ParamsSystem()
ParamsAge.name = "Network that extracts Age information"
ParamsAge.network = linearNetwork4L
ParamsAge.iTrain = [[iTrainAge]]
ParamsAge.sTrain = [[sTrainAge]]
ParamsAge.iSeenid = iSeenidAge
ParamsAge.sSeenid = sSeenidAge
ParamsAge.iNewid = [[iNewidAge]]
ParamsAge.sNewid = [[sNewidAge]]
ParamsAge.block_size = iTrainAge.block_size
ParamsAge.train_mode = 'mixed'
ParamsAge.analysis = None
ParamsAge.enable_reduced_image_sizes = False
ParamsAge.reduction_factor = 1.0
ParamsAge.hack_image_size = 128
ParamsAge.enable_hack_image_size = True


#PIPELINE FOR FACE DETECTION:
#Orig=TX: DX0=+/- 45, DY0=+/- 20, DS0= 0.55-1.1
#TY: DX1=+/- 20, DY0=+/- 20, DS0= 0.55-1.1
#S: DX1=+/- 20, DY1=+/- 10, DS0= 0.55-1.1
#TMX: DX1=+/- 20, DY1=+/- 10, DS1= 0.775-1.05
#TMY: DX2=+/- 10, DY1=+/- 10, DS1= 0.775-1.05
#MS: DX2=+/- 10, DY2=+/- 5, DS1= 0.775-1.05
#Out About: DX2=+/- 10, DY2=+/- 5, DS2= 0.8875-1.025
#notice: for dx* and dy* intervals are open, while for smin and smax intervals are closed
pipeline_fd = dict(dx0 = 45, dy0 = 20, smin0 = 0.55, smax0 = 1.1,
                dx1 = 20, dy1 = 10, smin1 = 0.775, smax1 = 1.05)
#Pipeline actually supports inputs in: [-dx0, dx0-2] [-dy0, dy0-2] [smin0, smax0] 
#Remember these values are before image resizing

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#This network actually supports images in the closed intervals: [smin0, smax0] [-dy0, dy0]
#but halb-open [-dx0, dx0) 
print "***** Setting Training Information Parameters for Real Translation X ******"
iSeq = iTrainRTransX = SystemParameters.ParamsInput()
iSeq.name = "Real Translation X: (-45, 45, 2) translation and y 40"
iSeq.data_base_dir = frgc_normalized_base_dir
iSeq.ids = numpy.arange(0,7965) # 8000, 7965

iSeq.trans = numpy.arange(-1 * pipeline_fd['dx0'], pipeline_fd['dx0'], 2) # (-50, 50, 2)
if len(iSeq.ids) % len(iSeq.trans) != 0:
    ex="Here the number of translations must be a divisor of the number of identities"
    raise Exception(ex)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=4, image_postfix=".jpg")
iSeq.input_files = iSeq.input_files * 4 # warning!!! 4, 8
#To avoid grouping similar images next to one other
numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.trans)
print "BLOCK SIZE =", iSeq.block_size 
iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.trans)), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Training Data Parameters for Real TransX  ****************"
sSeq = sTrainRTransX = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 128
sSeq.subimage_height = 128 


sSeq.trans_x_max = pipeline_fd['dx0']
sSeq.trans_x_min = -1 * pipeline_fd['dx0']

#WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
sSeq.trans_y_max = pipeline_fd['dy0']
sSeq.trans_y_min = -1 * sSeq.trans_y_max

#iSeq.scales = numpy.linspace(0.5, 1.30, 16) # (-50, 50, 2)
sSeq.min_sampling = pipeline_fd['smin0']
sSeq.max_sampling = pipeline_fd['smax0']

sSeq.pixelsampling_x = sSeq.pixelsampling_y = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
#sSeq.subimage_pixelsampling = 2
sSeq.subimage_first_row =  sSeq.image_height/2.0-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2.0-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
#random translation for th w coordinate
#sSeq.translation = 20 # 25, Should be 20!!! Warning: 25
#sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)                                                           
sSeq.translations_x = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
sSeq.translations_y = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, sSeq.num_images)
sSeq.trans_sampled = True
sSeq.name = "RTans X Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f)"%(sSeq.trans_x_min, 
    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.min_sampling*100, sSeq.max_sampling*100)
SystemParameters.test_object_contents(sSeq)

print "***** Setting Seen ID Information Parameters for Real Translation X *******"
iSeq = sSeq = None
iSeq = iSeenidRTransX = SystemParameters.ParamsInput()
iSeq.name = "Test Real Translation X: (-45, 45, 2) translation"
iSeq.data_base_dir = frgc_normalized_base_dir
iSeq.ids = numpy.arange(8000,8990) # WARNING 8900
iSeq.trans = numpy.arange(sTrainRTransX.trans_x_min, sTrainRTransX.trans_x_max, 2) #WARNING!!!! (-50, 50, 2)
if len(iSeq.ids) % len(iSeq.trans) != 0:
    ex="Here the number of translations must be a divisor of the number of identities"
    raise Exception(ex)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=4, image_postfix=".jpg")

iSeq.input_files = iSeq.input_files * 16 # Warning!!! 16, 32
numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.trans)

iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.trans)), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Seen Id Data Parameters for Real TransX  ****************"
sSeq = sSeenidRTransX = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 128
sSeq.subimage_height = 128 
sSeq.trans_x_max = sTrainRTransX.trans_x_max
sSeq.trans_x_min = sTrainRTransX.trans_x_min
sSeq.trans_y_max = sTrainRTransX.trans_y_max
sSeq.trans_y_min = sTrainRTransX.trans_y_min
#iSeq.scales = numpy.linspace(0.5, 1.30, 16) # (-50, 50, 2)
sSeq.min_sampling = sTrainRTransX.min_sampling
sSeq.max_sampling = sTrainRTransX.max_sampling
sSeq.pixelsampling_x = sSeq.pixelsampling_y = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
#sSeq.subimage_pixelsampling = 2
sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
#random translation for th w coordinate
#sSeq.translation = 20 #25, 20, WARNING!!!!!!!
#sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)                                                           
sSeq.translations_x = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
sSeq.translations_y = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, sSeq.num_images)
sSeq.trans_sampled = True
sSeq.name = "RTans X Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f)"%(sSeq.trans_x_min, 
    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.min_sampling*100, sSeq.max_sampling*100)
SystemParameters.test_object_contents(sSeq)


print "******** Setting New Id Information Parameters for Real Translation X *****"
iSeq = sSeq = None
iSeq = iNewidRTransX = SystemParameters.ParamsInput()
iSeq.name = "New ID Real Translation X: (-45, 45, 2) translation"
iSeq.data_base_dir = frgc_normalized_base_dir
iSeq.ids = numpy.arange(9000,9990) # 8000, 10000
iSeq.trans = numpy.arange(sTrainRTransX.trans_x_min, sTrainRTransX.trans_x_max, 2) # (-45, 45, 2)
if len(iSeq.ids) % len(iSeq.trans) != 0:
    ex="Here the number of translations must be a divisor of the number of identities"
    raise Exception(ex)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=4, image_postfix=".jpg")
 
#MEGAWARNING!!!!
iSeq.input_files = iSeq.input_files * 4 #warning * 4
numpy.random.shuffle(iSeq.input_files)  
iSeq.num_images = len(iSeq.input_files)
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.trans)

iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.trans)), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting New ID Data Parameters for Real TransX  ****************"
sSeq = sNewidRTransX = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 128
sSeq.subimage_height = 128 
sSeq.trans_x_max = sTrainRTransX.trans_x_max
sSeq.trans_x_min = sTrainRTransX.trans_x_min
sSeq.trans_y_max = sTrainRTransX.trans_y_max
sSeq.trans_y_min = sTrainRTransX.trans_y_min
#iSeq.scales = numpy.linspace(0.5, 1.30, 16) # (-50, 50, 2)
sSeq.min_sampling = sTrainRTransX.min_sampling
sSeq.max_sampling = sTrainRTransX.max_sampling
sSeq.pixelsampling_x = sSeq.pixelsampling_y = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
sSeq.subimage_pixelsampling = 2
sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
#random translation for th w coordinate
sSeq.translation = 20 #20
#sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)                                                           
sSeq.translations_x = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
sSeq.translations_y = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, sSeq.num_images)
sSeq.trans_sampled = True
sSeq.name = "RTans X Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f)"%(sSeq.trans_x_min, 
    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.min_sampling*100, sSeq.max_sampling*100)
SystemParameters.test_object_contents(sSeq)


####################################################################
###########    SYSTEM FOR REAL_TRANSLATION_X EXTRACTION      ############
####################################################################  
ParamsRTransX = SystemParameters.ParamsSystem()
ParamsRTransX.name = sTrainRTransX.name
ParamsRTransX.network = linearNetwork4L
ParamsRTransX.iTrain = iTrainRTransX
ParamsRTransX.sTrain = sTrainRTransX
ParamsRTransX.iSeenid = iSeenidRTransX
ParamsRTransX.sSeenid = sSeenidRTransX
ParamsRTransX.iNewid = iNewidRTransX
ParamsRTransX.sNewid = sNewidRTransX
##MEGAWARNING!!!!
#ParamsRTransX.iNewid = iNewidTransX
#ParamsRTransX.sNewid = sNewidTransX
#ParamsRTransX.sNewid.translations_y = ParamsRTransX.sNewid.translations_y * 0.0 + 8.0

ParamsRTransX.block_size = iTrainRTransX.block_size
ParamsRTransX.train_mode = 'mixed'
ParamsRTransX.analysis = None

ParamsRTransX.enable_reduced_image_sizes = True
ParamsRTransX.reduction_factor = 2.0
ParamsRTransX.hack_image_size = 64
ParamsRTransX.enable_hack_image_size = True


#GC / 17 signals, mse=11.5



# YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYyYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
#This network actually supports images in the closed intervals: [-dx1, dx1], [smin0, smax0]
#but halb-open [-dy0, dy0)
print "***** Setting Training Information Parameters for Real Translation Y ******"
iSeq = iTrainRTransY = SystemParameters.ParamsInput()
iSeq.name = "Real Translation Y: Y(-20, 20, 1) translation and dx 20"
iSeq.data_base_dir = frgc_normalized_base_dir
iSeq.ids = numpy.arange(0,8000) # 8000, 7965

iSeq.trans = numpy.arange(-1 * pipeline_fd['dy0'], pipeline_fd['dy0'], 1) # (-50, 50, 2)
if len(iSeq.ids) % len(iSeq.trans) != 0:
    ex="Here the number of translations must be a divisor of the number of identities"
    raise Exception(ex)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=4, image_postfix=".jpg")
iSeq.input_files = iSeq.input_files * 4 # warning!!! 4, 8
#To avoid grouping similar images next to one other
numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.trans)
print "BLOCK SIZE =", iSeq.block_size 
iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.trans)), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Training Data Parameters for Real TransY  ****************"
sSeq = sTrainRTransY = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 128
sSeq.subimage_height = 128 


sSeq.trans_x_max = pipeline_fd['dx1']
sSeq.trans_x_min = -1 * pipeline_fd['dx1']

sSeq.trans_y_max = pipeline_fd['dy0']
sSeq.trans_y_min = -1 * sSeq.trans_y_max

#iSeq.scales = numpy.linspace(0.5, 1.30, 16) # (-50, 50, 2)
sSeq.min_sampling = pipeline_fd['smin0']
sSeq.max_sampling = pipeline_fd['smax0']

sSeq.pixelsampling_x = sSeq.pixelsampling_y = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
#sSeq.subimage_pixelsampling = 2
sSeq.subimage_first_row =  sSeq.image_height/2.0-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2.0-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
#random translation for th w coordinate
#sSeq.translation = 20 # 25, Should be 20!!! Warning: 25
#sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)                                                           
sSeq.translations_x = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, sSeq.num_images)
sSeq.translations_y = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)

sSeq.trans_sampled = True
sSeq.name = "RTans Y Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f)"%(sSeq.trans_x_min, 
    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.min_sampling*100, sSeq.max_sampling*100)
iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)

print "***** Setting Seen ID Information Parameters for Real Translation Y *******"
iSeq = sSeq = None
iSeq = iSeenidRTransY = SystemParameters.ParamsInput()
iSeq.name = "Test Real Translation Y: (-20, 20, 1) translation"
iSeq.data_base_dir = frgc_normalized_base_dir
iSeq.ids = numpy.arange(8000,9000) # WARNING 8900
iSeq.trans = numpy.arange(sTrainRTransY.trans_y_min, sTrainRTransY.trans_y_max, 1) #WARNING!!!! (-50, 50, 2)
if len(iSeq.ids) % len(iSeq.trans) != 0:
    ex="Here the number of translations must be a divisor of the number of identities"
    raise Exception(ex)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=4, image_postfix=".jpg")

iSeq.input_files = iSeq.input_files * 16 # Warning!!! 16
numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.trans)

iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.trans)), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Seen Id Data Parameters for Real TransY  ****************"
sSeq = sSeenidRTransY = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 128
sSeq.subimage_height = 128 
sSeq.trans_x_max = sTrainRTransY.trans_x_max
sSeq.trans_x_min = sTrainRTransY.trans_x_min
sSeq.trans_y_max = sTrainRTransY.trans_y_max
sSeq.trans_y_min = sTrainRTransY.trans_y_min
#iSeq.scales = numpy.linspace(0.5, 1.30, 16) # (-50, 50, 2)
sSeq.min_sampling = sTrainRTransY.min_sampling
sSeq.max_sampling = sTrainRTransY.max_sampling
sSeq.pixelsampling_x = sSeq.pixelsampling_y = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
#sSeq.subimage_pixelsampling = 2
sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
#random translation for th w coordinate
#sSeq.translation = 20 #25, 20, WARNING!!!!!!!
#sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)                                                           
sSeq.translations_x = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, sSeq.num_images)
sSeq.translations_y = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
sSeq.trans_sampled = True
sSeq.name = "RTans Y Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f)"%(sSeq.trans_x_min, 
    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.min_sampling*100, sSeq.max_sampling*100)
iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)


print "******** Setting New Id Information Parameters for Real Translation Y *****"
iSeq = sSeq = None
iSeq = iNewidRTransY = SystemParameters.ParamsInput()
iSeq.name = "New ID Real Translation Y: (-20, 20, 1) translation"
iSeq.data_base_dir = frgc_normalized_base_dir
iSeq.ids = numpy.arange(9000,10000) # 8000, 10000
iSeq.trans = numpy.arange(sTrainRTransY.trans_y_min, sTrainRTransY.trans_y_max, 1) # (-50, 50, 2)
if len(iSeq.ids) % len(iSeq.trans) != 0:
    ex="Here the number of translations must be a divisor of the number of identities"
    raise Exception(ex)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=4, image_postfix=".jpg")
 
#MEGAWARNING!!!!
iSeq.input_files = iSeq.input_files * 4 #warning * 4
numpy.random.shuffle(iSeq.input_files)  
iSeq.num_images = len(iSeq.input_files)
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.trans)

iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.trans)), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting New ID Data Parameters for Real TransY  ****************"
sSeq = sNewidRTransY = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 128
sSeq.subimage_height = 128 
sSeq.trans_x_max = sTrainRTransY.trans_x_max
sSeq.trans_x_min = sTrainRTransY.trans_x_min
sSeq.trans_y_max = sTrainRTransY.trans_y_max
sSeq.trans_y_min = -1 * sTrainRTransY.trans_y_min
#iSeq.scales = numpy.linspace(0.5, 1.30, 16) # (-50, 50, 2)
sSeq.min_sampling = sTrainRTransY.min_sampling
sSeq.max_sampling = sTrainRTransY.max_sampling
sSeq.pixelsampling_x = sSeq.pixelsampling_y = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
sSeq.subimage_pixelsampling = 2
sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
#random translation for th w coordinate
sSeq.translation = 20 #20
#sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)                                                           
sSeq.translations_x = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, sSeq.num_images)
sSeq.translations_y = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
sSeq.trans_sampled = True
sSeq.name = "RTans Y Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f)"%(sSeq.trans_x_min, 
    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.min_sampling*100, sSeq.max_sampling*100)
iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)


####################################################################
###########    SYSTEM FOR REAL_TRANSLATION_Y EXTRACTION      ############
####################################################################  
ParamsRTransY = SystemParameters.ParamsSystem()
ParamsRTransY.name = sTrainRTransY.name
ParamsRTransY.network = linearNetwork4L
ParamsRTransY.iTrain = iTrainRTransY
ParamsRTransY.sTrain = sTrainRTransY
ParamsRTransY.iSeenid = iSeenidRTransY
ParamsRTransY.sSeenid = sSeenidRTransY
ParamsRTransY.iNewid = iNewidRTransY
ParamsRTransY.sNewid = sNewidRTransY
##MEGAWARNING!!!!
#ParamsRTransY.iNewid = iNewidTransY
#ParamsRTransY.sNewid = sNewidTransY
#ParamsRTransY.sNewid.translations_y = ParamsRTransY.sNewid.translations_y * 0.0 + 8.0

ParamsRTransY.block_size = iTrainRTransY.block_size
ParamsRTransY.train_mode = 'mixed'
ParamsRTransY.analysis = None
#Gaussian classifier:
#7 => 6.81
#10 => 6.64
#12 => 6.68
#15 =>
#17 => 6.68




#SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
#This network actually supports images in the closed intervals: [-dx1, dx1], [-dy1, dy1], [smin, smax]
print "Studienprojekt. Scale estimation datasets. By Jan, Stephan and Alberto "
print "***** Setting Training Information Parameters for Scale ******"
iSeq = iTrainRScale = SystemParameters.ParamsInput()
iSeq.name = "Real Scale: (0.55, 1.1,  50)"

iSeq.data_base_dir = alldbnormalized_base_dir
alldbnormalized_available_images = numpy.arange(0,55000)
numpy.random.shuffle(alldbnormalized_available_images)

iSeq.ids = alldbnormalized_available_images[0:30000]

iSeq.scales = numpy.linspace(pipeline_fd['smin0'], pipeline_fd['smax0'], 50) # (-50, 50, 2)
if len(iSeq.ids) % len(iSeq.scales) != 0:
    ex="Here the number of scales must be a divisor of the number of identities"
    raise Exception(ex)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=5, image_postfix=".jpg")
iSeq.input_files = iSeq.input_files * 2 # 2, 10
#To avoid grouping similar images next to one other
numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.scales)
print "BLOCK SIZE =", iSeq.block_size 
iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.scales)), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.scales, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Training Data Parameters for Real Scale  ****************"
sSeq = sTrainRScale = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 135
sSeq.subimage_height = 135

sSeq.trans_x_max = pipeline_fd['dx1']
sSeq.trans_x_min = -1 * pipeline_fd['dx1']
sSeq.trans_y_max = pipeline_fd['dy1']
sSeq.trans_y_min = -1 * pipeline_fd['dy1']
sSeq.min_sampling = pipeline_fd['smin0']
sSeq.max_sampling = pipeline_fd['smax0']
 
sSeq.pixelsampling_x = sfa_libs.wider_1Darray(iSeq.scales,  iSeq.block_size)
sSeq.pixelsampling_y =  sfa_libs.wider_1Darray(iSeq.scales,  iSeq.block_size)
sSeq.subimage_pixelsampling = 2
sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None

#random translation for th w coordinate
sSeq.translation = 20 # 25, Should be 20!!! Warning: 25
sSeq.translations_x = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, sSeq.num_images)
sSeq.translations_y = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, sSeq.num_images)

sSeq.trans_sampled = True
sSeq.name = "Scale. Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f, 50)"%(sSeq.trans_x_min, 
    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.min_sampling*100, sSeq.max_sampling*100)
iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)


print "***** Setting SeenId Information Parameters for Scale ******"
iSeq = iSeenidRScale = SystemParameters.ParamsInput()
iSeq.name = "Real Scale: (0.55, 1.1 / 50)"
iSeq.data_base_dir = alldbnormalized_base_dir
iSeq.ids = alldbnormalized_available_images[30000:45000]

iSeq.scales = numpy.linspace(sTrainRScale.min_sampling, sTrainRScale.max_sampling, 50) # (-50, 50, 2)
if len(iSeq.ids) % len(iSeq.scales) != 0:
    ex="Here the number of scales must be a divisor of the number of identities"
    raise Exception(ex)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=5, image_postfix=".jpg")
iSeq.input_files = iSeq.input_files * 2 # 3 Warning, 20, 32
#To avoid grouping similar images next to one other
numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.scales)
print "BLOCK SIZE =", iSeq.block_size 
iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.scales)), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.scales, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting SeenId Data Parameters for Real Scale  ****************"
sSeq = sSeenidRScale = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 135
sSeq.subimage_height = 135 

sSeq.trans_x_max = sTrainRScale.trans_x_max
sSeq.trans_x_min = sTrainRScale.trans_x_min
sSeq.trans_y_max = sTrainRScale.trans_y_max
sSeq.trans_y_min = sTrainRScale.trans_y_min
sSeq.min_sampling = sTrainRScale.min_sampling
sSeq.max_sampling = sTrainRScale.max_sampling

sSeq.pixelsampling_x = sfa_libs.wider_1Darray(iSeq.scales,  iSeq.block_size)
sSeq.pixelsampling_y =  sfa_libs.wider_1Darray(iSeq.scales,  iSeq.block_size)
sSeq.subimage_pixelsampling = 2
sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
#random translation for th w coo8rdinate
sSeq.translation = 8 # 25, Should be 20!!! Warning: 25
sSeq.translations_x = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, sSeq.num_images)
sSeq.translations_y = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, sSeq.num_images)

sSeq.trans_sampled = True
sSeq.name = "Scale. Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f, 50)"%(sSeq.trans_x_min, 
    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.min_sampling*100, sSeq.max_sampling*100)
iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)

print "***** Setting NewId Information Parameters for Scale ******"
iSeq = iNewidRScale = SystemParameters.ParamsInput()
iSeq.name = "Real Scale: (0.5, 1, 50)"
iSeq.data_base_dir = alldbnormalized_base_dir
iSeq.ids = alldbnormalized_available_images[45000:55000]
iSeq.scales = numpy.linspace(sTrainRScale.min_sampling, sTrainRScale.max_sampling, 50) # (-50, 50, 2)
if len(iSeq.ids) % len(iSeq.scales) != 0:
    ex="Here the number of scales must be a divisor of the number of identities"
    raise Exception(ex)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=5, image_postfix=".jpg")
iSeq.input_files = iSeq.input_files * 1 #8
#To avoid grouping similar images next to one other
numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.scales)
print "BLOCK SIZE =", iSeq.block_size 
iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.scales)), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.scales, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting NewId Data Parameters for Real Scale  ****************"
sSeq = sNewidRScale = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 135
sSeq.subimage_height = 135 

sSeq.trans_x_max = sTrainRScale.trans_x_max
sSeq.trans_x_min = sTrainRScale.trans_x_min
sSeq.trans_y_max = sTrainRScale.trans_y_max
sSeq.trans_y_min = sTrainRScale.trans_y_min
sSeq.min_sampling = sTrainRScale.min_sampling
sSeq.max_sampling = sTrainRScale.max_sampling

sSeq.pixelsampling_x = sfa_libs.wider_1Darray(iSeq.scales,  iSeq.block_size)
sSeq.pixelsampling_y =  sfa_libs.wider_1Darray(iSeq.scales,  iSeq.block_size)
sSeq.subimage_pixelsampling = 2
sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
#random translation for th w coordinate
sSeq.translation = 8 # 25, Should be 20!!! Warning: 25
sSeq.translations_x = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, sSeq.num_images)
sSeq.translations_y = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, sSeq.num_images)

sSeq.trans_sampled = True
sSeq.name = "Scale. Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f, 50)"%(sSeq.trans_x_min, 
    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.min_sampling*100, sSeq.max_sampling*100)
iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)

####################################################################
###########    SYSTEM FOR REAL_SCALE EXTRACTION      ############
####################################################################  
ParamsRScale = SystemParameters.ParamsSystem()
ParamsRScale.name = sTrainRScale.name
ParamsRScale.network = linearNetwork4L
ParamsRScale.iTrain = iTrainRScale
ParamsRScale.sTrain = sTrainRScale
ParamsRScale.iSeenid = iSeenidRScale
ParamsRScale.sSeenid = sSeenidRScale
ParamsRScale.iNewid = iNewidRScale
ParamsRScale.sNewid = sNewidRScale
##MEGAWARNING!!!!
#ParamsRScale.iNewid = iNewidScale
#ParamsRScale.sNewid = sNewidScale
#ParamsRScale.sNewid.translations_y = ParamsRScale.sNewid.translations_y * 0.0 + 8.0

ParamsRScale.block_size = iTrainRScale.block_size
ParamsRScale.train_mode = 'mixed'
ParamsRScale.analysis = None
ParamsRScale.enable_reduced_image_sizes = True
ParamsRScale.reduction_factor = 2.0 # WARNING 2, 4
ParamsRScale.hack_image_size = 64 # WARNING 64, 32
ParamsRScale.enable_hack_image_size = True


#GC (see text file)
# 4 => 0.00406
# 5 => 0.004042
# 15 => 0.00315

#b=[]
#flow, layers, benchmark = CreateNetwork(linearNetworkT6L, 128, 128, 100, 'mixed', b)


print "Studienprojekt. Illumination estimation datasets. By Jan and Stephan and Alberto "
print "***** Setting Training Information Parameters for Illumination ******"
iSeq = iTrainRIllumination = SystemParameters.ParamsInput()
iSeq.name = "Real Illumination:"
on_Jan = os.path.lexists("/home/jan")
if on_lok21:
    pathIllumination = "/local2/escalafl/Alberto/Erg"
if on_Jan:
    pathIllumination = "/home/jan/Dokumente/Studienprojekt/Pictures Illumination Cars/NewImages"
elif on_zappa01:
    pathIllumination = "/local/escalafl/Alberto/NewImages"
else:
    pathIllumination = "/local/escalafl/Alberto/Erg"

available_cars = numpy.arange(1,32)
numpy.random.shuffle(available_cars)

iSeq.data_base_dir = pathIllumination
iSeq.ids = available_cars[0:22]
iSeq.illumination = numpy.arange(0,870)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = iSeq.illumination
iSeq.slow_signal = 7 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "car", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=3, image_postfix=".bmp")
iSeq.input_files = iSeq.input_files * 1
#To avoid grouping similar images next to one other
#numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = 10*iSeq.num_images / len (iSeq.lightings)
print "BLOCK SIZE =", iSeq.block_size 
#print iSeq.block_size
iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.lightings)/10), iSeq.block_size)
#print len(iSeq.correct_classes)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.lightings/10, iSeq.block_size/10)
#print len(iSeq.correct_labels)


SystemParameters.test_object_contents(iSeq)

print "******** Setting Training Data Parameters for Real Illumination  ****************"
sSeq = sTrainRIllumination = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 150
sSeq.image_height = 150
sSeq.subimage_width = 64
sSeq.subimage_height = 64
sSeq.min_sampling = 1.7
sSeq.max_sampling = 1.8
sSeq.pixelsampling_y = sSeq.pixelsampling_x = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, sSeq.num_images)
sSeq.subimage_pixelsampling = 1
sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
#random translation for th w coordinate
sSeq.translation = 3 # 25, Should be 20!!! Warning: 25
sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)          
#sSeq.translations_x = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
sSeq.translations_y = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)
sSeq.trans_sampled = False
SystemParameters.test_object_contents(sSeq)


iSeq = iSeenidRIllumination = SystemParameters.ParamsInput()
iSeq.name = "Real Illumination: "
iSeq.data_base_dir = pathIllumination
iSeq.ids = available_cars[22:27]
iSeq.illumination = numpy.arange(0,870)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = iSeq.illumination
iSeq.slow_signal = 7 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "car", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=3, image_postfix=".bmp")
iSeq.input_files = iSeq.input_files * 1
#To avoid grouping similar images next to one other
#numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = 10*iSeq.num_images / len (iSeq.lightings)
print "BLOCK SIZE =", iSeq.block_size 
iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.lightings)/10), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.lightings/10, iSeq.block_size/10)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Training Data Parameters for Real Illumination  ****************"
sSeq = sSeenidRIllumination = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 150
sSeq.image_height = 150
sSeq.subimage_width = 64
sSeq.subimage_height = 64
sSeq.min_sampling = 1.7
sSeq.max_sampling = 1.8
sSeq.pixelsampling_y = sSeq.pixelsampling_x = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, sSeq.num_images)
sSeq.subimage_pixelsampling = 1
sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
#random translation for th w coordinate
sSeq.translation = 3 # 25, Should be 20!!! Warning: 25
sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)          
#sSeq.translations_x = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
sSeq.translations_y = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)
sSeq.trans_sampled = False
SystemParameters.test_object_contents(sSeq)


iSeq = iNewidRIllumination = SystemParameters.ParamsInput()
iSeq.name = "Real Illumination: "
iSeq.data_base_dir = pathIllumination
iSeq.ids = available_cars[27:32]
iSeq.illumination = numpy.arange(0,870)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = iSeq.illumination
iSeq.slow_signal = 7 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "car", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=3, image_postfix=".bmp")
iSeq.input_files = iSeq.input_files * 1
#To avoid grouping similar images next to one other
#numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = 10*iSeq.num_images / len (iSeq.lightings)
print "BLOCK SIZE =", iSeq.block_size 
iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.lightings)/10), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.lightings/10, iSeq.block_size/10)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Testing Data Parameters for Real Illumination  ****************"
sSeq = sNewidRIllumination = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 150
sSeq.image_height = 150
sSeq.subimage_width = 64
sSeq.subimage_height = 64
sSeq.min_sampling = 1.7
sSeq.max_sampling = 1.8
sSeq.pixelsampling_y = sSeq.pixelsampling_x = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, sSeq.num_images)
sSeq.subimage_pixelsampling = 1
sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
#random translation for th w coordinate
sSeq.translation = 3 # 25, Should be 20!!! Warning: 25
sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)          
#sSeq.translations_x = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
sSeq.translations_y = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)
sSeq.trans_sampled = False
SystemParameters.test_object_contents(sSeq)

####################################################################
###########    SYSTEM FOR REAL_TRANSLATION_X EXTRACTION      ############
####################################################################  
ParamsRIllumination = SystemParameters.ParamsSystem()
ParamsRIllumination.name = "Network that extracts Real Translation X information"
ParamsRIllumination.network = linearNetwork4L
ParamsRIllumination.iTrain = iTrainRIllumination
ParamsRIllumination.sTrain = sTrainRIllumination
ParamsRIllumination.iSeenid = iSeenidRIllumination
ParamsRIllumination.sSeenid = sSeenidRIllumination
ParamsRIllumination.iNewid = iNewidRIllumination
ParamsRIllumination.sNewid = sNewidRIllumination
##MEGAWARNING!!!!
#ParamsRIllumination.iNewid = iNewidIllumination
#ParamsRIllumination.sNewid = sNewidIllumination
#ParamsRIllumination.sNewid.translations_y = ParamsRIllumination.sNewid.translations_y * 0.0 + 8.0

ParamsRIllumination.block_size = iTrainRIllumination.block_size
ParamsRIllumination.train_mode = 'mixed'
ParamsRIllumination.analysis = None

#b=[]
#flow, layers, benchmark = CreateNetwork(linearNetworkT6L, 128, 128, 100, 'mixed', b)


print "Studienprojekt. Rotation estimation datasets. By Jan and Stephan and Alberto "
print "***** Setting Training Information Parameters for Rotation ******"
iSeq = iTrainRRotation = SystemParameters.ParamsInput()
iSeq.name = "Real Rotation: "
on_Jan = os.path.lexists("/home/jan")
if on_lok21:
    pathRotation = "/local2/escalafl/Alberto/Erg"
if on_Jan:
    pathRotation = "/media/7270C6F570C6BF5B/Pictures Rotation/Single Pictures"
else:
    pathRotation = "/home/Stephan/Erg"
available_cars = numpy.arange(1,40)
numpy.random.shuffle(available_cars)

iSeq.data_base_dir = pathRotation
iSeq.ids = available_cars[0:28]
iSeq.illumination = numpy.arange(0,500)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = iSeq.illumination
iSeq.slow_signal = 7 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "car", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=3, image_postfix=".bmp")
iSeq.input_files = iSeq.input_files * 1 

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = 10*iSeq.num_images / len (iSeq.lightings)
print "BLOCK SIZE =", iSeq.block_size 
#print iSeq.block_size
iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.lightings)/10), iSeq.block_size)
#print len(iSeq.correct_classes)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.lightings/10, iSeq.block_size/10)
#print len(iSeq.correct_labels)
SystemParameters.test_object_contents(iSeq)


print "******** Setting Training Data Parameters for Real Rotation  ****************"
sSeq = sTrainRRotation = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 80
sSeq.image_height = 80
sSeq.subimage_width = 64
sSeq.subimage_height = 64
sSeq.min_sampling = 1.0
sSeq.max_sampling = 1.0
sSeq.pixelsampling_y = sSeq.pixelsampling_x = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, sSeq.num_images)
sSeq.subimage_pixelsampling = 1
sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
#random translation for th w coordinate
sSeq.translation = 3 # 25, Should be 20!!! Warning: 25
sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)          
#sSeq.translations_x = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
sSeq.translations_y = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)
sSeq.trans_sampled = False
SystemParameters.test_object_contents(sSeq)


iSeq = iSeenidRRotation = SystemParameters.ParamsInput()
iSeq.name = "Real Rotation: "
iSeq.data_base_dir = pathRotation
iSeq.ids = available_cars[28:34]
iSeq.illumination = numpy.arange(0,500)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = iSeq.illumination
iSeq.slow_signal = 7 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "car", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=3, image_postfix=".bmp")
iSeq.input_files = iSeq.input_files * 1
#To avoid grouping similar images next to one other
#numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = 10*iSeq.num_images / len (iSeq.lightings)
print "BLOCK SIZE =", iSeq.block_size 
iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.lightings)/10), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.lightings/10, iSeq.block_size/10)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Training Data Parameters for Real Rotation  ****************"
sSeq = sSeenidRRotation = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 80
sSeq.image_height = 80
sSeq.subimage_width = 64
sSeq.subimage_height = 64
sSeq.min_sampling = 1.0
sSeq.max_sampling = 1.0
sSeq.pixelsampling_y = sSeq.pixelsampling_x = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, sSeq.num_images)
sSeq.subimage_pixelsampling = 1
sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
#random translation for th w coordinate
sSeq.translation = 3 # 25, Should be 20!!! Warning: 25
sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)          
#sSeq.translations_x = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
sSeq.translations_y = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)
sSeq.trans_sampled = False
SystemParameters.test_object_contents(sSeq)


iSeq = iNewidRRotation = SystemParameters.ParamsInput()
iSeq.name = "Real Rotation: "
iSeq.data_base_dir = pathRotation
iSeq.ids = available_cars[34:40]
iSeq.illumination = numpy.arange(0,500)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = iSeq.illumination
iSeq.slow_signal = 7 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "car", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=3, image_postfix=".bmp")
iSeq.input_files = iSeq.input_files * 1
#To avoid grouping similar images next to one other
#numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = 10*iSeq.num_images / len (iSeq.lightings)
print "BLOCK SIZE =", iSeq.block_size 
iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.lightings)/10), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.lightings/10, iSeq.block_size/10)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Testing Data Parameters for Real Illumination  ****************"
sSeq = sNewidRRotation = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 80
sSeq.image_height = 80
sSeq.subimage_width = 64
sSeq.subimage_height = 64
sSeq.min_sampling = 1.0
sSeq.max_sampling = 1.0
sSeq.pixelsampling_y = sSeq.pixelsampling_x = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, sSeq.num_images)
sSeq.subimage_pixelsampling = 1
sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
#random translation for th w coordinate
sSeq.translation = 3 # 25, Should be 20!!! Warning: 25
sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)          
#sSeq.translations_x = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
sSeq.translations_y = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)
sSeq.trans_sampled = False
SystemParameters.test_object_contents(sSeq)

####################################################################
###########    SYSTEM FOR REAL_TRANSLATION_X EXTRACTION      ############
####################################################################  
ParamsRRotation = SystemParameters.ParamsSystem()
ParamsRRotation.name = "Network that extracts Real Translation X information"
ParamsRRotation.network = linearNetwork4L
ParamsRRotation.iTrain = iTrainRRotation
ParamsRRotation.sTrain = sTrainRRotation
ParamsRRotation.iSeenid = iSeenidRRotation
ParamsRRotation.sSeenid = sSeenidRRotation
ParamsRRotation.iNewid = iNewidRRotation
ParamsRRotation.sNewid = sNewidRRotation
ParamsRRotation.block_size = iTrainRRotation.block_size
ParamsRRotation.train_mode = 'mixed'
ParamsRRotation.analysis = None


def double_uniform(min1, max1, min2, max2, size, p2):
    prob = numpy.random.uniform(0.0, 1.0, size)
    u1 = numpy.random.uniform(min1, max1, size)
    u2 = numpy.random.uniform(min2, max2, size)
    
    res = u1
    mask = (prob <= p2) #then take it from u2
    res[mask] = u2[mask]
    return res

#TODO: code is slow, improve
#Box is a list of pairs. Pair i contains the smallest and largest value for coordinate i
def box_sampling(box, num_samples=1):
    num_dimensions = len(box)
    output = numpy.zeros((num_samples, num_dimensions))
    for i in range(num_dimensions):
        output[:,i] = numpy.random.uniform(box[i][0], box[i][1], size=num_samples)
    return output

#x must be a two-dimensional array
def inside_box(x, box, num_dim):
    num_samples = len(x)
    inside = numpy.ones(num_samples, dtype="bool")
    for i in range(num_dim):
        inside = inside & (x[:, i] > box[i][0]) & (x[:, i] < box[i][1])    
    return inside
        
#TODO: code is slow, improve   
def sub_box_sampling(box_in, box_ext, num_samples=1):
    num_dimensions = len(box_in)
    if num_dimensions != len(box_ext):
        err = "Exterion and interior boxes have a different numbe of dimensions!!!"
        raise Exception(err)
    output = numpy.zeros((num_samples, num_dimensions))
    incorrect = numpy.ones(num_samples, dtype="bool")
    while incorrect.sum()>0:
#        print "incorrect.sum()=",incorrect.sum()
        new_candidates = box_sampling(box_ext, incorrect.sum()) 
        output[incorrect] = new_candidates       
        incorrect = inside_box(output, box_in, num_dimensions)        
    return output

#FACE / NO-FACE FACE / NO-FACE FACE / NO-FACE FACE / NO-FACE FACE / NO-FACE FACE / NO-FACE FACE / NO-FACE FACE / NO-FACE
#Attempt to distinguish between faces and no-faces
print "***** Setting Training Information Parameters for Face ******"
iSeq = iTrainRFace = SystemParameters.ParamsInput()
# (0.55+1.1)/2 = 0.825, 0.55/2 = 0.275, 0.55/4 = 0.1375, .825 + .1375 = .9625, .825 - .55/4 = .6875
iSeq.name = "Real FACE (Centered / Decentered)"

iSeq.data_base_dir = frgc_normalized_base_dir
iSeq.ids = numpy.arange(0,6000) # 6000
iSeq.faces = numpy.arange(0,2) # 0=centered normalized face, 1=not centered normalized face

#iSeq.scales = numpy.linspace(pipeline_fd['smin0'], pipeline_fd['smax0'], 50) # (-50, 50, 2)
if len(iSeq.ids) % len(iSeq.faces) != 0:
    ex="Here the number of scales must be a divisor of the number of identities"
    raise Exception(ex)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is whether there is a centered or descentered face
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=4, image_postfix=".jpg")
iSeq.input_files = iSeq.input_files * 4 #4 was overfitting non linear sfa slightly
#To avoid grouping similar images next to one other
numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.faces)
print "BLOCK SIZE =", iSeq.block_size 

iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.faces)), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.faces*2-1, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Training Data Parameters for Real Face  ****************"
sSeq = sTrainRFace = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 135
sSeq.subimage_height = 135

sSeq.trans_x_max = pipeline_fd['dx1']
sSeq.trans_x_min = -1 * pipeline_fd['dx1']
sSeq.trans_y_max = pipeline_fd['dy1']
sSeq.trans_y_min = -1 * pipeline_fd['dy1']
sSeq.min_sampling = pipeline_fd['smin1']
sSeq.max_sampling = pipeline_fd['smax1']

sSeq.noface_trans_x_max = 45
sSeq.noface_trans_x_min = -45
sSeq.noface_trans_y_max = 19
sSeq.noface_trans_y_min = -19
sSeq.noface_min_sampling = 0.55
sSeq.noface_max_sampling = 1.1
 

sSeq.pixelsampling_x = numpy.zeros(sSeq.num_images)
sSeq.pixelsampling_y = numpy.zeros(sSeq.num_images) 

#Centered Face
sSeq.pixelsampling_x[0:iSeq.block_size] = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, size=iSeq.block_size)
sSeq.pixelsampling_y[0:iSeq.block_size] = sSeq.pixelsampling_x[0:iSeq.block_size] + 0.0 #MUST BE A DIFFERENT OBJECT
#sSeq.pixelsampling_x[iSeq.block_size:] = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, size=iSeq.block_size)
#sSeq.pixelsampling_y[iSeq.block_size:] = sSeq.pixelsampling_x[iSeq.block_size:] + 0.0
#Decentered Face, using different x and y samplings
sSeq.pixelsampling_x[iSeq.block_size:] = double_uniform(sSeq.noface_min_sampling, sSeq.min_sampling, sSeq.max_sampling, sSeq.noface_max_sampling , size=iSeq.block_size,   p2=0.5)
sSeq.pixelsampling_y[iSeq.block_size:] = double_uniform(sSeq.noface_min_sampling, sSeq.min_sampling, sSeq.max_sampling, sSeq.noface_max_sampling , size=iSeq.block_size,   p2=0.5)

sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None

#random translation for th w coordinate
#sSeq.translation = 20 # 25, Should be 20!!! Warning: 25
sSeq.translations_x = numpy.zeros(sSeq.num_images)
sSeq.translations_y = numpy.zeros(sSeq.num_images)
#Centered Face
sSeq.translations_x[0:iSeq.block_size] = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, iSeq.block_size)
sSeq.translations_y[0:iSeq.block_size] = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, iSeq.block_size)
#sSeq.translations_x[iSeq.block_size:] = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, iSeq.block_size)
#sSeq.translations_y[iSeq.block_size:] = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, iSeq.block_size)
#Decentered Face
sSeq.translations_x[iSeq.block_size:] = double_uniform(sSeq.noface_trans_x_min, sSeq.trans_x_min, sSeq.trans_x_max, sSeq.noface_trans_x_max, size=iSeq.block_size, p2=0.5)
sSeq.translations_y[iSeq.block_size:] = double_uniform(sSeq.noface_trans_y_min, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.noface_trans_y_max, size=iSeq.block_size, p2=0.5)

sSeq.trans_sampled = True
sSeq.name = "Face. Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f, 50)"%(sSeq.trans_x_min, 
    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, int(sSeq.min_sampling*1000), int(sSeq.max_sampling*1000))


print " sSeq.subimage_first_row =", sSeq.subimage_first_row
print "sSeq.pixelsampling_x", sSeq.pixelsampling_x
print "sSeq.translations_x", sSeq.translations_x

iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)



print "***** Setting Seenid Information Parameters for Face ******"
iSeq = iSeenidRFace = SystemParameters.ParamsInput()
# (0.55+1.1)/2 = 0.825, 0.55/2 = 0.275, 0.55/4 = 0.1375, .825 + .1375 = .9625, .825 - .55/4 = .6875
iSeq.name = "Real FACE (Centered / Decentered)"

iSeq.data_base_dir = frgc_normalized_base_dir
iSeq.ids = numpy.arange(6000,8000) # 8000
iSeq.faces = numpy.arange(0,2) # 0=centered normalized face, 1=not centered normalized face

#iSeq.scales = numpy.linspace(pipeline_fd['smin0'], pipeline_fd['smax0'], 50) # (-50, 50, 2)
if len(iSeq.ids) % len(iSeq.faces) != 0:
    ex="Here the number of scales must be a divisor of the number of identities"
    raise Exception(ex)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is whether there is a centered or descentered face
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=4, image_postfix=".jpg")
iSeq.input_files = iSeq.input_files * 8
#To avoid grouping similar images next to one other
numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.faces)
print "BLOCK SIZE =", iSeq.block_size 

iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.faces)), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.faces*2-1, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting SeenID Data Parameters for Real Face  ****************"
sSeq = sSeenidRFace = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 135
sSeq.subimage_height = 135

sSeq.trans_x_max = pipeline_fd['dx1']
sSeq.trans_x_min = -1 * pipeline_fd['dx1']
sSeq.trans_y_max = pipeline_fd['dy1']
sSeq.trans_y_min = -1 * pipeline_fd['dy1']
sSeq.min_sampling = pipeline_fd['smin1']
sSeq.max_sampling = pipeline_fd['smax1']

sSeq.noface_trans_x_max = 45
sSeq.noface_trans_x_min = -45
sSeq.noface_trans_y_max = 19
sSeq.noface_trans_y_min = -19
sSeq.noface_min_sampling = 0.55
sSeq.noface_max_sampling = 1.1
 

sSeq.pixelsampling_x = numpy.zeros(sSeq.num_images)
sSeq.pixelsampling_y = numpy.zeros(sSeq.num_images) 

#Centered Face
sSeq.pixelsampling_x[0:iSeq.block_size] = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, size=iSeq.block_size)
sSeq.pixelsampling_y[0:iSeq.block_size] = sSeq.pixelsampling_x[0:iSeq.block_size] + 0.0 #MUST BE A DIFFERENT OBJECT
#sSeq.pixelsampling_x[iSeq.block_size:] = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, size=iSeq.block_size)
#sSeq.pixelsampling_y[iSeq.block_size:] = sSeq.pixelsampling_x[iSeq.block_size:] + 0.0
#Decentered Face, using different x and y samplings
sSeq.pixelsampling_x[iSeq.block_size:] = double_uniform(sSeq.noface_min_sampling, sSeq.min_sampling, sSeq.max_sampling, sSeq.noface_max_sampling , size=iSeq.block_size,   p2=0.5)
sSeq.pixelsampling_y[iSeq.block_size:] = double_uniform(sSeq.noface_min_sampling, sSeq.min_sampling, sSeq.max_sampling, sSeq.noface_max_sampling , size=iSeq.block_size,   p2=0.5)

sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None

#random translation for th w coordinate
#sSeq.translation = 20 # 25, Should be 20!!! Warning: 25
sSeq.translations_x = numpy.zeros(sSeq.num_images)
sSeq.translations_y = numpy.zeros(sSeq.num_images)
#Centered Face
sSeq.translations_x[0:iSeq.block_size] = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, iSeq.block_size)
sSeq.translations_y[0:iSeq.block_size] = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, iSeq.block_size)
#Decentered Face
sSeq.translations_x[iSeq.block_size:] = double_uniform(sSeq.noface_trans_x_min, sSeq.trans_x_min, sSeq.trans_x_max, sSeq.noface_trans_x_max, size=iSeq.block_size, p2=0.5)
sSeq.translations_y[iSeq.block_size:] = double_uniform(sSeq.noface_trans_y_min, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.noface_trans_y_max, size=iSeq.block_size, p2=0.5)

sSeq.trans_sampled = True
sSeq.name = "Face. Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f, 50)"%(sSeq.trans_x_min, 
    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, int(sSeq.min_sampling*1000), int(sSeq.max_sampling*1000))

iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)


print "***** Setting Newid Information Parameters for Face ******"
iSeq = iNewidRFace = SystemParameters.ParamsInput()
# (0.55+1.1)/2 = 0.825, 0.55/2 = 0.275, 0.55/4 = 0.1375, .825 + .1375 = .9625, .825 - .55/4 = .6875
iSeq.name = "Real FACE (Centered / Decentered)"

iSeq.data_base_dir = frgc_normalized_base_dir
iSeq.ids = numpy.arange(8000,10000) # 8000
iSeq.faces = numpy.arange(0,2) # 0=centered normalized face, 1=not centered normalized face

#iSeq.scales = numpy.linspace(pipeline_fd['smin0'], pipeline_fd['smax0'], 50) # (-50, 50, 2)
if len(iSeq.ids) % len(iSeq.faces) != 0:
    ex="Here the number of scales must be a divisor of the number of identities"
    raise Exception(ex)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is whether there is a centered or descentered face
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=4, image_postfix=".jpg")
iSeq.input_files = iSeq.input_files * 2
#To avoid grouping similar images next to one other
numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.faces)
print "BLOCK SIZE =", iSeq.block_size 

iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.faces)), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.faces*2-1, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Training Data Parameters for Real Face  ****************"
sSeq = sNewidRFace = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 135
sSeq.subimage_height = 135

sSeq.trans_x_max = pipeline_fd['dx1']
sSeq.trans_x_min = -1 * pipeline_fd['dx1']
sSeq.trans_y_max = pipeline_fd['dy1']
sSeq.trans_y_min = -1 * pipeline_fd['dy1']
sSeq.min_sampling = pipeline_fd['smin1']
sSeq.max_sampling = pipeline_fd['smax1']

sSeq.noface_trans_x_max = 45
sSeq.noface_trans_x_min = -45
sSeq.noface_trans_y_max = 19
sSeq.noface_trans_y_min = -19
sSeq.noface_min_sampling = 0.55
sSeq.noface_max_sampling = 1.1
 

sSeq.pixelsampling_x = numpy.zeros(sSeq.num_images)
sSeq.pixelsampling_y = numpy.zeros(sSeq.num_images) 

#Centered Face
sSeq.pixelsampling_x[0:iSeq.block_size] = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, size=iSeq.block_size)
sSeq.pixelsampling_y[0:iSeq.block_size] = sSeq.pixelsampling_x[0:iSeq.block_size] + 0.0 #MUST BE A DIFFERENT OBJECT
#sSeq.pixelsampling_x[iSeq.block_size:] = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, size=iSeq.block_size)
#sSeq.pixelsampling_y[iSeq.block_size:] = sSeq.pixelsampling_x[iSeq.block_size:] + 0.0
#Decentered Face, using different x and y samplings
sSeq.pixelsampling_x[iSeq.block_size:] = double_uniform(sSeq.noface_min_sampling, sSeq.min_sampling, sSeq.max_sampling, sSeq.noface_max_sampling , size=iSeq.block_size,   p2=0.5)
sSeq.pixelsampling_y[iSeq.block_size:] = double_uniform(sSeq.noface_min_sampling, sSeq.min_sampling, sSeq.max_sampling, sSeq.noface_max_sampling , size=iSeq.block_size,   p2=0.5)

sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
#sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None

#random translation for th w coordinate
#sSeq.translation = 20 # 25, Should be 20!!! Warning: 25
sSeq.translations_x = numpy.zeros(sSeq.num_images)
sSeq.translations_y = numpy.zeros(sSeq.num_images)
#Centered Face
sSeq.translations_x[0:iSeq.block_size] = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, iSeq.block_size)
sSeq.translations_y[0:iSeq.block_size] = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, iSeq.block_size)
#Decentered Face
sSeq.translations_x[iSeq.block_size:] = double_uniform(sSeq.noface_trans_x_min, sSeq.trans_x_min, sSeq.trans_x_max, sSeq.noface_trans_x_max, size=iSeq.block_size, p2=0.5)
sSeq.translations_y[iSeq.block_size:] = double_uniform(sSeq.noface_trans_y_min, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.noface_trans_y_max, size=iSeq.block_size, p2=0.5)

sSeq.trans_sampled = True
sSeq.name = "Face. Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f, 50)"%(sSeq.trans_x_min, 
    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, int(sSeq.min_sampling*1000), int(sSeq.max_sampling*1000))

iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)


####################################################################
###########    SYSTEM FOR REAL_FACE CLASSIFICATION      ############
####################################################################  
ParamsRFace = SystemParameters.ParamsSystem()
ParamsRFace.name = sTrainRFace.name
ParamsRFace.network = linearNetwork4L
ParamsRFace.iTrain = iTrainRFace
ParamsRFace.sTrain = sTrainRFace
ParamsRFace.iSeenid = iSeenidRFace
ParamsRFace.sSeenid = sSeenidRFace
ParamsRFace.iNewid = iNewidRFace 
ParamsRFace.sNewid = sNewidRFace 
##MEGAWARNING!!!!
#ParamsRFace.iNewid = iNewidScale
#ParamsRFace.sNewid = sNewidScale
#ParamsRFace.sNewid.translations_y = ParamsRFace.sNewid.translations_y * 0.0 + 8.0

ParamsRFace.block_size = iTrainRFace.block_size
ParamsRFace.train_mode = 'clustered' # 'mixed'
ParamsRFace.analysis = None

import object_cache as cache

robject_center_base_dir = "/local/escalafl/Alberto/FaubelSet_01/Center"
robject_down_base_dir = "/local/escalafl/Alberto/FaubelSet_01/Down"
robject_up_base_dir = "/local/escalafl/Alberto/FaubelSet_01/Up"

#Dirty function to deal with filename convention obj#-##__#-##_ ... and  to acces several directories at once
def find_filenames_beginning_with_numbers(base_dirs=[""], base_filename="obj", base_numbers=None, extension=".png"):
    if base_numbers == None:
        return cache.find_filenames_beginning_with(base_dirs, base_filename, recursion=False, extension=extension)
    else:
        filenames = []
        for n, i in enumerate(base_numbers):
            filenames.append([])
            for base_dir in base_dirs:
                filenames[n].extend(cache.find_filenames_beginning_with(base_dir, base_filename+"%d_"%i, recursion=False, extension=extension))
#            print "looking for %s //  %s"%(base_dir, base_filename+"%d_"%i)
#            print "found:", cache.find_filenames_beginning_with(base_dir, base_filename+"%d"%i, recursion=False, extension=extension)
        return filenames

#Cooperation with Christian Faubel. Object Recognition
#Images by Christian Faubel
#Attempt to distinguish some types of objects
robject_convert_format='L'
print "***** Setting Training Information Parameters for RObject ******"
iSeq = iTrainRObject = SystemParameters.ParamsInput()
# (0.55+1.1)/2 = 0.825, 0.55/2 = 0.275, 0.55/4 = 0.1375, .825 + .1375 = .9625, .825 - .55/4 = .6875
iSeq.name = "Real Object Recognition"

iSeq.data_base_dir = [robject_center_base_dir, robject_down_base_dir, robject_up_base_dir]
iSeq.ids = numpy.arange(1,21) # 1-31
#iSeq.faces = numpy.arange(0,2) # 0=centered normalized face, 1=not centered normalized face
#iSeq.scales = numpy.linspace(pipeline_fd['smin0'], pipeline_fd['smax0'], 50) # (-50, 50, 2)
#if len(iSeq.ids) % len(iSeq.faces) != 0:
#    ex="Here the number of scales must be a divisor of the number of identities"
#    raise Exception(ex)
iSeq.all_input_files = find_filenames_beginning_with_numbers(iSeq.data_base_dir, "obj", iSeq.ids, extension=".png")
#print "iSeq.all_input_files", iSeq.all_input_files

iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = map(len, iSeq.all_input_files)
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is whether there is a centered or descentered face
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = []
for single_file_list in iSeq.all_input_files:
    iSeq.input_files += single_file_list
iSeq.block_sizes = numpy.array(iSeq.poses)
print "totaling %d images"%len(iSeq.input_files)

#imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
#                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
#                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=4, image_postfix=".jpg")
#iSeq.input_files = iSeq.input_files * 8 #4 was overfitting non linear sfa slightly
#To avoid grouping similar images next to one other
#numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]

#iSeq.block_size = 10 # WAAARRNNNIIINNGGG!!!
iSeq.block_size = iSeq.block_sizes

#iSeq.num_images / len (iSeq.faces)
print "BLOCK SIZE =", iSeq.block_size 
print "BLOCK SIZES =", iSeq.block_sizes

iSeq.correct_classes = []
for i, block_size in enumerate(iSeq.block_sizes):
    iSeq.correct_classes += [iSeq.ids[i]]*block_size
iSeq.correct_classes = numpy.array(iSeq.correct_classes)
print iSeq.correct_classes
#quit()

#sfa_libs.wider_1Darray(numpy.arange(len(iSeq.faces)), iSeq.block_size)
iSeq.correct_labels = iSeq.correct_classes + 0.0
print iSeq.correct_labels



#sfa_libs.wider_1Darray(iSeq.faces*2-1, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Training Data Parameters for Real Object  ****************"
sSeq = sTrainRObject = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 52
sSeq.image_height = 52
sSeq.subimage_width = 32
sSeq.subimage_height = 32

#sSeq.trans_x_max = pipeline_fd['dx1']
#sSeq.trans_x_min = -1 * pipeline_fd['dx1']
#sSeq.trans_y_max = pipeline_fd['dy1']
#sSeq.trans_y_min = -1 * pipeline_fd['dy1']
#sSeq.min_sampling = pipeline_fd['smin1']
#sSeq.max_sampling = pipeline_fd['smax1']
#sSeq.noface_trans_x_max = 45
#sSeq.noface_trans_x_min = -45
#sSeq.noface_trans_y_max = 19
#sSeq.noface_trans_y_min = -19
#sSeq.noface_min_sampling = 0.55
#sSeq.noface_max_sampling = 1.1

sSeq.pixelsampling_x = 1.5 #1.625
sSeq.pixelsampling_y = 1.5

##Centered Face
#sSeq.pixelsampling_x[0:iSeq.block_size] = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, size=iSeq.block_size)
#sSeq.pixelsampling_y[0:iSeq.block_size] = sSeq.pixelsampling_x[0:iSeq.block_size] + 0.0 #MUST BE A DIFFERENT OBJECT
##sSeq.pixelsampling_x[iSeq.block_size:] = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, size=iSeq.block_size)
##sSeq.pixelsampling_y[iSeq.block_size:] = sSeq.pixelsampling_x[iSeq.block_size:] + 0.0
##Decentered Face, using different x and y samplings
#sSeq.pixelsampling_x[iSeq.block_size:] = double_uniform(sSeq.noface_min_sampling, sSeq.min_sampling, sSeq.max_sampling, sSeq.noface_max_sampling , size=iSeq.block_size,   p2=0.5)
#sSeq.pixelsampling_y[iSeq.block_size:] = double_uniform(sSeq.noface_min_sampling, sSeq.min_sampling, sSeq.max_sampling, sSeq.noface_max_sampling , size=iSeq.block_size,   p2=0.5)

sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
#sSeq.subimage_first_column =" sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = robject_convert_format
sSeq.background_type = None

#random translation for th w coordinate
#sSeq.translation = 20 # 25, Should be 20!!! Warning: 25
sSeq.translations_x = numpy.zeros(sSeq.num_images)
sSeq.translations_y = numpy.zeros(sSeq.num_images)
#Centered Face
##sSeq.translations_x[0:iSeq.block_size] = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, iSeq.block_size)
##sSeq.translations_y[0:iSeq.block_size] = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, iSeq.block_size)
###sSeq.translations_x[iSeq.block_size:] = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, iSeq.block_size)
###sSeq.translations_y[iSeq.block_size:] = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, iSeq.block_size)
###Decentered Face
##sSeq.translations_x[iSeq.block_size:] = double_uniform(sSeq.noface_trans_x_min, sSeq.trans_x_min, sSeq.trans_x_max, sSeq.noface_trans_x_max, size=iSeq.block_size, p2=0.5)
##sSeq.translations_y[iSeq.block_size:] = double_uniform(sSeq.noface_trans_y_min, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.noface_trans_y_max, size=iSeq.block_size, p2=0.5)

sSeq.trans_sampled = True
sSeq.name = "Object"
#sSeq.name = "Object. Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f, 50)"%(sSeq.trans_x_min, 
#    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, int(sSeq.min_sampling*1000), int(sSeq.max_sampling*1000))

print " sSeq.subimage_first_row =", sSeq.subimage_first_row
print "sSeq.pixelsampling_x", sSeq.pixelsampling_x
print "sSeq.translations_x", sSeq.translations_x

iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)


#2
print "***** Setting Seenid Information Parameters for RObject ******"
iSeq = iSeenidRObject = SystemParameters.ParamsInput()
# (0.55+1.1)/2 = 0.825, 0.55/2 = 0.275, 0.55/4 = 0.1375, .825 + .1375 = .9625, .825 - .55/4 = .6875
iSeq.name = "Real Object Recognition"

iSeq.data_base_dir = [robject_down_base_dir, robject_up_base_dir]
iSeq.ids = numpy.arange(21,31) # 1-31
#iSeq.faces = numpy.arange(0,2) # 0=centered normalized face, 1=not centered normalized face
#iSeq.scales = numpy.linspace(pipeline_fd['smin0'], pipeline_fd['smax0'], 50) # (-50, 50, 2)
#if len(iSeq.ids) % len(iSeq.faces) != 0:
#    ex="Here the number of scales must be a divisor of the number of identities"
#    raise Exception(ex)
iSeq.all_input_files = find_filenames_beginning_with_numbers(iSeq.data_base_dir, "obj", iSeq.ids, extension=".png")
#print "iSeq.all_input_files", iSeq.all_input_files

iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = map(len, iSeq.all_input_files)
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is whether there is a centered or descentered face
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = []
for single_file_list in iSeq.all_input_files:
    iSeq.input_files += single_file_list
iSeq.block_sizes = numpy.array(iSeq.poses)
print "totaling %d images"%len(iSeq.input_files)

#imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
#                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
#                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=4, image_postfix=".jpg")
#iSeq.input_files = iSeq.input_files * 8 #4 was overfitting non linear sfa slightly
#To avoid grouping similar images next to one other
#numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]

#iSeq.block_size = 10 # WAAARRNNNIIINNGGG!!!
iSeq.block_size = iSeq.block_sizes

#iSeq.num_images / len (iSeq.faces)
print "BLOCK SIZE =", iSeq.block_size 
print "BLOCK SIZES =", iSeq.block_sizes

iSeq.correct_classes = []
for i, block_size in enumerate(iSeq.block_sizes):
    iSeq.correct_classes += [iSeq.ids[i]]*block_size
iSeq.correct_classes = numpy.array(iSeq.correct_classes)
print iSeq.correct_classes
#quit()

#sfa_libs.wider_1Darray(numpy.arange(len(iSeq.faces)), iSeq.block_size)
iSeq.correct_labels = iSeq.correct_classes + 0.0
print iSeq.correct_labels



#sfa_libs.wider_1Darray(iSeq.faces*2-1, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Training Data Parameters for Real Object  ****************"
sSeq = sSeenidRObject = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 52
sSeq.image_height = 52
sSeq.subimage_width = 32
sSeq.subimage_height = 32

#sSeq.trans_x_max = pipeline_fd['dx1']
#sSeq.trans_x_min = -1 * pipeline_fd['dx1']
#sSeq.trans_y_max = pipeline_fd['dy1']
#sSeq.trans_y_min = -1 * pipeline_fd['dy1']
#sSeq.min_sampling = pipeline_fd['smin1']
#sSeq.max_sampling = pipeline_fd['smax1']
#sSeq.noface_trans_x_max = 45
#sSeq.noface_trans_x_min = -45
#sSeq.noface_trans_y_max = 19
#sSeq.noface_trans_y_min = -19
#sSeq.noface_min_sampling = 0.55
#sSeq.noface_max_sampling = 1.1

sSeq.pixelsampling_x = 1.5 #1.625
sSeq.pixelsampling_y = 1.5

##Centered Face
#sSeq.pixelsampling_x[0:iSeq.block_size] = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, size=iSeq.block_size)
#sSeq.pixelsampling_y[0:iSeq.block_size] = sSeq.pixelsampling_x[0:iSeq.block_size] + 0.0 #MUST BE A DIFFERENT OBJECT
##sSeq.pixelsampling_x[iSeq.block_size:] = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, size=iSeq.block_size)
##sSeq.pixelsampling_y[iSeq.block_size:] = sSeq.pixelsampling_x[iSeq.block_size:] + 0.0
##Decentered Face, using different x and y samplings
#sSeq.pixelsampling_x[iSeq.block_size:] = double_uniform(sSeq.noface_min_sampling, sSeq.min_sampling, sSeq.max_sampling, sSeq.noface_max_sampling , size=iSeq.block_size,   p2=0.5)
#sSeq.pixelsampling_y[iSeq.block_size:] = double_uniform(sSeq.noface_min_sampling, sSeq.min_sampling, sSeq.max_sampling, sSeq.noface_max_sampling , size=iSeq.block_size,   p2=0.5)

sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
#sSeq.subimage_first_column =" sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = robject_convert_format
sSeq.background_type = None

#random translation for th w coordinate
#sSeq.translation = 20 # 25, Should be 20!!! Warning: 25
sSeq.translations_x = numpy.zeros(sSeq.num_images)
sSeq.translations_y = numpy.zeros(sSeq.num_images)
#Centered Face
##sSeq.translations_x[0:iSeq.block_size] = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, iSeq.block_size)
##sSeq.translations_y[0:iSeq.block_size] = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, iSeq.block_size)
###sSeq.translations_x[iSeq.block_size:] = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, iSeq.block_size)
###sSeq.translations_y[iSeq.block_size:] = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, iSeq.block_size)
###Decentered Face
##sSeq.translations_x[iSeq.block_size:] = double_uniform(sSeq.noface_trans_x_min, sSeq.trans_x_min, sSeq.trans_x_max, sSeq.noface_trans_x_max, size=iSeq.block_size, p2=0.5)
##sSeq.translations_y[iSeq.block_size:] = double_uniform(sSeq.noface_trans_y_min, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.noface_trans_y_max, size=iSeq.block_size, p2=0.5)

sSeq.trans_sampled = True
sSeq.name = "Object"
#sSeq.name = "Object. Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f, 50)"%(sSeq.trans_x_min, 
#    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, int(sSeq.min_sampling*1000), int(sSeq.max_sampling*1000))

print " sSeq.subimage_first_row =", sSeq.subimage_first_row
print "sSeq.pixelsampling_x", sSeq.pixelsampling_x
print "sSeq.translations_x", sSeq.translations_x

iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)

#3
print "***** Setting Training Information Parameters for RObject ******"
iSeq = iNewidRObject = SystemParameters.ParamsInput()
# (0.55+1.1)/2 = 0.825, 0.55/2 = 0.275, 0.55/4 = 0.1375, .825 + .1375 = .9625, .825 - .55/4 = .6875
iSeq.name = "Real Object Recognition"

iSeq.data_base_dir =  [robject_center_base_dir]
iSeq.ids = numpy.arange(21,31) # 1-31
#iSeq.faces = numpy.arange(0,2) # 0=centered normalized face, 1=not centered normalized face
#iSeq.scales = numpy.linspace(pipeline_fd['smin0'], pipeline_fd['smax0'], 50) # (-50, 50, 2)
#if len(iSeq.ids) % len(iSeq.faces) != 0:
#    ex="Here the number of scales must be a divisor of the number of identities"
#    raise Exception(ex)
iSeq.all_input_files = find_filenames_beginning_with_numbers(iSeq.data_base_dir, "obj", iSeq.ids, extension=".png")
#print "iSeq.all_input_files", iSeq.all_input_files

iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = map(len, iSeq.all_input_files)
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is whether there is a centered or descentered face
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = []
for single_file_list in iSeq.all_input_files:
    iSeq.input_files += single_file_list
iSeq.block_sizes = numpy.array(iSeq.poses)
print "totaling %d images"%len(iSeq.input_files)

#imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
#                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
#                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=4, image_postfix=".jpg")
#iSeq.input_files = iSeq.input_files * 8 #4 was overfitting non linear sfa slightly
#To avoid grouping similar images next to one other
#numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]

#iSeq.block_size = 10 # WAAARRNNNIIINNGGG!!!
iSeq.block_size = iSeq.block_sizes

#iSeq.num_images / len (iSeq.faces)
print "BLOCK SIZE =", iSeq.block_size 
print "BLOCK SIZES =", iSeq.block_sizes

iSeq.correct_classes = []
for i, block_size in enumerate(iSeq.block_sizes):
    iSeq.correct_classes += [iSeq.ids[i]]*block_size
iSeq.correct_classes = numpy.array(iSeq.correct_classes)
print iSeq.correct_classes
#quit()

#sfa_libs.wider_1Darray(numpy.arange(len(iSeq.faces)), iSeq.block_size)
iSeq.correct_labels = iSeq.correct_classes + 0.0
print iSeq.correct_labels



#sfa_libs.wider_1Darray(iSeq.faces*2-1, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Training Data Parameters for Real Object  ****************"
sSeq = sNewidRObject = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 52
sSeq.image_height = 52
sSeq.subimage_width = 32
sSeq.subimage_height = 32

#sSeq.trans_x_max = pipeline_fd['dx1']
#sSeq.trans_x_min = -1 * pipeline_fd['dx1']
#sSeq.trans_y_max = pipeline_fd['dy1']
#sSeq.trans_y_min = -1 * pipeline_fd['dy1']
#sSeq.min_sampling = pipeline_fd['smin1']
#sSeq.max_sampling = pipeline_fd['smax1']
#sSeq.noface_trans_x_max = 45
#sSeq.noface_trans_x_min = -45
#sSeq.noface_trans_y_max = 19
#sSeq.noface_trans_y_min = -19
#sSeq.noface_min_sampling = 0.55
#sSeq.noface_max_sampling = 1.1

sSeq.pixelsampling_x = 1.5 #1.625
sSeq.pixelsampling_y = 1.5

##Centered Face
#sSeq.pixelsampling_x[0:iSeq.block_size] = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, size=iSeq.block_size)
#sSeq.pixelsampling_y[0:iSeq.block_size] = sSeq.pixelsampling_x[0:iSeq.block_size] + 0.0 #MUST BE A DIFFERENT OBJECT
##sSeq.pixelsampling_x[iSeq.block_size:] = numpy.random.uniform(sSeq.min_sampling, sSeq.max_sampling, size=iSeq.block_size)
##sSeq.pixelsampling_y[iSeq.block_size:] = sSeq.pixelsampling_x[iSeq.block_size:] + 0.0
##Decentered Face, using different x and y samplings
#sSeq.pixelsampling_x[iSeq.block_size:] = double_uniform(sSeq.noface_min_sampling, sSeq.min_sampling, sSeq.max_sampling, sSeq.noface_max_sampling , size=iSeq.block_size,   p2=0.5)
#sSeq.pixelsampling_y[iSeq.block_size:] = double_uniform(sSeq.noface_min_sampling, sSeq.min_sampling, sSeq.max_sampling, sSeq.noface_max_sampling , size=iSeq.block_size,   p2=0.5)

sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
#sSeq.subimage_first_column =" sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
sSeq.add_noise_L0 = True
sSeq.convert_format = robject_convert_format
sSeq.background_type = None

#random translation for th w coordinate
#sSeq.translation = 20 # 25, Should be 20!!! Warning: 25
sSeq.translations_x = numpy.zeros(sSeq.num_images)
sSeq.translations_y = numpy.zeros(sSeq.num_images)
#Centered Face
##sSeq.translations_x[0:iSeq.block_size] = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, iSeq.block_size)
##sSeq.translations_y[0:iSeq.block_size] = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, iSeq.block_size)
###sSeq.translations_x[iSeq.block_size:] = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, iSeq.block_size)
###sSeq.translations_y[iSeq.block_size:] = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, iSeq.block_size)
###Decentered Face
##sSeq.translations_x[iSeq.block_size:] = double_uniform(sSeq.noface_trans_x_min, sSeq.trans_x_min, sSeq.trans_x_max, sSeq.noface_trans_x_max, size=iSeq.block_size, p2=0.5)
##sSeq.translations_y[iSeq.block_size:] = double_uniform(sSeq.noface_trans_y_min, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.noface_trans_y_max, size=iSeq.block_size, p2=0.5)

sSeq.trans_sampled = True
sSeq.name = "Object"
#sSeq.name = "Object. Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f, 50)"%(sSeq.trans_x_min, 
#    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, int(sSeq.min_sampling*1000), int(sSeq.max_sampling*1000))

print " sSeq.subimage_first_row =", sSeq.subimage_first_row
print "sSeq.pixelsampling_x", sSeq.pixelsampling_x
print "sSeq.translations_x", sSeq.translations_x

iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)

####################################################################
###########    SYSTEM FOR REAL_OBJECT RECOGNITION       ############
####################################################################  
ParamsRObject = SystemParameters.ParamsSystem()
ParamsRObject.name = sTrainRObject.name
ParamsRObject.network = linearNetwork4L
ParamsRObject.iTrain = iTrainRObject
ParamsRObject.sTrain = sTrainRObject
ParamsRObject.iSeenid = iSeenidRObject
ParamsRObject.sSeenid = sSeenidRObject
ParamsRObject.iNewid = iNewidRObject
ParamsRObject.sNewid = sNewidRObject
##MEGAWARNING!!!!
#ParamsRFace.iNewid = iNewidScale
#ParamsRFace.sNewid = sNewidScale
#ParamsRFace.sNewid.translations_y = ParamsRFace.sNewid.translations_y * 0.0 + 8.0

ParamsRObject.block_size = iTrainRObject.block_size
ParamsRObject.train_mode = 'clustered' # 'mixed'
ParamsRObject.analysis = None
ParamsRObject.enable_reduced_image_sizes = False
ParamsRObject.reduction_factor = 1.0
ParamsRObject.hack_image_size = 32
ParamsRObject.enable_hack_image_size = True






print "***** Project: Processing natural images with SFA,  ******"
print "***** Image Patches courtesy of Niko Wilbert ******"
print "***** Setting Training Information Parameters for RawNatural ******"
iSeq = iTrainRawNatural = SystemParameters.ParamsInput()
iSeq.name = "Natural image patches"
iSeq.data_base_dir = "/home/escalafl/Databases/cooperations/igel/patches_8x8"
iSeq.base_filename = "bochum_natural_8_5000.bin"

iSeq.samples = numpy.arange(0, 4000, dtype="int")
iSeq.ids = iSeq.samples # 1 - 4000+1
iSeq.num_images = len(iSeq.samples)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #slow parameter is the image number
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = "LoadRawData"
iSeq.block_size = 1
print "totaling %d samples"% iSeq.num_images

iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]

print "BLOCK SIZE =", iSeq.block_size 

iSeq.correct_classes = iSeq.ids * 1
#print iSeq.correct_classes
#quit()
#sfa_libs.wider_1Darray(numpy.arange(len(iSeq.faces)), iSeq.block_size)
iSeq.correct_labels = iSeq.correct_classes * 1
#print iSeq.correct_labels

#sfa_libs.wider_1Darray(iSeq.faces*2-1, iSeq.block_size)
SystemParameters.test_object_contents(iSeq)

print "******** Setting Training Data Parameters for RawNatural  ****************"
sSeq = sTrainRawNatural = SystemParameters.ParamsDataLoading()
sSeq.base_filename = iSeq.base_filename
sSeq.data_base_dir = iSeq.data_base_dir
sSeq.input_files = iSeq.input_files
sSeq.samples = iSeq.samples
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.subimage_width = 8
sSeq.subimage_height = 8
sSeq.input_dim = 64
sSeq.dtype = "uint8"
sSeq.convert_format = "binary"
sSeq.name = "Natural Patch. 8x8, input_dim = %d, num_images %d"%(sSeq.input_dim, iSeq.num_images)
iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)


print "***** Setting Training Information Parameters for RawNatural ******"
iSeq = iNewidRawNatural = SystemParameters.ParamsInput()
iSeq.name = "Natural image patches"
iSeq.data_base_dir = "/home/escalafl/Databases/cooperations/igel/patches_8x8"
iSeq.base_filename = "bochum_natural_8_5000.bin"

iSeq.samples = numpy.arange(4000, 5000, dtype="int")
iSeq.ids = iSeq.samples # 1 - 4000+1
iSeq.num_images = len(iSeq.samples)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #slow parameter is the image number
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = "LoadRawData"
iSeq.block_size = 1
print "totaling %d samples"% iSeq.num_images

iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]

print "BLOCK SIZE =", iSeq.block_size 

iSeq.correct_classes = iSeq.ids * 1
#print iSeq.correct_classes
#quit()
#sfa_libs.wider_1Darray(numpy.arange(len(iSeq.faces)), iSeq.block_size)
iSeq.correct_labels = iSeq.correct_classes * 1
#print iSeq.correct_labels

#sfa_libs.wider_1Darray(iSeq.faces*2-1, iSeq.block_size)
SystemParameters.test_object_contents(iSeq)

print "******** Setting Training Data Parameters for Natural  ****************"
sSeq = sNewidRawNatural = SystemParameters.ParamsDataLoading()
sSeq.base_filename = iSeq.base_filename
sSeq.data_base_dir = iSeq.data_base_dir
sSeq.input_files = iSeq.input_files
sSeq.samples = iSeq.samples
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.subimage_width = 8
sSeq.subimage_height = 8
sSeq.input_dim = 64
sSeq.dtype = "uint8"
sSeq.convert_format = "binary"
sSeq.name = "Natural Patch. 8x8, input_dim = %d, num_images %d"%(sSeq.input_dim, iSeq.num_images)
iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)

####################################################################
###########    SYSTEM FOR RAW NATURAL IMAGE PATCHES      ###########
####################################################################  
ParamsRawNatural = SystemParameters.ParamsSystem()
ParamsRawNatural.name = sTrainRawNatural.name
ParamsRawNatural.network = None
ParamsRawNatural.iTrain = iTrainRawNatural
ParamsRawNatural.sTrain = sTrainRawNatural
ParamsRawNatural.iSeenid = iTrainRawNatural
ParamsRawNatural.sSeenid = sTrainRawNatural
ParamsRawNatural.iNewid = iNewidRawNatural
ParamsRawNatural.sNewid = sNewidRawNatural
ParamsRawNatural.block_size = iTrainRawNatural.block_size
ParamsRawNatural.train_mode = 'serial' # 'mixed'
ParamsRawNatural.analysis = False
ParamsRawNatural.enable_reduced_image_sizes = False
ParamsRawNatural.reduction_factor = -1
ParamsRawNatural.hack_image_size = -1
ParamsRawNatural.enable_hack_image_size = False





print "***** Project: Integration of RBM and SFA,  ******"
rbm_sfa_iteration = 19999 # 99 or 4999, now also 9999. 14999, 19999
rbm_sfa_numHid = 64 #64 or 128
rbm_sfa_data_base_dir = "/home/escalafl/Databases/cooperations/igel/rbm_%d"%rbm_sfa_numHid # 64 or 128

print "***** Setting Training Information Parameters for Natural ******"
iSeq = iTrainNatural = SystemParameters.ParamsInput()
# (0.55+1.1)/2 = 0.825, 0.55/2 = 0.275, 0.55/4 = 0.1375, .825 + .1375 = .9625, .825 - .55/4 = .6875
iSeq.name = "Natural images RBM"

iSeq.data_base_dir = rbm_sfa_data_base_dir
iSeq.iteration = rbm_sfa_iteration
iSeq.base_filename = "data_bin_%d.bin"%(iSeq.iteration+1)

(iSeq.magic_num, iteration, iSeq.numSamples, iSeq.numHid, iSeq.sampleSpan) = imageLoader.read_binary_header(iSeq.data_base_dir, iSeq.base_filename)
if iteration != iSeq.iteration:
    er = "wrong iteration number in file, was %d, should be %d"%(iteration, iSeq.iteration)
    raise Exception(er)

if iSeq.numHid != rbm_sfa_numHid:
    er = "wrong number of output Neurons %d, 64 were assumed"%iSeq.numHid
    raise Exception(er)

if iSeq.numSamples != 5000:
    er = "wrong number of Samples %d, 5000 were assumed"%iSeq.numSamples
    raise Exception(er)

iSeq.numSamples = 4000
iSeq.samples = numpy.arange(0, iSeq.numSamples, dtype="int")
iSeq.ids = numpy.arange(1,iSeq.numSamples+1) # 1 - 4000+1
iSeq.num_images = len(iSeq.samples)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #slow parameter is the image number
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = "LoadBinaryData00"
iSeq.block_size = 1
print "totaling %d samples"% iSeq.num_images

iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]

#iSeq.num_images / len (iSeq.faces)
print "BLOCK SIZE =", iSeq.block_size 

iSeq.correct_classes = iSeq.ids * 1
#print iSeq.correct_classes
#quit()
#sfa_libs.wider_1Darray(numpy.arange(len(iSeq.faces)), iSeq.block_size)
iSeq.correct_labels = iSeq.correct_classes * 1
#print iSeq.correct_labels

#sfa_libs.wider_1Darray(iSeq.faces*2-1, iSeq.block_size)
SystemParameters.test_object_contents(iSeq)

print "******** Setting Training Data Parameters for Natural  ****************"
sSeq = sTrainNatural = SystemParameters.ParamsDataLoading()
sSeq.base_filename = iSeq.base_filename
sSeq.data_base_dir = iSeq.data_base_dir
sSeq.input_files = iSeq.input_files
sSeq.samples = iSeq.samples
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.subimage_width = rbm_sfa_numHid / 8
sSeq.subimage_height = rbm_sfa_numHid / sSeq.subimage_width
sSeq.convert_format = "binary"
sSeq.name = "RBM Natural. 8x8 (exp 64=%d), iter %d, num_images %d"%(iSeq.numHid, iSeq.iteration, iSeq.num_images)
iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)

#NOTE: There is no Seenid training data because we do not have labels for the classifier

print "***** Setting Newid Information Parameters for Natural ******"
iSeq = iNewidNatural = SystemParameters.ParamsInput()
iSeq.name = "Natural images"

iSeq.data_base_dir = rbm_sfa_data_base_dir
#iSeq.magic_num = 666
iSeq.iteration = rbm_sfa_iteration #0
iSeq.base_filename = "data_bin_%d.bin"%(iSeq.iteration+1)

(iSeq.magic_num, iteration, iSeq.numSamples, iSeq.numHid, iSeq.sampleSpan) = imageLoader.read_binary_header(iSeq.data_base_dir, iSeq.base_filename)
if iteration != iSeq.iteration:
    er = "wrong iteration number in file, was %d, should be %d"%(iteration, iSeq.iteration)
    raise Exception(er)

if iSeq.numHid != rbm_sfa_numHid:
    er = "wrong number of output Neurons %d, %d were assumed"%(iSeq.numHid,rbm_sfa_numHid)
    raise Exception(er)

if iSeq.numSamples != 5000:
    er = "wrong number of Samples %d, 5000 were assumed"%iSeq.numSamples
    raise Exception(er)

iSeq.samples = numpy.arange(4000, 5000, dtype="int")
iSeq.ids = iSeq.samples+1 # 1 - 4000+1
iSeq.num_images = len(iSeq.samples)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #slow parameter is the image number
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = "LoadBinaryData00"
iSeq.block_size = 1
print "totaling %d images"% iSeq.num_images

iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]

#iSeq.num_images / len (iSeq.faces)
print "BLOCK SIZE =", iSeq.block_size 

iSeq.correct_classes = iSeq.ids * 1
#print iSeq.correct_classes
#quit()
#sfa_libs.wider_1Darray(numpy.arange(len(iSeq.faces)), iSeq.block_size)
iSeq.correct_labels = iSeq.correct_classes * 1
#print iSeq.correct_labels

#sfa_libs.wider_1Darray(iSeq.faces*2-1, iSeq.block_size)
SystemParameters.test_object_contents(iSeq)

print "******** Setting Newid Data Parameters for Natural ****************"
sSeq = sNewidNatural = SystemParameters.ParamsDataLoading()
sSeq.base_filename = iSeq.base_filename
sSeq.data_base_dir = iSeq.data_base_dir
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.samples = iSeq.samples
sSeq.subimage_width = rbm_sfa_numHid / 8
sSeq.subimage_height = rbm_sfa_numHid / sSeq.subimage_width
sSeq.convert_format = "binary"
sSeq.name = "RBM Natural. 8x8 (exp 100=%d), iter %d, num_images %d"%(iSeq.numHid, iSeq.iteration, iSeq.num_images)
iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)
#quit()



####################################################################
###########     SYSTEM FOR RBM NATURAL ANALYSIS         ############
####################################################################  
ParamsNatural = SystemParameters.ParamsSystem()
ParamsNatural.name = sTrainNatural.name
ParamsNatural.network = None
ParamsNatural.iTrain = iTrainNatural
ParamsNatural.sTrain = sTrainNatural
ParamsNatural.iSeenid = copy.deepcopy(iTrainNatural)
ParamsNatural.sSeenid = copy.deepcopy(sTrainNatural)
ParamsNatural.iNewid = iNewidNatural
ParamsNatural.sNewid = sNewidNatural
#ParamsNatural.iSeenid = iSeenidNatural
#ParamsNatural.sSeenid = sSeenidNatural
#ParamsNatural.iNewid = iNewidNatural
#ParamsNatural.sNewid = sNewidNatural
ParamsNatural.block_size = iTrainNatural.block_size
ParamsNatural.train_mode = 'serial' # 'mixed'
ParamsNatural.analysis = False
ParamsNatural.enable_reduced_image_sizes = False
ParamsNatural.reduction_factor = -1
ParamsNatural.hack_image_size = -1
ParamsNatural.enable_hack_image_size = False









#FaceDiscrimination
#TODO: Explain this, enlarge largest face, 
print "***** Setting Training Information Parameters for RFaceCentering******"
iSeq = iTrainRFaceCentering= SystemParameters.ParamsInput()
# (0.55+1.1)/2 = 0.825, 0.55/2 = 0.275, 0.55/4 = 0.1375, .825 + .1375 = .9625, .825 - .55/4 = .6875
iSeq.name = "Real FACE DISCRIMINATION (Centered / Decentered)"

iSeq.data_base_dir = alldbnormalized_base_dir
alldbnormalized_available_images = numpy.arange(0,55000)
numpy.random.shuffle(alldbnormalized_available_images)
alldb_noface_available_images = numpy.arange(0,12000)
numpy.random.shuffle(alldb_noface_available_images)

iSeq.ids = alldbnormalized_available_images[0:6000] #30000, numpy.arange(0,6000) # 6000
iSeq.faces = numpy.arange(0,10) # 0=centered normalized face, 1=not centered normalized face
block_sizeT = len(iSeq.ids) / len(iSeq.faces)

#iSeq.scales = numpy.linspace(pipeline_fd['smin0'], pipeline_fd['smax0'], 50) # (-50, 50, 2)
if len(iSeq.ids) % len(iSeq.faces) != 0:
    ex="Here the number of scales must be a divisor of the number of identities"
    raise Exception(ex)

iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is whether there is the amount of face centering
iSeq.step = 1
iSeq.offset = 0
repetition_factorT = 2 # WARNING 2, 8
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=5, image_postfix=".jpg")
iSeq.input_files = iSeq.input_files * repetition_factorT  #  4 was overfitting non linear sfa slightly
#To avoid grouping similar images next to one other or in the same block
numpy.random.shuffle(iSeq.input_files)  

iSeq.data2_base_dir = alldb_noface_base_dir
iSeq.ids2 = alldb_noface_available_images[0: block_sizeT * repetition_factorT]
iSeq.input_files2 = imageLoader.create_image_filenames3(iSeq.data2_base_dir, "image", iSeq.slow_signal, iSeq.ids2, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=5, image_postfix=".jpg")
iSeq.input_files2 = iSeq.input_files2 
numpy.random.shuffle(iSeq.input_files2)

iSeq.input_files = iSeq.input_files[0:-block_sizeT* repetition_factorT] + iSeq.input_files2

iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.faces)
print "BLOCK SIZE =", iSeq.block_size 

iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.faces)), iSeq.block_size)
iSeq.correct_labels = iSeq.correct_classes / (len(iSeq.faces)-1)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Training Data Parameters for Real Face Centering****************"
sSeq = sTrainRFaceCentering= SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 135
sSeq.subimage_height = 135

sSeq.trans_x_max = pipeline_fd['dx0'] * 1.0
sSeq.trans_y_max = pipeline_fd['dy0'] * 1.0 * 0.998
sSeq.min_sampling = pipeline_fd['smin0'] - 0.1 #WARNING!!!
sSeq.max_sampling = pipeline_fd['smax0']
sSeq.avg_sampling = (sSeq.min_sampling + sSeq.max_sampling)/2


sSeq.pixelsampling_x = numpy.zeros(sSeq.num_images)
sSeq.pixelsampling_y = numpy.zeros(sSeq.num_images) 
sSeq.translations_x = numpy.zeros(sSeq.num_images)
sSeq.translations_y = numpy.zeros(sSeq.num_images)


num_blocks = sSeq.num_images/sSeq.block_size
for block_nr in range(num_blocks):
    #For exterior box
    fraction = ((block_nr+1.0) / (num_blocks-1)) ** 0.333
    if fraction > 1:
        fraction = 1
    x_max = sSeq.trans_x_max * fraction
    y_max = sSeq.trans_y_max * fraction
    samp_max = sSeq.avg_sampling + (sSeq.max_sampling-sSeq.avg_sampling) * fraction
    samp_min = sSeq.avg_sampling + (sSeq.min_sampling-sSeq.avg_sampling) * fraction

    box_ext = [(-x_max, x_max), (-y_max, y_max), (samp_min, samp_max), (samp_min, samp_max)] 

    if block_nr >= 0:
        #For interior boxiSeq.ids = alldbnormalized_available_images[30000:45000]       
        if block_nr < num_blocks-1:
            eff_block_nr = block_nr
        else:
            eff_block_nr = block_nr-1
        fraction2 = (eff_block_nr / (num_blocks-1)) ** 0.333
        if fraction2 > 1:
            fraction2 = 1
        x_max = sSeq.trans_x_max * fraction2
        y_max = sSeq.trans_y_max * fraction2
        samp_max = sSeq.avg_sampling + (sSeq.max_sampling-sSeq.avg_sampling) * fraction2
        samp_min = sSeq.avg_sampling + (sSeq.min_sampling-sSeq.avg_sampling) * fraction2
        box_in = [(-x_max, x_max), (-y_max, y_max), (samp_min, samp_max), (samp_min, samp_max)] 
    
    samples = sub_box_sampling(box_in, box_ext, sSeq.block_size)
    sSeq.translations_x[block_nr*sSeq.block_size: (block_nr+1)*sSeq.block_size] = samples[:,0]
    sSeq.translations_y[block_nr*sSeq.block_size: (block_nr+1)*sSeq.block_size] = samples[:,1]
    sSeq.pixelsampling_x[block_nr*sSeq.block_size: (block_nr+1)*sSeq.block_size] = samples[:,2]
    sSeq.pixelsampling_y[block_nr*sSeq.block_size: (block_nr+1)*sSeq.block_size] = samples[:,3]
            
sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0

sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
sSeq.trans_sampled = True

sSeq.name = "Face Centering. Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f, 50)"%(-sSeq.trans_x_max, 
    sSeq.trans_x_max, -sSeq.trans_y_max, sSeq.trans_y_max, int(sSeq.min_sampling*1000), int(sSeq.max_sampling*1000))
print sSeq.name
iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)



print "***** Setting SeenId Information Parameters for RFaceCentering******"
iSeq = iSeenidRFaceCentering= SystemParameters.ParamsInput()
# (0.55+1.1)/2 = 0.825, 0.55/2 = 0.275, 0.55/4 = 0.1375, .825 + .1375 = .9625, .825 - .55/4 = .6875
iSeq.name = "Real FACE DISCRIMINATION (Centered / Decentered)"

iSeq.data_base_dir = alldbnormalized_base_dir
iSeq.ids = alldbnormalized_available_images[30000:45000] # 30000-45000 numpy.arange(6000,8000) # 6000-8000
iSeq.faces = numpy.arange(0,10) # 0=centered normalized face, 1=not centered normalized face
block_sizeS = len(iSeq.ids) / len(iSeq.faces)

#iSeq.scales = numpy.linspace(pipeline_fd['smin0'], pipeline_fd['smax0'], 50) # (-50, 50, 2)
if len(iSeq.ids) % len(iSeq.faces) != 0:
    ex="Here the number of scales must be a divisor of the number of identities"
    raise Exception(ex)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is whether there is the amount of face centering
iSeq.step = 1
iSeq.offset = 0
repetition_factorS = 1 # 2 was 4
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=5, image_postfix=".jpg")
iSeq.input_files = iSeq.input_files * repetition_factorS #  4 was overfitting non linear sfa slightly
#To avoid grouping similar images next to one other or in the same block
numpy.random.shuffle(iSeq.input_files)  


iSeq.data2_base_dir = alldb_noface_base_dir
iSeq.ids2 = alldb_noface_available_images[block_sizeT* repetition_factorT: block_sizeT*repetition_factorT+block_sizeS*repetition_factorS]

iSeq.input_files2 = imageLoader.create_image_filenames3(iSeq.data2_base_dir, "image", iSeq.slow_signal, iSeq.ids2, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=5, image_postfix=".jpg")
iSeq.input_files2 = iSeq.input_files2 
numpy.random.shuffle(iSeq.input_files2)

iSeq.input_files = iSeq.input_files[0:-block_sizeS * repetition_factorS] + iSeq.input_files2


iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.faces)
print "BLOCK SIZE =", iSeq.block_size 

iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.faces)), iSeq.block_size)
iSeq.correct_labels = iSeq.correct_classes / (len(iSeq.faces)-1)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Seenid Data Parameters for Real Face Centering****************"
sSeq = sSeenidRFaceCentering= SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 135
sSeq.subimage_height = 135

sSeq.trans_x_max = pipeline_fd['dx0'] * 1.0
sSeq.trans_y_max = pipeline_fd['dy0'] * 1.0 * 0.998
sSeq.min_sampling = pipeline_fd['smin0']- 0.1 #WARNING!!!
sSeq.max_sampling = pipeline_fd['smax0']
sSeq.avg_sampling = (sSeq.min_sampling + sSeq.max_sampling)/2


sSeq.pixelsampling_x = numpy.zeros(sSeq.num_images)
sSeq.pixelsampling_y = numpy.zeros(sSeq.num_images) 
sSeq.translations_x = numpy.zeros(sSeq.num_images)
sSeq.translations_y = numpy.zeros(sSeq.num_images)

num_blocks = sSeq.num_images/sSeq.block_size
for block_nr in range(num_blocks):
    #For exterior box
    fraction = ((block_nr+1.0) / (num_blocks-1)) ** 0.333
    if fraction > 1:
        fraction = 1
    x_max = sSeq.trans_x_max * fraction
    y_max = sSeq.trans_y_max * fraction
    samp_max = sSeq.avg_sampling + (sSeq.max_sampling-sSeq.avg_sampling) * fraction
    samp_min = sSeq.avg_sampling + (sSeq.min_sampling-sSeq.avg_sampling) * fraction

    box_ext = [(-x_max, x_max), (-y_max, y_max), (samp_min, samp_max), (samp_min, samp_max)] 

    if block_nr >= 0:
        #For interior boxiSeq.ids = alldbnormalized_available_images[30000:45000]       
        if block_nr < num_blocks-1:
            eff_block_nr = block_nr
        else:
            eff_block_nr = block_nr-1
        fraction2 = (eff_block_nr / (num_blocks-1)) ** 0.333
        if fraction2 > 1:
            fraction2 = 1
        x_max = sSeq.trans_x_max * fraction2
        y_max = sSeq.trans_y_max * fraction2
        samp_max = sSeq.avg_sampling + (sSeq.max_sampling-sSeq.avg_sampling) * fraction2
        samp_min = sSeq.avg_sampling + (sSeq.min_sampling-sSeq.avg_sampling) * fraction2
        box_in = [(-x_max, x_max), (-y_max, y_max), (samp_min, samp_max), (samp_min, samp_max)] 
           
    samples = sub_box_sampling(box_in, box_ext, sSeq.block_size)
    sSeq.translations_x[block_nr*sSeq.block_size: (block_nr+1)*sSeq.block_size] = samples[:,0]
    sSeq.translations_y[block_nr*sSeq.block_size: (block_nr+1)*sSeq.block_size] = samples[:,1]
    sSeq.pixelsampling_x[block_nr*sSeq.block_size: (block_nr+1)*sSeq.block_size] = samples[:,2]
    sSeq.pixelsampling_y[block_nr*sSeq.block_size: (block_nr+1)*sSeq.block_size] = samples[:,3]
            
sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0

sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
sSeq.trans_sampled = True

sSeq.name = "Face Centering. Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f, 50)"%(-sSeq.trans_x_max, 
    sSeq.trans_x_max, -sSeq.trans_y_max, sSeq.trans_y_max, int(sSeq.min_sampling*1000), int(sSeq.max_sampling*1000))
print sSeq.name
iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)


print "***** Setting NewId Information Parameters for RFaceCentering******"
iSeq = iNewidRFaceCentering= SystemParameters.ParamsInput()
# (0.55+1.1)/2 = 0.825, 0.55/2 = 0.275, 0.55/4 = 0.1375, .825 + .1375 = .9625, .825 - .55/4 = .6875
iSeq.name = "Real FACE DISCRIMINATION (Centered / Decentered)"

iSeq.data_base_dir = alldbnormalized_base_dir
iSeq.ids = alldbnormalized_available_images[45000:46000] #45000:55000 numpy.arange(8000,10000) # 6000-8000
iSeq.faces = numpy.arange(0,10) # 0=centered normalized face, 1=not centered normalized face
block_sizeN = len(iSeq.ids) / len(iSeq.faces)

#iSeq.scales = numpy.linspace(pipeline_fd['smin0'], pipeline_fd['smax0'], 50) # (-50, 50, 2)
if len(iSeq.ids) % len(iSeq.faces) != 0:
    ex="Here the number of scales must be a divisor of the number of identities"
    raise Exception(ex)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is whether there is the amount of face centering
iSeq.step = 1
iSeq.offset = 0
repetition_factorN = 2 # was 4
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=5, image_postfix=".jpg")
iSeq.input_files = iSeq.input_files * repetition_factorN#  4 was overfitting non linear sfa slightly
#To avoid grouping similar images next to one other or in the same block
numpy.random.shuffle(iSeq.input_files)  

iSeq.data2_base_dir = alldb_noface_base_dir
iSeq.ids2 = alldb_noface_available_images[block_sizeT*repetition_factorT+block_sizeS*repetition_factorS: block_sizeT*repetition_factorT+block_sizeS*repetition_factorS+block_sizeN*repetition_factorN]

iSeq.input_files2 = imageLoader.create_image_filenames3(iSeq.data2_base_dir, "image", iSeq.slow_signal, iSeq.ids2, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=5, image_postfix=".jpg")
iSeq.input_files2 = iSeq.input_files2
numpy.random.shuffle(iSeq.input_files2)

iSeq.input_files = iSeq.input_files[0:-block_sizeN * repetition_factorN] + iSeq.input_files2


iSeq.num_images = len(iSeq.input_files)
#iSeq.params = [ids, expressions, morphs, poses, lightings]
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                  iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.faces)
print "BLOCK SIZE =", iSeq.block_size 

iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.faces)), iSeq.block_size)
iSeq.correct_labels = iSeq.correct_classes / (len(iSeq.faces)-1)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Seenid Data Parameters for Real Face Centering****************"
sSeq = sNewidRFaceCentering= SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 135
sSeq.subimage_height = 135

sSeq.trans_x_max = pipeline_fd['dx0'] * 1.0
sSeq.trans_y_max = pipeline_fd['dy0'] * 1.0 * 0.998
sSeq.min_sampling = pipeline_fd['smin0']- 0.1 #WARNING!!!
sSeq.max_sampling = pipeline_fd['smax0']
sSeq.avg_sampling = (sSeq.min_sampling + sSeq.max_sampling)/2


sSeq.pixelsampling_x = numpy.zeros(sSeq.num_images)
sSeq.pixelsampling_y = numpy.zeros(sSeq.num_images) 
sSeq.translations_x = numpy.zeros(sSeq.num_images)
sSeq.translations_y = numpy.zeros(sSeq.num_images)

num_blocks = sSeq.num_images/sSeq.block_size
for block_nr in range(num_blocks):
    #For exterior box
    fraction = ((block_nr+1.0) / (num_blocks-1)) ** 0.333
    if fraction > 1:
        fraction = 1
    x_max = sSeq.trans_x_max * fraction
    y_max = sSeq.trans_y_max * fraction
    samp_max = sSeq.avg_sampling + (sSeq.max_sampling-sSeq.avg_sampling) * fraction
    samp_min = sSeq.avg_sampling + (sSeq.min_sampling-sSeq.avg_sampling) * fraction

    box_ext = [(-x_max, x_max), (-y_max, y_max), (samp_min, samp_max), (samp_min, samp_max)] 

    if block_nr >= 0:
        #For interior boxiSeq.ids = alldbnormalized_available_images[30000:45000]       
        if block_nr < num_blocks-1:
            eff_block_nr = block_nr
        else:
            eff_block_nr = block_nr-1
        fraction2 = (eff_block_nr / (num_blocks-1)) ** 0.333
        if fraction2 > 1:
            fraction2 = 1
        x_max = sSeq.trans_x_max * fraction2
        y_max = sSeq.trans_y_max * fraction2
        samp_max = sSeq.avg_sampling + (sSeq.max_sampling-sSeq.avg_sampling) * fraction2
        samp_min = sSeq.avg_sampling + (sSeq.min_sampling-sSeq.avg_sampling) * fraction2
        box_in = [(-x_max, x_max), (-y_max, y_max), (samp_min, samp_max), (samp_min, samp_max)] 
        
    samples = sub_box_sampling(box_in, box_ext, sSeq.block_size)
    sSeq.translations_x[block_nr*sSeq.block_size: (block_nr+1)*sSeq.block_size] = samples[:,0]
    sSeq.translations_y[block_nr*sSeq.block_size: (block_nr+1)*sSeq.block_size] = samples[:,1]
    sSeq.pixelsampling_x[block_nr*sSeq.block_size: (block_nr+1)*sSeq.block_size] = samples[:,2]
    sSeq.pixelsampling_y[block_nr*sSeq.block_size: (block_nr+1)*sSeq.block_size] = samples[:,3]
            
sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0

sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
sSeq.trans_sampled = True

sSeq.name = "Face Centering. Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f, 50)"%(-sSeq.trans_x_max, 
    sSeq.trans_x_max, -sSeq.trans_y_max, sSeq.trans_y_max, int(sSeq.min_sampling*1000), int(sSeq.max_sampling*1000))
print sSeq.name
iSeq.name = sSeq.name
SystemParameters.test_object_contents(sSeq)


####################################################################
###########    SYSTEM FOR REAL_FACE CLASSIFICATION      ############
####################################################################  
ParamsRFaceCentering = SystemParameters.ParamsSystem()
ParamsRFaceCentering.name = sTrainRFaceCentering.name
ParamsRFaceCentering.network = linearNetwork4L
ParamsRFaceCentering.iTrain = iTrainRFaceCentering
ParamsRFaceCentering.sTrain = sTrainRFaceCentering
ParamsRFaceCentering.iSeenid = iSeenidRFaceCentering
ParamsRFaceCentering.sSeenid = sSeenidRFaceCentering
ParamsRFaceCentering.iNewid = iNewidRFaceCentering
ParamsRFaceCentering.sNewid = sNewidRFaceCentering
##MEGAWARNING!!!!
#ParamsRFace.iNewid = iNewidScale
#ParamsRFace.sNewid = sNewidScale
#ParamsRFace.sNewid.translations_y = ParamsRFace.sNewid.translations_y * 0.0 + 8.0

ParamsRFaceCentering.block_size = iTrainRFaceCentering.block_size
ParamsRFaceCentering.train_mode = 'clustered' #clustered improves final performance! mixed
# 'mixed'!!! mse 0.31 clustered @ 6000 samples LSFA 11L, mse 0.30 mixed
# uexp08 => 0.019, 0.0147 @ 30 Signals (clustered)
# uexp08 => 0.063  @ 30 Signals (mixed, 5 levels)
# uexp08 => 0.034 @ 30 Signals (mixed, 10 levels)
# uexp08 => 0.027 @ 30 signals(mixed, 10 levels, pca_sfa_expo) pca_expo

ParamsRFaceCentering.analysis = None
ParamsRFaceCentering.enable_reduced_image_sizes = True
ParamsRFaceCentering.reduction_factor = 8.0 # WARNING 2.0, 4.0, 8.0
ParamsRFaceCentering.hack_image_size = 16 # WARNING 64, 32, 16
ParamsRFaceCentering.enable_hack_image_size = True

#sys.path.append("/home/escalafl/workspace/hiphi/src/hiphi/utils")







# EYE_L_X, EYE_L_X, EYE_L_X, EYE_L_X, EYE_L_X, EYE_L_X, EYE_L_X, EYE_L_X, EYE_L_X, EYE_L_X, EYE_L_X, EYE_L_X, EYE_L_X, EYE_L_X, EYE_L_X, EYE_L_X, EYE_L_X, EYE_L_X
#This network actually supports images in the closed intervals: [-eye_dx, eye_dx], [-eye_dy, eye_dy], [eye_smin0, eye_smax0]
eye_dx = 10
eye_dy = 10
eye_smax0 = 0.825 + 0.15
eye_smin0 = 0.825 - 0.15

print "***** Setting Training Information Parameters for Real Eye Translation X ******"
iSeq = iTrainREyeTransX = SystemParameters.ParamsInput()
iSeq.data_base_dir = frgc_eyeL_normalized_base_dir
iSeq.ids = numpy.arange(0,6000) # 8000, 7965

iSeq.trans = numpy.arange(-1 * eye_dx, eye_dx, 1)
if len(iSeq.ids) % len(iSeq.trans) != 0:
    ex="Here the number of translations must be a divisor of the number of identities"
    raise Exception(ex)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=5, image_postfix=".jpg")
iSeq.input_files = iSeq.input_files * 8 # warning!!! 8
#To avoid grouping similar images next to one other
numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.trans)
iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.trans)), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Training Data Parameters for Real Eye TransX  ****************"
sSeq = sTrainREyeTransX = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 64
sSeq.subimage_height = 64 

sSeq.trans_x_max = eye_dx
sSeq.trans_x_min = -eye_dx
sSeq.trans_y_max = eye_dy
sSeq.trans_y_min = -eye_dy
sSeq.min_sampling = eye_smin0
sSeq.max_sampling = eye_smax0

sSeq.pixelsampling_x = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
sSeq.pixelsampling_y = sSeq.pixelsampling_x * 1.0
sSeq.subimage_first_row =  sSeq.image_height/2.0-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2.0-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
sSeq.translations_x = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
sSeq.translations_y = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, sSeq.num_images)

sSeq.trans_sampled = True
sSeq.name = "REyeTans X Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f)"%(sSeq.trans_x_min, 
    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.min_sampling*100, sSeq.max_sampling*100)
iSeq.name = sSeq.name
print sSeq.name
SystemParameters.test_object_contents(sSeq)


print "***** Setting Seenid Information Parameters for Real Eye Translation X ******"
iSeq = iSeenidREyeTransX = SystemParameters.ParamsInput()
iSeq.data_base_dir = frgc_eyeL_normalized_base_dir
iSeq.ids = numpy.arange(6000,8000) # 8000, 7965

iSeq.trans = numpy.arange(-1 * eye_dx, eye_dx, 1)
if len(iSeq.ids) % len(iSeq.trans) != 0:
    ex="Here the number of translations must be a divisor of the number of identities"
    raise Exception(ex)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=5, image_postfix=".jpg")
iSeq.input_files = iSeq.input_files * 4 # warning!!! 4
#To avoid grouping similar images next to one other
numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.trans)
iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.trans)), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Seenid Data Parameters for Real Eye TransX  ****************"
sSeq = sSeenidREyeTransX = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 64
sSeq.subimage_height = 64 

sSeq.trans_x_max = eye_dx
sSeq.trans_x_min = -eye_dx
sSeq.trans_y_max = eye_dy
sSeq.trans_y_min = -eye_dy
sSeq.min_sampling = eye_smin0
sSeq.max_sampling = eye_smax0

sSeq.pixelsampling_x = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
sSeq.pixelsampling_y = sSeq.pixelsampling_x * 1.0
sSeq.subimage_first_row =  sSeq.image_height/2.0-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2.0-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
sSeq.translations_x = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
sSeq.translations_y = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, sSeq.num_images)

sSeq.trans_sampled = True
sSeq.name = "REyeTans X Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f)"%(sSeq.trans_x_min, 
    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.min_sampling*100, sSeq.max_sampling*100)
iSeq.name = sSeq.name
print sSeq.name
SystemParameters.test_object_contents(sSeq)


print "***** Setting Newid Information Parameters for Real Eye Translation X ******"
iSeq = iNewidREyeTransX = SystemParameters.ParamsInput()
iSeq.data_base_dir = frgc_eyeL_normalized_base_dir
iSeq.ids = numpy.arange(8000,10000) # 8000, 7965

iSeq.trans = numpy.arange(-1 * eye_dx, eye_dx, 1)
if len(iSeq.ids) % len(iSeq.trans) != 0:
    ex="Here the number of translations must be a divisor of the number of identities"
    raise Exception(ex)
iSeq.ages = [None]
iSeq.genders = [None]
iSeq.racetweens = [None]
iSeq.expressions = [None]
iSeq.morphs = [None]
iSeq.poses = [None]
iSeq.lightings = [None]
iSeq.slow_signal = 0 #real slow signal is the translation in the x axis, added during image loading
iSeq.step = 1
iSeq.offset = 0
iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=5, image_postfix=".jpg")
iSeq.input_files = iSeq.input_files * 4 # warning!!! 4
#To avoid grouping similar images next to one other
numpy.random.shuffle(iSeq.input_files)  

iSeq.num_images = len(iSeq.input_files)
iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, iSeq.poses, iSeq.lightings]
iSeq.block_size = iSeq.num_images / len (iSeq.trans)
iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.trans)), iSeq.block_size)
iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)

SystemParameters.test_object_contents(iSeq)

print "******** Setting Newid Data Parameters for Real Eye TransX  ****************"
sSeq = sNewidREyeTransX = SystemParameters.ParamsDataLoading()
sSeq.input_files = iSeq.input_files
sSeq.num_images = iSeq.num_images
sSeq.block_size = iSeq.block_size
sSeq.image_width = 256
sSeq.image_height = 192
sSeq.subimage_width = 64
sSeq.subimage_height = 64 

sSeq.trans_x_max = eye_dx
sSeq.trans_x_min = -eye_dx
sSeq.trans_y_max = eye_dy
sSeq.trans_y_min = -eye_dy
sSeq.min_sampling = eye_smin0
sSeq.max_sampling = eye_smax0

sSeq.pixelsampling_x = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
sSeq.pixelsampling_y = sSeq.pixelsampling_x * 1.0
sSeq.subimage_first_row =  sSeq.image_height/2.0-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
sSeq.subimage_first_column = sSeq.image_width/2.0-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
sSeq.add_noise_L0 = True
sSeq.convert_format = "L"
sSeq.background_type = None
sSeq.translations_x = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
sSeq.translations_y = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, sSeq.num_images)

sSeq.trans_sampled = True
sSeq.name = "REyeTans X Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f)"%(sSeq.trans_x_min, 
    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.min_sampling*100, sSeq.max_sampling*100)
iSeq.name = sSeq.name
print sSeq.name
SystemParameters.test_object_contents(sSeq)

#iSeenidREyeTransX = copy.deepcopy(iTrainREyeTransX)
#sSeenidREyeTransX = copy.deepcopy(sTrainREyeTransX)
#iNewidREyeTransX = copy.deepcopy(iTrainREyeTransX)
#sNewidREyeTransX = copy.deepcopy(sTrainREyeTransX)

####################################################################
#######    SYSTEM FOR REAL_EYE_TRANSLATION_X EXTRACTION     ########
####################################################################  
ParamsREyeTransX = SystemParameters.ParamsSystem()
ParamsREyeTransX.name = sTrainREyeTransX.name
ParamsREyeTransX.network = linearNetwork4L
ParamsREyeTransX.iTrain = iTrainREyeTransX
ParamsREyeTransX.sTrain = sTrainREyeTransX
ParamsREyeTransX.iSeenid = iSeenidREyeTransX
ParamsREyeTransX.sSeenid = sSeenidREyeTransX
ParamsREyeTransX.iNewid = iNewidREyeTransX
ParamsREyeTransX.sNewid = sNewidREyeTransX
ParamsREyeTransX.block_size = iTrainREyeTransX.block_size
ParamsREyeTransX.train_mode = 'mixed'
ParamsREyeTransX.analysis = None
ParamsREyeTransX.enable_reduced_image_sizes = True
ParamsREyeTransX.reduction_factor = 2.0
ParamsREyeTransX.hack_image_size = 32
ParamsREyeTransX.enable_hack_image_size = True


#REAL_EYE_TRANSLATION_Y REAL_EYE_TRANSLATION_Y REAL_EYE_TRANSLATION_Y REAL_EYE_TRANSLATION_Y REAL_EYE_TRANSLATION_Y REAL_EYE_TRANSLATION_Y
#Just exchange translations_x and translations_y to create iTrainREyeTransY
iTrainREyeTransY = copy.deepcopy(iTrainREyeTransX)
sTrainREyeTransY = copy.deepcopy(sTrainREyeTransX)
sTrainREyeTransY.name = sTrainREyeTransY.name[0:10]+"Y"+sTrainREyeTransY.name[11:]
tmp = sTrainREyeTransY.translations_x
sTrainREyeTransY.translations_x = sTrainREyeTransY.translations_y
sTrainREyeTransY.translations_y = tmp

iSeenidREyeTransY = copy.deepcopy(iSeenidREyeTransX)
sSeenidREyeTransY = copy.deepcopy(sSeenidREyeTransX)
sSeenidREyeTransY.name = sSeenidREyeTransY.name[0:10]+"Y"+sSeenidREyeTransY.name[11:]
tmp = sSeenidREyeTransY.translations_x
sSeenidREyeTransY.translations_x = sSeenidREyeTransY.translations_y
sSeenidREyeTransY.translations_y = tmp

iNewidREyeTransY = copy.deepcopy(iNewidREyeTransX)
sNewidREyeTransY = copy.deepcopy(sNewidREyeTransX)
sNewidREyeTransY.name= sSeenidREyeTransY.name[0:10]+"Y"+sSeenidREyeTransY.name[11:]
tmp = sNewidREyeTransY.translations_x
sNewidREyeTransY.translations_x = sNewidREyeTransY.translations_y
sNewidREyeTransY.translations_y = tmp

####################################################################
######    SYSTEM FOR REAL_EYE_TRANSLATION_Y EXTRACTION     #########
####################################################################  
ParamsREyeTransY = SystemParameters.ParamsSystem()
ParamsREyeTransY.name = sTrainREyeTransY.name
ParamsREyeTransY.network = linearNetwork4L
ParamsREyeTransY.iTrain = iTrainREyeTransY
ParamsREyeTransY.sTrain = sTrainREyeTransY
ParamsREyeTransY.iSeenid = iSeenidREyeTransY
ParamsREyeTransY.sSeenid = sSeenidREyeTransY
ParamsREyeTransY.iNewid = iNewidREyeTransY
ParamsREyeTransY.sNewid = sNewidREyeTransY
ParamsREyeTransY.block_size = iTrainREyeTransY.block_size
ParamsREyeTransY.train_mode = 'mixed' #mixed
ParamsREyeTransY.analysis = None
ParamsREyeTransY.enable_reduced_image_sizes = True
ParamsREyeTransY.reduction_factor = 2.0
ParamsREyeTransY.hack_image_size = 32
ParamsREyeTransY.enable_hack_image_size = True



############################################################
########## Function Defined Data Sources ###################
############################################################
def iSeqCreateRFaceCentering(num_images, alldbnormalized_available_images, alldb_noface_available_images, first_image=0, repetition_factor=1, seed=-1):
    if seed >= 0 or seed == None: #also works for 
        numpy.random.seed(seed)
    #else seed <0 then, do not change seed
    
    print "***** Setting information Parameters for RFaceCentering******"
    iSeq = SystemParameters.ParamsInput()
    # (0.55+1.1)/2 = 0.825, 0.55/2 = 0.275, 0.55/4 = 0.1375, .825 + .1375 = .9625, .825 - .55/4 = .6875
    iSeq.name = "Real FACE DISCRIMINATION (Centered / Decentered)"

    
    if num_images > len(alldbnormalized_available_images):
        err = "Number of images to be used exceeds the number of available images"
        raise Exception(err) 

    if num_images/10 * repetition_factor> len(alldb_noface_available_images):
        err = "Number of no_face images to be used exceeds the number of available images"
        raise Exception(err) 

    iSeq.data_base_dir = alldbnormalized_base_dir
   
    iSeq.ids = alldbnormalized_available_images[first_image:first_image + num_images] #30000, numpy.arange(0,6000) # 6000
    iSeq.faces = numpy.arange(0,10) # 0=centered normalized face, 1=not centered normalized face
    block_size = len(iSeq.ids) / len(iSeq.faces)
    
    #iSeq.scales = numpy.linspace(pipeline_fd['smin0'], pipeline_fd['smax0'], 50) # (-50, 50, 2)
    if len(iSeq.ids) % len(iSeq.faces) != 0:
        ex="Here the number of scales must be a divisor of the number of identities"
        raise Exception(ex)
    
    iSeq.ages = [None]
    iSeq.genders = [None]
    iSeq.racetweens = [None]
    iSeq.expressions = [None]
    iSeq.morphs = [None]
    iSeq.poses = [None]
    iSeq.lightings = [None]
    iSeq.slow_signal = 0 #real slow signal is whether there is the amount of face centering
    iSeq.step = 1
    iSeq.offset = 0
    iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                                iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                                iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=5, image_postfix=".jpg")
    iSeq.input_files = iSeq.input_files * repetition_factor  #  4 was overfitting non linear sfa slightly
    #To avoid grouping similar images next to one other or in the same block
    numpy.random.shuffle(iSeq.input_files)  
    
    #Background images are not duplicated, instead more are taken
    iSeq.data2_base_dir = alldb_noface_base_dir
    iSeq.ids2 = alldb_noface_available_images[0: block_size * repetition_factor]
    iSeq.input_files2 = imageLoader.create_image_filenames3(iSeq.data2_base_dir, "image", iSeq.slow_signal, iSeq.ids2, iSeq.ages, \
                                                iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                                iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=5, image_postfix=".jpg")
    iSeq.input_files2 = iSeq.input_files2 
    numpy.random.shuffle(iSeq.input_files2)
    
    iSeq.input_files = iSeq.input_files[0:-block_size* repetition_factor] + iSeq.input_files2
    
    iSeq.num_images = len(iSeq.input_files)
    #iSeq.params = [ids, expressions, morphs, poses, lightings]
    iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                      iSeq.morphs, iSeq.poses, iSeq.lightings]
    iSeq.block_size = iSeq.num_images / len (iSeq.faces)
    iSeq.train_mode = "clustered"
    print "BLOCK SIZE =", iSeq.block_size 
    
    iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.faces)), iSeq.block_size)
    iSeq.correct_labels = iSeq.correct_classes / (len(iSeq.faces)-1)
    
    SystemParameters.test_object_contents(iSeq)
    return iSeq


def sSeqCreateRFaceCentering(iSeq, seed=-1):
    if seed >= 0 or seed == None: #also works for 
        numpy.random.seed(seed)
    #else seed <0 then, do not change seed
    
    print "******** Setting Training Data Parameters for Real Face Centering****************"
    sSeq = SystemParameters.ParamsDataLoading()
    sSeq.input_files = iSeq.input_files
    sSeq.num_images = iSeq.num_images
    sSeq.block_size = iSeq.block_size
    sSeq.train_mode = iSeq.train_mode
    sSeq.image_width = 256
    sSeq.image_height = 192
    sSeq.subimage_width = 135
    sSeq.subimage_height = 135
    
    sSeq.trans_x_max = pipeline_fd['dx0'] * 1.0
    sSeq.trans_y_max = pipeline_fd['dy0'] * 1.0 * 0.998
    sSeq.min_sampling = pipeline_fd['smin0'] - 0.1 #WARNING!!!
    sSeq.max_sampling = pipeline_fd['smax0']
    sSeq.avg_sampling = (sSeq.min_sampling + sSeq.max_sampling)/2
    
    
    sSeq.pixelsampling_x = numpy.zeros(sSeq.num_images)
    sSeq.pixelsampling_y = numpy.zeros(sSeq.num_images) 
    sSeq.translations_x = numpy.zeros(sSeq.num_images)
    sSeq.translations_y = numpy.zeros(sSeq.num_images)
    
    
    num_blocks = sSeq.num_images/sSeq.block_size
    for block_nr in range(num_blocks):
        #For exterior box
        fraction = ((block_nr+1.0) / (num_blocks-1)) ** 0.333
        if fraction > 1:
            fraction = 1
        x_max = sSeq.trans_x_max * fraction
        y_max = sSeq.trans_y_max * fraction
        samp_max = sSeq.avg_sampling + (sSeq.max_sampling-sSeq.avg_sampling) * fraction
        samp_min = sSeq.avg_sampling + (sSeq.min_sampling-sSeq.avg_sampling) * fraction
    
        box_ext = [(-x_max, x_max), (-y_max, y_max), (samp_min, samp_max), (samp_min, samp_max)] 
    
        if block_nr >= 0:
            #For interior boxiSeq.ids = alldbnormalized_available_images[30000:45000]       
            if block_nr < num_blocks-1:
                eff_block_nr = block_nr
            else:
                eff_block_nr = block_nr-1
            fraction2 = (eff_block_nr / (num_blocks-1)) ** 0.333
            if fraction2 > 1:
                fraction2 = 1
            x_max = sSeq.trans_x_max * fraction2
            y_max = sSeq.trans_y_max * fraction2
            samp_max = sSeq.avg_sampling + (sSeq.max_sampling-sSeq.avg_sampling) * fraction2
            samp_min = sSeq.avg_sampling + (sSeq.min_sampling-sSeq.avg_sampling) * fraction2
            box_in = [(-x_max, x_max), (-y_max, y_max), (samp_min, samp_max), (samp_min, samp_max)] 
        
        samples = sub_box_sampling(box_in, box_ext, sSeq.block_size)
        sSeq.translations_x[block_nr*sSeq.block_size: (block_nr+1)*sSeq.block_size] = samples[:,0]
        sSeq.translations_y[block_nr*sSeq.block_size: (block_nr+1)*sSeq.block_size] = samples[:,1]
        sSeq.pixelsampling_x[block_nr*sSeq.block_size: (block_nr+1)*sSeq.block_size] = samples[:,2]
        sSeq.pixelsampling_y[block_nr*sSeq.block_size: (block_nr+1)*sSeq.block_size] = samples[:,3]
                
    sSeq.subimage_first_row =  sSeq.image_height/2-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
    sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
    
    sSeq.add_noise_L0 = True
    sSeq.convert_format = "L"
    sSeq.background_type = None
    sSeq.trans_sampled = True
    
    sSeq.name = "Face Centering. Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f, 50)"%(-sSeq.trans_x_max, 
        sSeq.trans_x_max, -sSeq.trans_y_max, sSeq.trans_y_max, int(sSeq.min_sampling*1000), int(sSeq.max_sampling*1000))
    print sSeq.name
    iSeq.name = sSeq.name

    
    sSeq.load_data = lambda : load_data_from_sSeq(sSeq)

    SystemParameters.test_object_contents(sSeq)


alldbnormalized_available_images = numpy.arange(0,55000)
numpy.random.shuffle(alldbnormalized_available_images)  
av_fim = alldbnormalized_available_images
alldb_noface_available_images = numpy.arange(0,12000)
numpy.random.shuffle(alldb_noface_available_images)
av_nfim =alldb_noface_available_images

iSeq = iSeqCreateRFaceCentering(3000, av_fim, av_nfim, first_image=0, repetition_factor=2, seed=-1)
sSeq = sSeqCreateRFaceCentering(iSeq, seed=-1)
print ";)"


iSeq = iTrainRFaceCentering2 = \
[[iSeqCreateRFaceCentering(3000, av_fim, av_nfim, first_image=0, repetition_factor=2, seed=-1)], \
[iSeqCreateRFaceCentering(3000, av_fim, av_nfim, first_image=3000, repetition_factor=2, seed=-1),iSeqCreateRFaceCentering(3000, av_fim, av_nfim, first_image=6000, repetition_factor=2, seed=-1)]]

sTrainRFaceCentering2 = [[sSeqCreateRFaceCentering(iSeq[0][0], seed=-1)], \
                        [sSeqCreateRFaceCentering(iSeq[1][0], seed=-1), sSeqCreateRFaceCentering(iSeq[1][1], seed=-1)]]
iSeq = iSeenidRFaceCentering2 = [[iSeqCreateRFaceCentering(1000, av_fim, av_nfim, first_image=30000, repetition_factor=2, seed=-1)]]
sSeenidRFaceCentering2 =  [[sSeqCreateRFaceCentering(iSeq[0][0], seed=-1)]]
iSeq = iNewidRFaceCentering2 = [[iSeqCreateRFaceCentering(1000, av_fim, av_nfim, first_image=45000, repetition_factor=2, seed=-1)]]
sNewidRFaceCentering2 = [[sSeqCreateRFaceCentering(iSeq[0][0], seed=-1)]]
      
ParamsRFaceCenteringFunc = SystemParameters.ParamsSystem()
ParamsRFaceCenteringFunc.name = "Function Based Data Creation for RFaceCentering"
ParamsRFaceCenteringFunc.network = linearNetwork4L #Default Network, but ignored
ParamsRFaceCenteringFunc.iTrain =iTrainRFaceCentering2
ParamsRFaceCenteringFunc.sTrain = sTrainRFaceCentering2
ParamsRFaceCenteringFunc.iSeenid = iSeenidRFaceCentering2
ParamsRFaceCenteringFunc.sSeenid = sSeenidRFaceCentering2
ParamsRFaceCenteringFunc.iNewid = iNewidRFaceCentering2
ParamsRFaceCenteringFunc.sNewid = sNewidRFaceCentering2
ParamsRFaceCenteringFunc.block_size = iTrainRFaceCentering2[0][0].block_size
ParamsRFaceCenteringFunc.train_mode = 'clustered' #clustered improves final performance! mixed
ParamsRFaceCenteringFunc.analysis = None
ParamsRFaceCenteringFunc.enable_reduced_image_sizes = True
ParamsRFaceCenteringFunc.reduction_factor = 8.0 # WARNING 2.0, 4.0, 8.0
ParamsRFaceCenteringFunc.hack_image_size = 16 # WARNING 64, 32, 16
ParamsRFaceCenteringFunc.enable_hack_image_size = True






#PIPELINE FOR FACE DETECTION:
#Orig=TX: DX0=+/- 45, DY0=+/- 20, DS0= 0.55-1.1
#TY: DX1=+/- 20, DY0=+/- 20, DS0= 0.55-1.1
#S: DX1=+/- 20, DY1=+/- 10, DS0= 0.55-1.1
#TMX: DX1=+/- 20, DY1=+/- 10, DS1= 0.775-1.05
#TMY: DX2=+/- 10, DY1=+/- 10, DS1= 0.775-1.05
#MS: DX2=+/- 10, DY2=+/- 5, DS1= 0.775-1.05
#Out About: DX2=+/- 10, DY2=+/- 5, DS2= 0.8875-1.025
#notice: for dx* and dy* intervals are open, while for smin and smax intervals are closed
#Pipeline actually supports inputs in: [-dx0, dx0-2] [-dy0, dy0-2] [smin0, smax0] 
#Remember these values are before image resizing
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#This network actually supports images in the closed intervals: [smin0, smax0] [-dy0, dy0]
#but halb-open [-dx0, dx0) 
pipeline_fd = dict(dx0 = 45, dy0 = 20, smin0 = 0.55, smax0 = 1.1,
                dx1 = 20, dy1 = 10, smin1 = 0.775, smax1 = 1.05)
#7965, 8
def iSeqCreateRTransX(num_images, alldbnormalized_available_images, first_image=0, repetition_factor=1, seed=-1):
    if seed >= 0 or seed == None: #also works for 
        numpy.random.seed(seed)

    if num_images > len(alldbnormalized_available_images):
        err = "Number of images to be used exceeds the number of available images"
        raise Exception(err) 


    print "***** Setting Information Parameters for Real Translation X ******"
    iSeq = SystemParameters.ParamsInput()
    iSeq.name = "Real Translation X: (-45, 45, 2) translation and y 40"
    iSeq.data_base_dir = alldbnormalized_base_dir
    iSeq.ids = alldbnormalized_available_images[first_image:first_image + num_images] 

    iSeq.trans = numpy.arange(-1 * pipeline_fd['dx0'], pipeline_fd['dx0'], 2) # (-50, 50, 2)
    if len(iSeq.ids) % len(iSeq.trans) != 0:
        ex="Here the number of translations must be a divisor of the number of identities"
        raise Exception(ex)
    iSeq.ages = [None]
    iSeq.genders = [None]
    iSeq.racetweens = [None]
    iSeq.expressions = [None]
    iSeq.morphs = [None]
    iSeq.poses = [None]
    iSeq.lightings = [None]
    iSeq.slow_signal = 0 #real slow signal is the translation in the x axis (correlated to identity), added during image loading
    iSeq.step = 1
    iSeq.offset = 0
    iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                                iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                                iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=5, image_postfix=".jpg")
    iSeq.input_files = iSeq.input_files * repetition_factor # warning!!! 4, 8
    #To avoid grouping similar images next to one other, even though available images already shuffled
    numpy.random.shuffle(iSeq.input_files)  
    
    iSeq.num_images = len(iSeq.input_files)
    #iSeq.params = [ids, expressions, morphs, poses, lightings]
    iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                      iSeq.morphs, iSeq.poses, iSeq.lightings]
    iSeq.block_size = iSeq.num_images / len (iSeq.trans)
    iSeq.train_mode = "mixed"
    print "BLOCK SIZE =", iSeq.block_size 
    iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.trans)), iSeq.block_size)
    iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
    
    SystemParameters.test_object_contents(iSeq)
    return iSeq

def sSeqCreateRTransX(iSeq, seed=-1):
    if seed >= 0 or seed == None: #also works for 
        numpy.random.seed(seed)
    #else seed <0 then, do not change seed
    
    print "******** Setting Training Data Parameters for Real TransX  ****************"
    sSeq = SystemParameters.ParamsDataLoading()
    sSeq.input_files = iSeq.input_files
    sSeq.num_images = iSeq.num_images
    sSeq.block_size = iSeq.block_size
    sSeq.train_mode = iSeq.train_mode
    sSeq.include_latest = iSeq.include_latest
    sSeq.image_width = 256
    sSeq.image_height = 192
    sSeq.subimage_width = 128
    sSeq.subimage_height = 128 
    
    sSeq.trans_x_max = pipeline_fd['dx0']
    sSeq.trans_x_min = -1 * pipeline_fd['dx0']
    
    #WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    sSeq.trans_y_max = pipeline_fd['dy0']
    sSeq.trans_y_min = -1 * sSeq.trans_y_max
    
    #iSeq.scales = numpy.linspace(0.5, 1.30, 16) # (-50, 50, 2)
    sSeq.min_sampling = pipeline_fd['smin0']
    sSeq.max_sampling = pipeline_fd['smax0']
    
    sSeq.pixelsampling_x = sSeq.pixelsampling_y = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
    #sSeq.subimage_pixelsampling = 2
    sSeq.subimage_first_row =  sSeq.image_height/2.0-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
    sSeq.subimage_first_column = sSeq.image_width/2.0-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
    #sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
    sSeq.add_noise_L0 = True
    sSeq.convert_format = "L"
    sSeq.background_type = None
    #random translation for th w coordinate
    #sSeq.translation = 20 # 25, Should be 20!!! Warning: 25
    #sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)                                                           
    sSeq.translations_x = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
    sSeq.translations_y = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, sSeq.num_images)
    sSeq.trans_sampled = True
    sSeq.name = "RTans X Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f)"%(sSeq.trans_x_min, 
        sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.min_sampling*100, sSeq.max_sampling*100)

    sSeq.load_data = load_data_from_sSeq
    SystemParameters.test_object_contents(sSeq)
    return sSeq
    


####################################################################
###########    SYSTEM FOR REAL_TRANSLATION_X EXTRACTION      ############
####################################################################  
av_fim = alldbnormalized_available_images = numpy.arange(0,55000)
numpy.random.shuffle(alldbnormalized_available_images)  

#iSeq = iSeqCreateRTransX(4500, av_fim, first_image=0, repetition_factor=2, seed=-1)
#sSeq = sSeqCreateRTransX(iSeq, seed=-1)
#print ";) 2"
#iSeq=None 
#sSeq =None

## CASE [[F]]
iSeq_set = iTrainRTransX2 = [[iSeqCreateRTransX(9000, av_fim, first_image=0, repetition_factor=1, seed=-1)]]
sSeq_set = sTrainRTransX2 = [[sSeqCreateRTransX(iSeq_set[0][0], seed=-1)]]
#print sSeq_set[0][0].block_size

## CASE [[F1, F2]]
#iSeq_set = iTrainRTransX2 = [[iSeqCreateRTransX(4500, av_fim, first_image=0, repetition_factor=1, seed=-1), 
#                              iSeqCreateRTransX(2250, av_fim, first_image=4500, repetition_factor=1, seed=-1)]]
#sSeq_set = sTrainRTransX2 = [[sSeqCreateRTransX(iSeq_set[0][0], seed=-1), sSeqCreateRTransX(iSeq_set[0][1], seed=-1)]]

## CASE [[F1],[F2], ...] 
#iSeq_set = iTrainRTransX2 = [[iSeqCreateRTransX(1800, av_fim, first_image=0, repetition_factor=1, seed=-1)], 
#                             [iSeqCreateRTransX(1800, av_fim, first_image=1800, repetition_factor=1, seed=-1)], 
#                             [iSeqCreateRTransX(1800, av_fim, first_image=3600, repetition_factor=1, seed=-1)], 
#                             [iSeqCreateRTransX(1800, av_fim, first_image=5400, repetition_factor=1, seed=-1)], 
#                             [iSeqCreateRTransX(1800, av_fim, first_image=7200, repetition_factor=1, seed=-1)]]
#sSeq_set = sTrainRTransX2 = [[sSeqCreateRTransX(iSeq_set[0][0], seed=-1)], 
#                             [sSeqCreateRTransX(iSeq_set[1][0], seed=-1)], 
#                             [sSeqCreateRTransX(iSeq_set[2][0], seed=-1)], 
#                             [sSeqCreateRTransX(iSeq_set[3][0], seed=-1)], 
#                             [sSeqCreateRTransX(iSeq_set[4][0], seed=-1)]]


## CASE [[F1, F2],[F1, F3], ...] 
#iSeq0 = iSeqCreateRTransX(4500, av_fim, first_image=0, repetition_factor=1, seed=-1)
#sSeq0 = sSeqCreateRTransX(iSeq0, seed=-1)
#
#iSeq_set = iTrainRTransX2 = [[copy.deepcopy(iSeq0), iSeqCreateRTransX(900, av_fim, first_image=4500, repetition_factor=1, seed=-1)], 
#                             [copy.deepcopy(iSeq0), iSeqCreateRTransX(900, av_fim, first_image=5400, repetition_factor=1, seed=-1)], 
#                             [copy.deepcopy(iSeq0), iSeqCreateRTransX(900, av_fim, first_image=6300, repetition_factor=1, seed=-1)], 
#                             [copy.deepcopy(iSeq0), iSeqCreateRTransX(900, av_fim, first_image=7200, repetition_factor=1, seed=-1)],
#                             [copy.deepcopy(iSeq0), iSeqCreateRTransX(900, av_fim, first_image=8100, repetition_factor=1, seed=-1)],                             
#                             ]
#sSeq_set = sTrainRTransX2 = [[copy.deepcopy(sSeq0), sSeqCreateRTransX(iSeq_set[0][1], seed=-1)], 
#                             [copy.deepcopy(sSeq0), sSeqCreateRTransX(iSeq_set[1][1], seed=-1)], 
#                             [copy.deepcopy(sSeq0), sSeqCreateRTransX(iSeq_set[2][1], seed=-1)],
#                             [copy.deepcopy(sSeq0), sSeqCreateRTransX(iSeq_set[3][1], seed=-1)],                              
#                             [copy.deepcopy(sSeq0), sSeqCreateRTransX(iSeq_set[4][1], seed=-1)],
#                             ]


print sSeq_set[0][0].subimage_width 


iSeq = iSeenidRTransX2 = iSeqCreateRTransX(9000, av_fim, first_image=9000, repetition_factor=1, seed=-1)
sSeenidRTransX2 = sSeqCreateRTransX(iSeq, seed=-1)
print sSeenidRTransX2.subimage_width

iSeq_set = iNewidRTransX2 = [[iSeqCreateRTransX(4500, av_fim, first_image=18000, repetition_factor=1, seed=-1)]]
sNewidRTransX2 = [[sSeqCreateRTransX(iSeq_set[0][0], seed=-1)]]
print sSeq_set[0][0].subimage_width


ParamsRTransXFunc = SystemParameters.ParamsSystem()
ParamsRTransXFunc.name = "Function Based Data Creation for RTransX"
ParamsRTransXFunc.network = linearNetwork4L #Default Network, but ignored
ParamsRTransXFunc.iTrain =iTrainRTransX2
ParamsRTransXFunc.sTrain = sTrainRTransX2

ParamsRTransXFunc.iSeenid = iSeenidRTransX2
ParamsRTransXFunc.sSeenid = sSeenidRTransX2

ParamsRTransXFunc.iNewid = iNewidRTransX2
ParamsRTransXFunc.sNewid = sNewidRTransX2

ParamsRTransXFunc.block_size = iTrainRTransX2[0][0].block_size
ParamsRTransXFunc.train_mode = 'clustered' #clustered improves final performance! mixed
ParamsRTransXFunc.analysis = None
ParamsRTransXFunc.enable_reduced_image_sizes = True
ParamsRTransXFunc.reduction_factor = 8.0 # WARNING 1.0, 2.0, 4.0, 8.0
ParamsRTransXFunc.hack_image_size = 16 # WARNING 128, 64, 32, 16
ParamsRTransXFunc.enable_hack_image_size = True






# YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYyYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
#This network actually supports images in the closed intervals: [-dx1, dx1], [smin0, smax0]
#but halb-open [-dy0, dy0)
def iSeqCreateRTransY(num_images, alldbnormalized_available_images, first_image=0, repetition_factor=1, seed=-1): 
    if seed >= 0 or seed == None: #also works for 
        numpy.random.seed(seed)

    if num_images > len(alldbnormalized_available_images):
        err = "Number of images to be used exceeds the number of available images"
        raise Exception(err) 

    print "***** Setting Training Information Parameters for Real Translation Y ******"
    iSeq = SystemParameters.ParamsInput()
    iSeq.name = "Real Translation Y: Y(-20, 20, 1) translation and dx 20"

    iSeq.data_base_dir = alldbnormalized_base_dir
    iSeq.ids = alldbnormalized_available_images[first_image:first_image + num_images] 
    
    iSeq.trans = numpy.arange(-1 * pipeline_fd['dy0'], pipeline_fd['dy0'], 1) # (-50, 50, 2)
    if len(iSeq.ids) % len(iSeq.trans) != 0:
        ex="Here the number of translations (%d) must be a divisor of the number of identities (%d)"%(len(iSeq.ids), len(iSeq.trans))
        raise Exception(ex)
        
    iSeq.ages = [None]
    iSeq.genders = [None]
    iSeq.racetweens = [None]
    iSeq.expressions = [None]
    iSeq.morphs = [None]
    iSeq.poses = [None]
    iSeq.lightings = [None]
    iSeq.slow_signal = 0 #real slow signal is the translation in the x axis, added during image loading
    iSeq.step = 1
    iSeq.offset = 0
    iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                                iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                                iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=5, image_postfix=".jpg")
    iSeq.input_files = iSeq.input_files * repetition_factor # warning!!! 4, 8
    #To avoid grouping similar images next to one other
    numpy.random.shuffle(iSeq.input_files)  
    
    iSeq.num_images = len(iSeq.input_files)
    #iSeq.params = [ids, expressions, morphs, poses, lightings]
    iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                      iSeq.morphs, iSeq.poses, iSeq.lightings]
    iSeq.block_size = iSeq.num_images / len (iSeq.trans)
    iSeq.train_mode = "mixed"    
    print "BLOCK SIZE =", iSeq.block_size 
    iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.trans)), iSeq.block_size)
    iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)   
    SystemParameters.test_object_contents(iSeq)
    return iSeq

def sSeqCreateRTransY(iSeq, seed=-1):
    if seed >= 0 or seed == None: #also works for 
        numpy.random.seed(seed)
    #else seed <0 then, do not change seed

    print "******** Setting Training Data Parameters for Real TransY  ****************"
    sSeq = SystemParameters.ParamsDataLoading()
    sSeq.input_files = iSeq.input_files
    sSeq.num_images = iSeq.num_images
    sSeq.block_size = iSeq.block_size
    sSeq.train_mode = iSeq.train_mode
    sSeq.image_width = 256
    sSeq.image_height = 192
    sSeq.subimage_width = 128
    sSeq.subimage_height = 128 
    
    sSeq.trans_x_max = pipeline_fd['dx1']
    sSeq.trans_x_min = -1 * pipeline_fd['dx1']
    
    sSeq.trans_y_max = pipeline_fd['dy0']
    sSeq.trans_y_min = -1 * sSeq.trans_y_max
    
    #iSeq.scales = numpy.linspace(0.5, 1.30, 16) # (-50, 50, 2)
    sSeq.min_sampling = pipeline_fd['smin0']
    sSeq.max_sampling = pipeline_fd['smax0']
    
    sSeq.pixelsampling_x = sSeq.pixelsampling_y = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
    #sSeq.subimage_pixelsampling = 2
    sSeq.subimage_first_row =  sSeq.image_height/2.0-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
    sSeq.subimage_first_column = sSeq.image_width/2.0-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
    #sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
    sSeq.add_noise_L0 = True
    sSeq.convert_format = "L"
    sSeq.background_type = None
    #random translation for th w coordinate
    #sSeq.translation = 20 # 25, Should be 20!!! Warning: 25
    #sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)                                                           
    sSeq.translations_x = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, sSeq.num_images)
    sSeq.translations_y = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
    
    sSeq.trans_sampled = True
    sSeq.name = "RTans Y Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f)"%(sSeq.trans_x_min, 
        sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.min_sampling*100, sSeq.max_sampling*100)
    iSeq.name = sSeq.name
    SystemParameters.test_object_contents(sSeq)
    return sSeq


av_fim = alldbnormalized_available_images = numpy.arange(0,55000)
numpy.random.shuffle(alldbnormalized_available_images)  

#iSeq = iSeqCreateRTransY(4800, av_fim, first_image=0, repetition_factor=2, seed=-1)
#sSeq = sSeqCreateRTransY(iSeq, seed=-1)
#print ";) 3"
#quit()

iSeq=None 
sSeq =None

## CASE [[F]]
iSeq_set = iTrainRTransY2 = [[iSeqCreateRTransY(10000, av_fim, first_image=0, repetition_factor=1, seed=-1)]]
sSeq_set = sTrainRTransY2 = [[sSeqCreateRTransY(iSeq_set[0][0], seed=-1)]]

## CASE [[F1, F2]]
#iSeq_set = iTrainRTransY2 = [[iSeqCreateRTransY(4500, av_fim, first_image=0, repetition_factor=1, seed=-1), 
#                              iSeqCreateRTransY(2250, av_fim, first_image=4500, repetition_factor=1, seed=-1)]]
#sSeq_set = sTrainRTransY2 = [[sSeqCreateRTransY(iSeq_set[0][0], seed=-1), sSeqCreateRTransY(iSeq_set[0][1], seed=-1)]]

## CASE [[F1],[F2], ...] 
#iSeq_set = iTrainRTransY2 = [[iSeqCreateRTransY(1800, av_fim, first_image=0, repetition_factor=1, seed=-1)], 
#                             [iSeqCreateRTransY(1800, av_fim, first_image=1800, repetition_factor=1, seed=-1)], 
#                             [iSeqCreateRTransY(1800, av_fim, first_image=3600, repetition_factor=1, seed=-1)], 
#                             [iSeqCreateRTransY(1800, av_fim, first_image=5400, repetition_factor=1, seed=-1)], 
#                             [iSeqCreateRTransY(1800, av_fim, first_image=7200, repetition_factor=1, seed=-1)]]
#sSeq_set = sTrainRTransY2 = [[sSeqCreateRTransY(iSeq_set[0][0], seed=-1)], 
#                             [sSeqCreateRTransY(iSeq_set[1][0], seed=-1)], 
#                             [sSeqCreateRTransY(iSeq_set[2][0], seed=-1)], 
#                             [sSeqCreateRTransY(iSeq_set[3][0], seed=-1)], 
#                             [sSeqCreateRTransY(iSeq_set[4][0], seed=-1)]]


## CASE [[F1, F2],[F1, F3], ...] 
#iSeq0 = iSeqCreateRTransY(4500, av_fim, first_image=0, repetition_factor=1, seed=-1)
#sSeq0 = sSeqCreateRTransY(iSeq0, seed=-1)
#
#iSeq_set = iTrainRTransY2 = [[copy.deepcopy(iSeq0), iSeqCreateRTransY(900, av_fim, first_image=4500, repetition_factor=1, seed=-1)], 
#                             [copy.deepcopy(iSeq0), iSeqCreateRTransY(900, av_fim, first_image=5400, repetition_factor=1, seed=-1)], 
#                             [copy.deepcopy(iSeq0), iSeqCreateRTransY(900, av_fim, first_image=6300, repetition_factor=1, seed=-1)], 
#                             [copy.deepcopy(iSeq0), iSeqCreateRTransY(900, av_fim, first_image=7200, repetition_factor=1, seed=-1)],
#                             [copy.deepcopy(iSeq0), iSeqCreateRTransY(900, av_fim, first_image=8100, repetition_factor=1, seed=-1)],                             
#                             ]
#sSeq_set = sTrainRTransY2 = [[copy.deepcopy(sSeq0), sSeqCreateRTransY(iSeq_set[0][1], seed=-1)], 
#                             [copy.deepcopy(sSeq0), sSeqCreateRTransY(iSeq_set[1][1], seed=-1)], 
#                             [copy.deepcopy(sSeq0), sSeqCreateRTransY(iSeq_set[2][1], seed=-1)],
#                             [copy.deepcopy(sSeq0), sSeqCreateRTransY(iSeq_set[3][1], seed=-1)],                              
#                             [copy.deepcopy(sSeq0), sSeqCreateRTransY(iSeq_set[4][1], seed=-1)],
#                             ]


print sSeq_set[0][0].subimage_width 


iSeq = iSeenidRTransY2 = iSeqCreateRTransY(10000, av_fim, first_image=10000, repetition_factor=1, seed=-1)
sSeenidRTransY2 = sSeqCreateRTransY(iSeq, seed=-1)
print sSeenidRTransY2.subimage_width

iSeq_set = iNewidRTransY2 = [[iSeqCreateRTransY(5000, av_fim, first_image=20000, repetition_factor=1, seed=-1)]]
sNewidRTransY2 = [[sSeqCreateRTransY(iSeq_set[0][0], seed=-1)]]
print sSeq_set[0][0].subimage_width


ParamsRTransYFunc = SystemParameters.ParamsSystem()
ParamsRTransYFunc.name = "Function Based Data Creation for RTransY"
ParamsRTransYFunc.network = linearNetwork4L #Default Network, but ignored
ParamsRTransYFunc.iTrain =iTrainRTransY2
ParamsRTransYFunc.sTrain = sTrainRTransY2

ParamsRTransYFunc.iSeenid = iSeenidRTransY2
ParamsRTransYFunc.sSeenid = sSeenidRTransY2

ParamsRTransYFunc.iNewid = iNewidRTransY2
ParamsRTransYFunc.sNewid = sNewidRTransY2

ParamsRTransYFunc.block_size = iTrainRTransY2[0][0].block_size
ParamsRTransYFunc.train_mode = 'clustered' #clustered improves final performance! mixed
ParamsRTransYFunc.analysis = None
ParamsRTransYFunc.enable_reduced_image_sizes = True
ParamsRTransYFunc.reduction_factor = 8.0 # WARNING 1.0, 2.0, 4.0, 8.0
ParamsRTransYFunc.hack_image_size = 16 # WARNING 128, 64, 32, 16
ParamsRTransYFunc.enable_hack_image_size = True



#Mixed training PosX & PosY
#NOTE: This is a hack to avoid repetition, the logic needs to be improved
iTrainRTransXY2 = [[copy.deepcopy(iTrainRTransX2[0][0]), copy.deepcopy(iTrainRTransY2[0][0])],
                   None,
                   None,
                   None,
                   [copy.deepcopy(iTrainRTransX2[0][0])],
                   ] 
                   
sTrainRTransXY2 = [[copy.deepcopy(sTrainRTransX2[0][0]), copy.deepcopy(sTrainRTransY2[0][0])],
                   None,
                   None,
                   None,
                   [copy.deepcopy(sTrainRTransX2[0][0])],
                   ]

ParamsRTransXYFunc = SystemParameters.ParamsSystem()
ParamsRTransXYFunc.name = "Function Based Data Creation for RTransY with generic data RTransX & RTransY"
ParamsRTransXYFunc.network = linearNetwork4L #Default Network, but ignored
ParamsRTransXYFunc.iTrain =iTrainRTransXY2
ParamsRTransXYFunc.sTrain = sTrainRTransXY2

ParamsRTransXYFunc.iSeenid = iSeenidRTransX2
ParamsRTransXYFunc.sSeenid = sSeenidRTransX2

ParamsRTransXYFunc.iNewid = iNewidRTransX2
ParamsRTransXYFunc.sNewid = sNewidRTransX2

ParamsRTransXYFunc.block_size = iTrainRTransXY2[0][0].block_size
ParamsRTransXYFunc.train_mode = 'clustered' #clustered improves final performance! mixed
ParamsRTransXYFunc.analysis = None
ParamsRTransXYFunc.enable_reduced_image_sizes = True
ParamsRTransXYFunc.reduction_factor = 8.0 # WARNING 1.0, 2.0, 4.0, 8.0
ParamsRTransXYFunc.hack_image_size = 16 # WARNING 128, 64, 32, 16
ParamsRTransXYFunc.enable_hack_image_size = True


##############################################
#The German Traffic Sign Recognition Benchmark
##############################################
#Base code originally taken from the competition website
import csv

# function for reading the image annotations and labels
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of image filenames, list of all tracks, list of all image annotations, list of all labels, 
def readTrafficSignsAnnotations(rootpath, shrink_signs=True, shrink_factor = 0.8, correct_sizes=True, include_labels=False):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
#    images = [] # image filenames
    annotations =  [] #annotations for each image
#    labels = [] # corresponding labels
#    tracks = []
    # loop over all 42 classes
    delta_rand_scale=0.0
    repetition_factors=None
    
    if repetition_factors == None:
        repetition_factors = [1]*43
    for c in range(0,43):
        prefix = rootpath + '/' + "%05d"%c + '/' # subdirectory for class. format(c, '05d')
        gtFile = open(prefix + 'GT-'+ "%05d"%c + '.csv') # annotations file. format(c, '05d')
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        gtReader.next() # skip header
        # loop over all images in current annotations file
        for ii, row in enumerate(gtReader):           
#            if ii%1000==0:
#                print row
            image_filename = prefix + row[0]
            extended_row = [image_filename] # extended_row: filename, track number, im_width, im_height, x0, y0, x1, y1
            if correct_sizes:   
                im = Image.open(image_filename)
#                row[1:3] = map(int, row[1:3])
#                if row[1] != im.size[0] or row[2] != im.size[1]:
#                    print "Image %s has incorrect size label"%image_filename, row[1:3], im.size[0:2]
                row[1] = im.size[0]
                row[2] = im.size[1]
                del im
            extended_row.append(int(row[0][0:5])) #Extract track number
            if shrink_signs:
                sign_coordinates = map(float, row[3:7])
                center_x = (sign_coordinates[0] + sign_coordinates[2])/2.0
                center_y = (sign_coordinates[1] + sign_coordinates[3])/2.0
                rand_scale_factor1 = 1.0 + numpy.random.uniform(-delta_rand_scale,delta_rand_scale)
                rand_scale_factor2 = 1.0 + numpy.random.uniform(-delta_rand_scale,delta_rand_scale)
                width = (sign_coordinates[2] - sign_coordinates[0]+1)*shrink_factor* rand_scale_factor1
                height = (sign_coordinates[3] - sign_coordinates[1]+1)*shrink_factor * rand_scale_factor2         
                row[3] = center_x - width/2
                row[5] = center_x + width/2
                row[4] = center_y - height/2
                row[6] = center_y + height/2
            extended_row = extended_row + map(float, row[1:7])
            if include_labels:
                extended_row.append(int(row[7])) # the 8th column is the label
            for i in range(repetition_factors[c]):
                annotations.append(extended_row)
        gtFile.close()

    return annotations


def readTrafficSignsAnnotationsOnline(prefix, csv_file, shrink_signs=True, correct_sizes=True, include_labels=False):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # image filenames
    annotations =  [] #annotations for each image
    labels = [] # corresponding labels
    tracks = []
    # loop over all 42 classes
    gtFile = open(prefix + '/' + csv_file) # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    gtReader.next() # skip header
    # loop over all images in current annotations file
    for ii, row in enumerate(gtReader):           
#            if ii%1000==0:
#                print row
        image_filename = prefix + "/" + row[0]
        extended_row = [image_filename] # extended_row: filename, track number, im_width, im_height, x0, y0, x1, y1
        if correct_sizes:   
            im = Image.open(image_filename)
#                row[1:3] = map(int, row[1:3])
#                if row[1] != im.size[0] or row[2] != im.size[1]:
#                    print "Image %s has incorrect size label"%image_filename, row[1:3], im.size[0:2]
            row[1] = im.size[0]
            row[2] = im.size[1]
            del im
        extended_row.append(0) #Fictuous track number
        if shrink_signs:
            shrink_factor = 0.8
            sign_coordinates = map(float, row[3:7])
            center_x = (sign_coordinates[0] + sign_coordinates[2])/2.0
            center_y = (sign_coordinates[1] + sign_coordinates[3])/2.0
            width = (sign_coordinates[2] - sign_coordinates[0]+1)*shrink_factor
            height = (sign_coordinates[3] - sign_coordinates[1]+1)*shrink_factor                
            row[3] = center_x - width/2
            row[5] = center_x + width/2
            row[4] = center_y - height/2
            row[6] = center_y + height/2
        extended_row = extended_row + map(float, row[1:7])
        if include_labels:
            extended_row.append(int(row[7])) # the 8th column is the label, not available on online test
        else:
            extended_row.append(ii*43/12569)
        annotations.append(extended_row)
    gtFile.close()
    return annotations


def count_annotation_classes(annotations):
    classes = []
    count = []
    for row in annotations:
        classes.append(row[8])
    classes = numpy.array(classes)
    for c in range(0,43):
        print "  class %d :"%c, (classes[:]==c).sum(),
        count.append((classes[:]==c).sum())
    return numpy.array(count)


def sample_annotations(annotations, flag=None, passing=1.0):
    num_annot = len(annotations)
    #W!!! why 2 here!!!?? 
    w = numpy.random.uniform(0.0, 1.0, size=num_annot)
    mask = (w <= passing)

    t=0
    out_annotations = []    
    for ii, row in enumerate(annotations):
        if (flag=="Test" or flag==None):
            if mask[ii] and 2 <= row[1] < 4: # Keep only track 0 and 1 for testing
                out_annotations.append(row)
                t += 1
        elif flag=="Even":
            if mask[ii] and row[1]%2==0: # and (row[1]<2 or row[1]>=4): # row[1] >= 2
                out_annotations.append(row)
        elif flag=="Odd":
            if mask[ii] and row[1]%2==1: # and (row[1]<2 or row[1]>=4): #row[1]>=2:
                out_annotations.append(row)
        elif flag=="2/3":
            if mask[ii] and (row[1]%3==1 or row[1]%3==2): # and (row[1]<2 or row[1]>=4): # row[1] >= 2
                out_annotations.append(row)
        elif flag=="4/3":
            if mask[ii] and (row[1]%3==0 or row[1]%3==1): # and (row[1]<2 or row[1]>=4): # row[1] >= 2
                out_annotations.append(row)
        elif flag=="1/3":
            if mask[ii] and row[1]%3==0: # and (row[1]<2 or row[1]>=4): # row[1] >= 2
                out_annotations.append(row)
        elif flag=="AllP":
            if mask[ii] and (row[1]<2 or row[1]>=4): #row[1]>=2:
                out_annotations.append(row)
        elif flag=="Univ":
            if mask[ii]:
                out_annotations.append(row)
        else:
            quit()
    print "Track 0: ", t
    return out_annotations

GTSRB_Images_dir_training = "/local/escalafl/Alberto/GTSRB/Training"
GTSRB_HOG_dir_training = "/local/escalafl/Alberto/GTSRB/GTSRB_Features_HOG/training"
GTSRB_SFA_dir_training = "/local/escalafl/Alberto/GTSRB/GTSRB_Features_SFA/training"

GTSRB_Images_dir_Online = "/local/escalafl/Alberto/GTSRB/Online-Test-sort/Images"
GTSRB_HOG_dir_Online = "/local/escalafl/Alberto/GTSRB/Online-Test-sort/HOG"
GTSRB_SFA_dir_Online = "/local/escalafl/Alberto/GTSRB/Online-Test-sort/SFA"

GTSRB_Images_dir_UnlabeledOnline = "/local/escalafl/Alberto/GTSRB/Online-Test/Images"
GTSRB_HOG_dir_UnlabeledOnline = "/local/escalafl/Alberto/GTSRB/Online-Test/HOG"
GTSRB_SFA_dir_UnlabeledOnline = "/local/escalafl/Alberto/GTSRB/Online-Test/SFA"

      
#Switch either HOG or SFA here
def load_HOG_data(base_GTSRB_dir="/home/eban/GTSRB", filenames=None, switch_SFA_over_HOG="HOG02", feature_noise = 0.0, padding=False):
    all_data = []
    print "HOG DATA LOADING %d images..."%len(filenames)
    online_base_dir = GTSRB_Images_dir_UnlabeledOnline #Unlabeled data dir

    #warning, assumes all filenames belong to same directory!!!
    #make this a good function and apply it to every file
    if filenames[0][0:len(online_base_dir)] == online_base_dir: #or final data base dir
        if switch_SFA_over_HOG in ["HOG01", "HOG02", "HOG03"]:     
            base_hog_dir = GTSRB_HOG_dir_UnlabeledOnline
        elif switch_SFA_over_HOG in ["SFA"]:
            base_hog_dir = GTSRB_SFA_dir_UnlabeledOnline            
        else:
            er = "Incorrect 1 switch_SFA_over_HOG value:", switch_SFA_over_HOG
            raise Exception(er)
        unlabeled_data = True
    elif filenames[0][0:len(GTSRB_Images_dir_training)] == GTSRB_Images_dir_training:
        if switch_SFA_over_HOG in ["HOG01", "HOG02", "HOG03"]:     
            base_hog_dir = GTSRB_HOG_dir_training
        elif switch_SFA_over_HOG in ["SFA02"]:
            base_hog_dir = GTSRB_SFA_dir_training
        else:
            er = "Incorrect 2 switch_SFA_over_HOG value:", switch_SFA_over_HOG
            raise Exception(er)
        unlabeled_data = False
    elif filenames[0][0:len(GTSRB_Images_dir_Online)] == GTSRB_Images_dir_Online:
        if switch_SFA_over_HOG in ["HOG01", "HOG02", "HOG03"]:     
            base_hog_dir = GTSRB_HOG_dir_Online
        elif switch_SFA_over_HOG in ["SFA02"]:
            base_hog_dir = GTSRB_SFA_dir_Online
        else:
            er = "Incorrect 2.3 switch_SFA_over_HOG value:", switch_SFA_over_HOG
            raise Exception(er)
        unlabeled_data = False
    else:
        er = "Filename does not belong to known data sets: ", filenames[0]
        raise Exception(er)

    feat_set = switch_SFA_over_HOG[-2:]
    sample_hog_filename = "00000/00001_00000.txt"
    print filenames[0], filenames[-1]
    for ii, image_filename in enumerate(filenames):
        if unlabeled_data:
            if switch_SFA_over_HOG in ["HOG01", "HOG02", "HOG03"]:   
                hog_filename = base_hog_dir + "/HOG_" + feat_set + "/" + image_filename[-9:-3]+"txt"
            elif switch_SFA_over_HOG in ["SFA02"]:
                hog_filename = base_hog_dir + "/SFA_" + feat_set + "/" + image_filename[-9:-3]+"txt"
            else:
                er = "Incorrect 3 switch_SFA_over_HOG value:", switch_SFA_over_HOG
                raise Exception(er)
        else:
            if switch_SFA_over_HOG in ["HOG01", "HOG02", "HOG03"]:
                hog_filename = base_hog_dir + "/HOG_" + feat_set + "/" + image_filename[-len(sample_hog_filename):-3]+"txt"
            elif switch_SFA_over_HOG in ["SFA02"]:
                hog_filename = base_hog_dir + "/SFA_" + feat_set + "/" + image_filename[-len(sample_hog_filename):-3]+"txt"
            else:
                er = "Incorrect 4 switch_SFA_over_HOG value:", switch_SFA_over_HOG
                raise Exception(er)               
        if ii==0:
            print hog_filename
        data_file = open(hog_filename, "rb") 
        data = [float(line) for line in data_file.readlines()]
        data_file.close( )
        data = numpy.array(data)
        #print data    
        all_data.append(data)
    all_data = numpy.array(all_data)
    
    #0.035 => New Id: 0.938 CR_CCC, CR_Gauss 0.943,
    #0.025 => New Id: 0.938 CR_CCC, CR_Gauss 0.940,
    #Adding repetitions for balancing: 0.025 => New Id: 0.939 CR_CCC, CR_Gauss 0.949
    #Adding repetitions for balancing: 0.04 => New Id: 0.936 CR_CCC, CR_Gauss 0.956
    if feature_noise > 0.0:
        noise = numpy.random.normal(loc=0.0, scale=feature_noise, size=all_data.shape)
        print "feature (SFA/HOG) Noise added in amount:", feature_noise
        all_data += noise
    else:
        print "NO feature (SFA/HOG) Noise added"    

    if padding:
        if switch_SFA_over_HOG in ["HOG01", "HOG02"]:
            num_samples = all_data.shape[0]
            true_feature_data_shape = (num_samples, 14, 14, 8) 
            desired_feature_data_shape = (num_samples, 16, 16, 8)
            all_data = numpy.reshape(all_data, true_feature_data_shape)
            noise_data = numpy.random.normal(loc=1.0, scale=0.05, size=desired_feature_data_shape) 
            noise_data[:, 0:14, 0:14, 0:8] = all_data[:,:,:,:]
            all_data = numpy.reshape(noise_data, (num_samples, 16*16*8))
        else:
            er = "Padding not supported for data type: ", switch_SFA_over_HOG
            raise Exception(er)
    return all_data


#Real last_track is num_tracks - skip_end_tracks -1
def iSeqCreateRGTSRB_Labels(annotations, first_track=0, last_track=500, labels_available=True, repetition_factors=1, seed=-1): 
    if seed >= 0 or seed == None:
        numpy.random.seed(seed)

    print "***** Setting Training Information Parameters for German Traffic Sign Recognition Benchmark ******"
    iSeq = SystemParameters.ParamsInput()
    iSeq.name = "German Traffic Sign Recognition Benchmark"

    iSeq.data_base_dir = ""
#
    num_classes = 43
    c_filenames = []
    c_info = []
    c_labels = []
    for c in range(num_classes):
        c_filenames.append([])
        c_info.append([])
        c_labels.append([])

    counter = 0
    for row in annotations:
#        print "T=%d"%row[1],
        if row[1] >= first_track and row[1] < last_track:
            c = row[8] #extract class number
#            print c, len(c_filenames)
            c_filenames[c].append(row[0])
            c_info[c].append(row[2:8])
            c_labels[c].append(c)
            counter += 1
    print "Sample counter before repetition=", counter

    if repetition_factors == None:
        repetition_factors = 1
    if isinstance(repetition_factors, int):
        repetition_factors = [repetition_factors]*num_classes
    print repetition_factors
    for c in range(num_classes):  
        print "c=", c, len(c_info[c]), len(c_info), len(c_info[0]), len(c_info[1])
        c_filenames[c] = (c_filenames[c]) * (repetition_factors[c])
        c_info[c] = (c_info[c]) * (repetition_factors[c])
        c_labels[c] = (c_labels[c]) * (repetition_factors[c])  
    
##    counter = 0
##    for row in annotations:
###        print "T=%d"%row[1],
##        if row[1] >= first_track and row[1] < last_track:
##            c = row[8] #extract class number
###            print c, len(c_filenames)
##            c_filenames[c].append(row[0])
##            c_info[c].append(row[2:8])
##            c_labels[c].append(c)
##            counter += 1
##    print "Sample counter after repetition=", counter
##    quit()
#    quit()
#    print c_filenames[0][0], c_filenames[1][0]
#    print c_filenames[0][1], c_filenames[1][1]
    
    conc_c_info = []
    for entry in c_info:
        conc_c_info += entry
    #Avoids void entries
    iSeq.c_info = numpy.array(conc_c_info)

    print len(iSeq.c_info)
    #zeros((counter, 6))

    iSeq.ids = []
    iSeq.input_files = []
    iSeq.block_size = []
    for c in range(num_classes):
        iSeq.block_size.append(len(c_filenames[c]))
        print "iSeq.block_size[%d]="%c, iSeq.block_size[c]
        for filename in c_filenames[c]:
            iSeq.ids.append(c)
            iSeq.input_files.append(filename)

    iSeq.block_size = numpy.array(iSeq.block_size)
#    print iSeq.block_size
#    quit()
#    ii = 130
#    print ii, iSeq.input_files[ii], iSeq.c_info[ii]
#    print len(iSeq.input_files), len(iSeq.c_info)
#    quit()
      
    iSeq.ids = numpy.array(iSeq.ids)
    print "iSeq.ids", iSeq.ids
#    quit()
    iSeq.ages = [None]
    iSeq.genders = [None]
    iSeq.racetweens = [None]
    iSeq.expressions = [None]
    iSeq.morphs = [None]
    iSeq.poses = [None]
    iSeq.lightings = [None]
    iSeq.slow_signal = 0 #real slow signal is the class number (type of sign)
    iSeq.step = 1
    iSeq.offset = 0
    
    iSeq.num_images = len(iSeq.input_files)
    #iSeq.params = [ids, expressions, morphs, poses, lightings]
    iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                      iSeq.morphs, iSeq.poses, iSeq.lightings]
    if labels_available:
        iSeq.train_mode = "clustered"    
    else:
        iSeq.train_mode = "unlabeled"
    print "BLOCK SIZE =", iSeq.block_size 
    iSeq.correct_classes = copy.deepcopy(iSeq.ids)
    iSeq.correct_labels = copy.deepcopy(iSeq.ids) 
    SystemParameters.test_object_contents(iSeq)
    return iSeq



def sSeqCreateRGTSRB(iSeq, delta_translation = 2.0, delta_scaling = 0.1, delta_rotation = 4.0, contrast_enhance=True, activate_HOG=False, switch_SFA_over_HOG="HOG02", feature_noise = 0.0, seed=-1):
    if seed >= 0 or seed == None: #also works for 
        numpy.random.seed(seed)
    #else seed <0 then, do not change seed

    print "******** Setting Training Data Parameters for German Traffic Sign Recognition Benchmark  ****************"
    sSeq = SystemParameters.ParamsDataLoading()
    sSeq.input_files = iSeq.input_files
    sSeq.num_images = iSeq.num_images
    sSeq.block_size = iSeq.block_size
    sSeq.train_mode = iSeq.train_mode
    sSeq.image_width = iSeq.c_info[:, 0]
    sSeq.image_height = iSeq.c_info[:,1]
    sSeq.subimage_width = 32
    sSeq.subimage_height = 32 
    
#    sign_size = ((iSeq.c_info[:,4] - iSeq.c_info[:,2]) + (iSeq.c_info[:,5] - iSeq.c_info[:,3]))/2.0
    #Keep aspect ratio as in the original
#    sSeq.pixelsampling_x = sign_size * 1.0 /  sSeq.subimage_width
#    sSeq.pixelsampling_y = sign_size * 1.0 /  sSeq.subimage_height
#    sSeq.subimage_first_row =  (iSeq.c_info[:,5] + iSeq.c_info[:,3])/2.0 - sign_size / 2.0
#    sSeq.subimage_first_column = (iSeq.c_info[:,4] + iSeq.c_info[:,2])/2.0 - sign_size / 2.0

    sSeq.scales = 1+numpy.random.uniform(-delta_scaling, delta_scaling, sSeq.num_images)
        
    sign_widths = (iSeq.c_info[:,4] - iSeq.c_info[:,2]+1) * sSeq.scales
    sign_heights = iSeq.c_info[:,5] - iSeq.c_info[:,3]+1 * sSeq.scales
    sign_centers_x = (iSeq.c_info[:,4] + iSeq.c_info[:,2])*0.5
    sign_centers_y = (iSeq.c_info[:,5] + iSeq.c_info[:,3])*0.5

    sSeq.pixelsampling_x = sign_widths /  sSeq.subimage_width
    sSeq.pixelsampling_y = sign_heights /  sSeq.subimage_height   
    sSeq.subimage_first_row = sign_centers_y - sign_heights/2 
    sSeq.subimage_first_column = sign_centers_x - sign_widths/2  

    #sSeq.image_width/2.0-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
    sSeq.add_noise_L0 = True
    if activate_HOG==False:
        sSeq.convert_format = "RGB"       
    else:
        sSeq.convert_format = switch_SFA_over_HOG

    sSeq.background_type = None


    sSeq.translations_x = numpy.random.uniform(-delta_translation, delta_translation, sSeq.num_images)                                                           
    sSeq.translations_y = numpy.random.uniform(-delta_translation, delta_translation, sSeq.num_images)                                                           
    
    sSeq.rotation = numpy.random.uniform(-delta_rotation, delta_rotation, sSeq.num_images)
        
#    if contrast_enhance == False:
#        print "WTF 2???"
#        quit()
        
    sSeq.contrast_enhance = contrast_enhance
    sSeq.trans_sampled = True #Translations are given in subimage coordinates
    sSeq.name = "GTSRB"
    iSeq.name = sSeq.name

    if activate_HOG == False:
        sSeq.load_data = load_data_from_sSeq
    else:
        if switch_SFA_over_HOG in ["HOG01", "HOG02", "HOG03", "SFA02"]:
            sSeq.load_data = lambda sSeq: load_HOG_data(base_GTSRB_dir="/local/scalafl/Alberto/GTSRB", filenames=sSeq.input_files, switch_SFA_over_HOG=switch_SFA_over_HOG,feature_noise=feature_noise,padding=True)
        elif switch_SFA_over_HOG == "SFA02_HOG02":
            sSeq.load_data = lambda sSeq: numpy.concatenate((
                load_HOG_data(base_GTSRB_dir="/local/scalafl/Alberto/GTSRB", filenames=sSeq.input_files, switch_SFA_over_HOG="SFA02",feature_noise=feature_noise)[:,0:49*2],
                load_HOG_data(base_GTSRB_dir="/local/scalafl/Alberto/GTSRB", filenames=sSeq.input_files, switch_SFA_over_HOG="HOG02",feature_noise=feature_noise)),axis=1)
        else:
            er = "Unknown switch_SFA_over_HOG:", switch_SFA_over_HOG
            raise Exception(er)

        if switch_SFA_over_HOG in ["HOG01", "HOG02"]:
            sSeq.subimage_width = 16 #14 #49
            sSeq.subimage_height = 16 #14 #32 #channel dim = 8
            sSeq.convert_format = switch_SFA_over_HOG
        elif switch_SFA_over_HOG in ["HOG03"]:
            sSeq.subimage_width = 54
            sSeq.subimage_height = 54
            sSeq.convert_format = switch_SFA_over_HOG    
        elif switch_SFA_over_HOG == "SFA02":
            sSeq.subimage_width = 7
            sSeq.subimage_height = 7
            sSeq.convert_format = switch_SFA_over_HOG
        elif switch_SFA_over_HOG == "SFA02_HOG02":
            sSeq.subimage_width = 49
            sSeq.subimage_height = 34
            sSeq.convert_format = switch_SFA_over_HOG
        else:
            quit()
    sSeq.train_mode = 'clustered'
    SystemParameters.test_object_contents(sSeq)
    return sSeq



#Annotations: filename, track, Width, Height, ROI.x1, ROI.y1, ROI.x2, ROI.y2, [label]

#class 0 : 150   class 1 : 1500   class 2 : 1500   class 3 : 960   class 4 : 1320   class 5 : 1260   class 6 : 300   class 7 : 960   class 8 : 960   
#class 9 : 990   class 10 : 1350   class 11 : 900   class 12 : 1410   class 13 : 1440   class 14 : 540   class 15 : 420   class 16 : 300   
#class 17 : 750   class 18 : 810   class 19 : 150   class 20 : 240   class 21 : 240   class 22 : 270   class 23 : 360   class 24 : 180   
#class 25 : 1020   class 26 : 420   class 27 : 180   class 28 : 360   class 29 : 180   class 30 : 300   class 31 : 540   class 32 : 180   
#class 33 : 480   class 34 : 300   class 35 : 810   class 36 : 270   class 37 : 150   class 38 : 1380   class 39 : 210   class 40 : 240   
#class 41 : 180   class 42 : 180

#repetition_factors = [10  1  1  1  1  1  5  1  1  1  1  1  1  1  2  3  5  2  1 10  6  6  5  4  8
#  1  3  8  4  8  5  2  8  3  5  1  5 10  1  7  6  8  8]

#repetition_factors = [5  1  1  1  1
#                      1  3  1  1  1
#                      1  1  1  1  2
#                      2  3  2  1  5  4  
#                      4  3  4  8
#  1  3  8  4  8  5  2  8  3  5  1  5 10  1  7  6  8  8]

#For all tracks:
#repetition_factorsTrain = [10,  1,  1,  1,  1,  1,  5,  1,  1,  1,  1,  1,  1,  1,  2,  3,  5,  2, 1, 10,  6,  6,  5,  4,  8,
#                       1,  3,  8,  4,  8,  5,  2,  8,  3,  5,  1,  5, 10,  1,  7,  6,  8,  8]


#To support even/odd distribution
#W
#repetition_factors = numpy.array(repetition_factors)*2
#repetition_factorsTrain = numpy.array(repetition_factorsTrain)*2

#At most 15 tracks:
#repetition_factors = [ 4, 1, 1, 1, 1, 1,
#  2, 1, 1, 1, 1, 1,
#  1, 1, 1, 2, 2, 1,
#  1, 4, 3, 3, 2, 2,
#  3, 1, 2, 3, 2, 3,
#  2, 1, 3, 2, 2, 1,
#  2, 4, 1, 3, 3, 3,
#  3]

GTSRB_present = os.path.lexists("/local/escalafl/Alberto/GTSRB/") and False
if GTSRB_present:
    GTSRBTrain_base_dir = "/local/escalafl/Alberto/GTSRB/Final_Training/Images" #"/local/escalafl/Alberto/GTSRB/Final_Training/Images"
    GTSRBOnline_base_dir = "/local/escalafl/Alberto/GTSRB/Online-Test-sort/Images" #"/local/escalafl/Alberto/GTSRB/Online-Test-sort/Images"
    repetition_factorsTrain = None
    
#TODO:Repetition factors should not be at this level, also randomization of scales does not belong here.
#TODO:Update the databases used to the final system
      
    GTSRB_annotationsTrain_Train = readTrafficSignsAnnotations(GTSRBTrain_base_dir, include_labels=True) #delta_rand_scale=0.07
    print "Len GTSRB_annotationsTrain_Train=,", len(GTSRB_annotationsTrain_Train)
    GTSRB_annotationsTrain_Seenid = readTrafficSignsAnnotations(GTSRBTrain_base_dir, include_labels=True) #delta_rand_scale=0.07
    GTSRB_annotationsTrain_Newid = readTrafficSignsAnnotations(GTSRBTrain_base_dir, include_labels=True)
    
#    GTSRB_annotationsOnline_Train = readTrafficSignsAnnotations(GTSRBOnline_base_dir, include_labels=True)
#    GTSRB_annotationsOnline_Seenid = readTrafficSignsAnnotations(GTSRBOnline_base_dir, include_labels=True)
#    GTSRB_annotationsOnline_Newid = readTrafficSignsAnnotations(GTSRBOnline_base_dir, include_labels=True)
#    print "Len GTSRB_annotationsOnline_Newid=", len(GTSRB_annotationsOnline_Newid)
#    quit()
    
    #count = count_annotation_classes(GTSRB_annotationsTest)
    #print count
    #print 500.0/count+0.99
    #print 700/count
    #print GTSRB_annotationsTrain[0]
    
    #     
    GTSRB_annotationsTrain_Train = sample_annotations(GTSRB_annotationsTrain_Train, flag="2/3", passing=1.0) #W 0.6, 0.5, Odd, AllP, Univ, 2/3
    GTSRB_annotationsTrain_Seenid = sample_annotations(GTSRB_annotationsTrain_Seenid, flag="1/3", passing=1.0) #W 1.0, 0.3, Even, 1/3
    GTSRB_annotationsTrain_Newid = sample_annotations(GTSRB_annotationsTrain_Newid, flag="Test", passing=1.0) #make testing faster for now, 0.3 , 1.0
    
 #   GTSRB_annotationsOnline_Train = sample_annotations(GTSRB_annotationsOnline_Train, flag="Univ", passing=0.25) #W 0.6, 0.5, Odd, AllP, Univ
 #   GTSRB_annotationsOnline_Seenid = sample_annotations(GTSRB_annotationsOnline_Seenid, flag="Univ", passing=0.25) #W 1.0, 0.3, Even
 #   GTSRB_annotationsOnline_Newid = sample_annotations(GTSRB_annotationsOnline_Newid, flag="Univ", passing=1.0) #make testing faster for now, 0.3 , 1.0
    
    #count = count_annotation_classes(GTSRB_annotationsTest)      
    #print count    
    #quit()
    
    GTSRB_Unlabeled_base_dir = "/local/escalafl/Alberto/GTSRB/Final_Test/Images" # "/local/escalafl/Alberto/GTSRB/Online-Test/Images"
    GTSRB_Unlabeled_csvfile =  "GT-final_test.csv" #  "GT-final_test.csv" / "GT-final_test.test.csv" / "GT-online_test.test.csv"
    GTSRB_UnlabeledAnnotations = readTrafficSignsAnnotationsOnline(GTSRB_Unlabeled_base_dir, GTSRB_Unlabeled_csvfile, shrink_signs=True, correct_sizes=True, include_labels=True) #include_labels=False
    GTSRB_UnlabeledAnnotations = sample_annotations(GTSRB_UnlabeledAnnotations, flag="Univ", passing=1.0) #make testing faster for now, 0.3 , 1.0
    print GTSRB_UnlabeledAnnotations[0], len(GTSRB_UnlabeledAnnotations)
    #quit() 
    
    GTSRB_annotationsTrain = GTSRB_annotationsTrain_Train
    GTSRB_annotationsSeenid = GTSRB_annotationsTrain_Seenid
    GTSRB_annotationsTest = GTSRB_UnlabeledAnnotations # GTSRB_UnlabeledAnnotations, GTSRB_annotationsOnline_Newid
    
    ## CASE [[F]]
    #WARNING!
    ##############################'''WAAAAAARNNNNIIINNNNNNGGGG TRTRAAAACKKKKSSSSSS 1111
    activate_HOG = True and False #for direct image processing, true for SFA/HOG features
    #TODO: HOG set selection HOG02, SFA, 
    switch_SFA_over_HOG = "SFA" # "HOG02", "SFA"
    #W first track=1
    ##iSeq_set = iTrainRGTSRB_Labels = [[iSeqCreateRGTSRB_Labels(GTSRB_annotationsTrain, first_track = 0, last_track=100, seed=-1),
    ##                                   iSeqCreateRGTSRB_Labels(GTSRB_annotationsTestData, first_track = 0, last_track=4, labels_available=False, seed=-1)]]
    ##sSeq_set = sTrainRGTSRB = [[sSeqCreateRGTSRB(iSeq_set[0][0], enable_translation=True, enable_rotation=True, contrast_enhance=True, activate_HOG=activate_HOG, switch_SFA_over_HOG= switch_SFA_over_HOG, add_HOG_noise=True, seed=-1),
    ##                            sSeqCreateRGTSRB(iSeq_set[0][1], enable_translation=False, enable_rotation=False, contrast_enhance=True, activate_HOG=activate_HOG, switch_SFA_over_HOG= switch_SFA_over_HOG, add_HOG_noise=True,  seed=-1)]]
    ##
    ##iSeq_set = iSeenidRGTSRB_Labels = iSeqCreateRGTSRB_Labels(GTSRB_annotationsSeenid, first_track = 0, last_track=100, seed=-1)
    ##sSeq_set = sSeenidRGTSRB = sSeqCreateRGTSRB(iSeq_set, enable_translation=True, enable_rotation=True, contrast_enhance=True, activate_HOG=activate_HOG, switch_SFA_over_HOG= switch_SFA_over_HOG, add_HOG_noise=True, seed=-1)
    
##    iSeq_set = iTrainRGTSRB_Labels = [[iSeqCreateRGTSRB_Labels(GTSRB_annotationsTrain, first_track = 0, last_track=100, seed=-1),
##                                       iSeqCreateRGTSRB_Labels(GTSRB_annotationsTest, first_track = 0, last_track=100, labels_available=False, seed=-1)]]
##    sSeq_set = sTrainRGTSRB = [[sSeqCreateRGTSRB(iSeq_set[0][0], enable_translation=True, enable_rotation=True, contrast_enhance=True, activate_HOG=activate_HOG, switch_SFA_over_HOG= switch_SFA_over_HOG, feature_noise=0.03, seed=-1),
##                                sSeqCreateRGTSRB(iSeq_set[0][1], enable_translation=True, enable_rotation=True, contrast_enhance=True, activate_HOG=activate_HOG, switch_SFA_over_HOG= switch_SFA_over_HOG, feature_noise=0.0,  seed=-1)]]

    #delta_translation=0.0, delta_scaling=0.1, delta_rotation=4.0
    contrast_enhance=False or True
    iSeq_set = iTrainRGTSRB_Labels = [[iSeqCreateRGTSRB_Labels(GTSRB_annotationsTrain, first_track = 0, last_track=100, repetition_factors=4, seed=-1)]] #last_track=100
#    sSeq_set = sTrainRGTSRB = [[sSeqCreateRGTSRB(iSeq_set[0][0], delta_translation=0.0, delta_scaling=0.0, delta_rotation=0.0, contrast_enhance=contrast_enhance, activate_HOG=activate_HOG, switch_SFA_over_HOG= switch_SFA_over_HOG, feature_noise=0.00, seed=-1) ]] #feature_noise=0.04
    sSeq_set = sTrainRGTSRB = [[sSeqCreateRGTSRB(iSeq_set[0][0], delta_translation=1.5, delta_scaling=0.1, delta_rotation=3.5, contrast_enhance=contrast_enhance, activate_HOG=activate_HOG, switch_SFA_over_HOG= switch_SFA_over_HOG, feature_noise=0.00, seed=-1) ]] #feature_noise=0.04
    
    iSeq_set = iSeenidRGTSRB_Labels = iSeqCreateRGTSRB_Labels(GTSRB_annotationsSeenid, first_track = 0, last_track=100, repetition_factors=2,  seed=-1)
#    sSeq_set = sSeenidRGTSRB = sSeqCreateRGTSRB(iSeq_set, delta_translation=0.0, delta_scaling=0.00, delta_rotation=0.0, contrast_enhance=contrast_enhance, activate_HOG=activate_HOG, switch_SFA_over_HOG= switch_SFA_over_HOG, feature_noise=0.00, seed=-1) #feature_noise=0.07
    sSeq_set = sSeenidRGTSRB = sSeqCreateRGTSRB(iSeq_set, delta_translation=1.25, delta_scaling=0.075, delta_rotation=2.5, contrast_enhance=contrast_enhance, activate_HOG=activate_HOG, switch_SFA_over_HOG= switch_SFA_over_HOG, feature_noise=0.00, seed=-1) #feature_noise=0.07
    
    #GTSRB_onlineAnnotations, GTSRB_annotationsTest
    iSeq_set = iNewidRGTSRB_Labels = iSeqCreateRGTSRB_Labels(GTSRB_annotationsTest, first_track = 0, last_track=4, repetition_factors=1, seed=-1)
    sSeq_set = sNewidRGTSRB = sSeqCreateRGTSRB(iSeq_set, delta_translation=0.0, delta_scaling=0.0, delta_rotation=0.0, contrast_enhance=contrast_enhance, activate_HOG=activate_HOG, switch_SFA_over_HOG= switch_SFA_over_HOG, feature_noise=0.0, seed=-1)
    #sSeq_set.add_noise_L0 = False
    #sSeq_set.rotation = 0.0              
    
#    print iTrainRGTSRB_Labels[0][0].input_files[0:5]
#    print iSeenidRGTSRB_Labels.input_files[0:5]
#    quit()
    
    ParamsRGTSRBFunc = SystemParameters.ParamsSystem()
    ParamsRGTSRBFunc.name = "Function Based Data Creation for GTSRB"
    ParamsRGTSRBFunc.network = linearNetwork4L #Default Network, but ignored
    ParamsRGTSRBFunc.iTrain =iTrainRGTSRB_Labels
    ParamsRGTSRBFunc.sTrain = sTrainRGTSRB
    
    ParamsRGTSRBFunc.iSeenid = iSeenidRGTSRB_Labels
    ParamsRGTSRBFunc.sSeenid = sSeenidRGTSRB
    
    ParamsRGTSRBFunc.iNewid = iNewidRGTSRB_Labels
    ParamsRGTSRBFunc.sNewid = sNewidRGTSRB
    
    ParamsRGTSRBFunc.block_size = iTrainRGTSRB_Labels[0][0].block_size
    ParamsRGTSRBFunc.train_mode = 'clustered' #Identity recognition task
    ParamsRGTSRBFunc.analysis = None
    ParamsRGTSRBFunc.activate_HOG = activate_HOG
    
    if activate_HOG==False:
        ParamsRGTSRBFunc.enable_reduced_image_sizes = True #and False #Set to false if network is a cascade
        ParamsRGTSRBFunc.reduction_factor = 1.0 # WARNING 0.5, 1.0, 2.0, 4.0, 8.0
        ParamsRGTSRBFunc.hack_image_size = 32 # WARNING    64,  32,  16,   8
        ParamsRGTSRBFunc.enable_hack_image_size = True #and False #Set to false if network is a cascade
    else:
        if switch_SFA_over_HOG == "HOG02":
            ParamsRGTSRBFunc.enable_reduced_image_sizes = False
            ParamsRGTSRBFunc.enable_hack_image_size = True
            ParamsRGTSRBFunc.hack_image_size = 16 # WARNING    32,  16,   8
        else:
            ParamsRGTSRBFunc.enable_reduced_image_sizes = False
            ParamsRGTSRBFunc.enable_hack_image_size = False
else:
    print "GTSRBFunc not present or disabled"
    ParamsRGTSRBFunc = None


### Function based definitions for face detections
###PIPELINE FOR FACE DETECTION:
###Orig=TX: DX0=+/- 45, DY0=+/- 20, DS0= 0.55-1.1
###TY: DX1=+/- 20, DY0=+/- 20, DS0= 0.55-1.1
###S: DX1=+/- 20, DY1=+/- 10, DS0= 0.55-1.1
###TMX: DX1=+/- 20, DY1=+/- 10, DS1= 0.775-1.05
###TMY: DX2=+/- 10, DY1=+/- 10, DS1= 0.775-1.05
###MS: DX2=+/- 10, DY2=+/- 5, DS1= 0.775-1.05
###Out About: DX2=+/- 10, DY2=+/- 5, DS2= 0.8875-1.025
##notice: for dx* and dy* intervals are open, while for smin and smax intervals are closed
#pipeline_fd = dict(dx0 = 45, dy0 = 20, smin0 = 0.55, smax0 = 1.1,
#                dx1 = 20, dy1 = 10, smin1 = 0.775, smax1 = 1.05)
##Pipeline actually supports inputs in: [-dx0, dx0-2] [-dy0, dy0-2] [smin0, smax0] 
##Remember these values are before image resizing
#
##XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
##This network actually supports images in the closed intervals: [smin0, smax0] [-dy0, dy0]
##but halb-open [-dx0, dx0) 
#print "***** Setting Training Information Parameters for Real Translation X ******"
#iSeq = iTrainRTransX = SystemParameters.ParamsInput()
#iSeq.name = "Real Translation X: (-45, 45, 2) translation and y 40"
#iSeq.data_base_dir = frgc_normalized_base_dir
#iSeq.ids = numpy.arange(0,7965) # 8000, 7965
#
#iSeq.trans = numpy.arange(-1 * pipeline_fd['dx0'], pipeline_fd['dx0'], 2) # (-50, 50, 2)
#if len(iSeq.ids) % len(iSeq.trans) != 0:
#    ex="Here the number of translations must be a divisor of the number of identities"
#    raise Exception(ex)
#iSeq.ages = [None]
#iSeq.genders = [None]
#iSeq.racetweens = [None]
#iSeq.expressions = [None]
#iSeq.morphs = [None]
#iSeq.poses = [None]
#iSeq.lightings = [None]
#iSeq.slow_signal = 0 #real slow signal is the translation in the x axis, added during image loading
#iSeq.step = 1
#iSeq.offset = 0
#iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
#                                            iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
#                                            iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=4, image_postfix=".jpg")
#iSeq.input_files = iSeq.input_files * 4 # warning!!! 4, 8
##To avoid grouping similar images next to one other
#numpy.random.shuffle(iSeq.input_files)  
#
#iSeq.num_images = len(iSeq.input_files)
##iSeq.params = [ids, expressions, morphs, poses, lightings]
#iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
#                  iSeq.morphs, iSeq.poses, iSeq.lightings]
#iSeq.block_size = iSeq.num_images / len (iSeq.trans)
#print "BLOCK SIZE =", iSeq.block_size 
#iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.trans)), iSeq.block_size)
#iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
#
#SystemParameters.test_object_contents(iSeq)
#
#print "******** Setting Training Data Parameters for Real TransX  ****************"
#sSeq = sTrainRTransX = SystemParameters.ParamsDataLoading()
#sSeq.input_files = iSeq.input_files
#sSeq.num_images = iSeq.num_images
#sSeq.block_size = iSeq.block_size
#sSeq.image_width = 256
#sSeq.image_height = 192
#sSeq.subimage_width = 128
#sSeq.subimage_height = 128 
#
#
#sSeq.trans_x_max = pipeline_fd['dx0']
#sSeq.trans_x_min = -1 * pipeline_fd['dx0']
#
##WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#sSeq.trans_y_max = pipeline_fd['dy0']
#sSeq.trans_y_min = -1 * sSeq.trans_y_max
#
##iSeq.scales = numpy.linspace(0.5, 1.30, 16) # (-50, 50, 2)
#sSeq.min_sampling = pipeline_fd['smin0']
#sSeq.max_sampling = pipeline_fd['smax0']
#
#sSeq.pixelsampling_x = sSeq.pixelsampling_y = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
##sSeq.subimage_pixelsampling = 2
#sSeq.subimage_first_row =  sSeq.image_height/2.0-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
#sSeq.subimage_first_column = sSeq.image_width/2.0-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
##sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
#sSeq.add_noise_L0 = True
#sSeq.convert_format = "L"
#sSeq.background_type = None
##random translation for th w coordinate
##sSeq.translation = 20 # 25, Should be 20!!! Warning: 25
##sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)                                                           
#sSeq.translations_x = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
#sSeq.translations_y = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, sSeq.num_images)
#sSeq.trans_sampled = True
#sSeq.name = "RTans X Dx in (%d, %d) Dy in (%d, %d), sampling in (%f, %f)"%(sSeq.trans_x_min, 
#    sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, sSeq.min_sampling*100, sSeq.max_sampling*100)
#SystemParameters.test_object_contents(sSeq)
experiment_seed = os.getenv("CUICUILCO_EXPERIMENT_SEED") #1112223339 #1112223339
if experiment_seed:
    experiment_seed = int(experiment_seed)
else:
    experiment_seed = 1112223334 #111222333
    ex = "CUICUILCO_EXPERIMENT_SEED unset"
    raise Exception(ex)
print "PosXseed. experiment_seed=", experiment_seed
numpy.random.seed(experiment_seed) #seed|-5789
print "experiment_seed=", experiment_seed

#This subsumes RTransX, RTransY, RTrainsScale
#pipeline_fd:
#dx0 = 45, 45 Steps,
#dy0 = 20, 20 Steps,
#smin0 = 0.55, smax0 = 1.1, 40 Steps
def iSeqCreateRTransXYScale(dx=45, dy=20, smin=0.55, smax=1.1, num_steps=20, slow_var = "X", continuous=False, pre_mirroring="none", num_images_used=10000, images_base_dir= alldbnormalized_base_dir, normalized_images = alldbnormalized_available_images, first_image_index=0, repetition_factor=1, seed=-1):
    if seed >= 0 or seed == None: #also works for 
        numpy.random.seed(seed)

    if first_image_index + num_images_used > len(alldbnormalized_available_images):
        err = "Images to be used exceeds the number of available images"
        raise Exception(err) 

    print "***** Setting Information Parameters for Real Translation XYScale ******"
    iSeq = SystemParameters.ParamsInput()
    iSeq.name = "Real Translation " + slow_var + ": (%d, %d, %d, %d) numsteps %d"%(dx, dy, smin, smax, num_steps)
    iSeq.data_base_dir = images_base_dir
    iSeq.ids = normalized_images[first_image_index:first_image_index + num_images_used] 
    iSeq.slow_var = slow_var
    iSeq.dx = dx
    iSeq.dy = dy
    iSeq.smin = smin
    iSeq.smax = smax
    iSeq.pre_mirroring = pre_mirroring
    
    if continuous == True:
        real_num_steps = num_images_used * repetition_factor
    else:
        real_num_steps = num_steps

    if iSeq.pre_mirroring == "duplicate":
        real_num_steps *= 2     

    #Here only the unique discrete values for each class are coumputed, these might need to be repeated multiple times
    if slow_var == "X":
        #iSeq.trans = numpy.arange(-1 * pipeline_fd['dx0'], pipeline_fd['dx0'], 2) # (-50, 50, 2)
        iSeq.trans = numpy.linspace(-1 * dx, dx, real_num_steps) # (-50, 50, 2)
    elif slow_var == "Y":
        iSeq.trans = numpy.linspace(-1 * dy, dy, real_num_steps)        
    elif slow_var == "Scale":
        iSeq.scales = numpy.linspace(smin, smax, real_num_steps)      
    else:
        er = "Wrong slow_variable: ", slow_var
        raise Exception(er)  

    if len(iSeq.ids) % len(iSeq.trans) != 0 and continuous == False:
        ex="Here the number of translations/scalings must be a divisor of the number of identities"
        raise Exception(ex)

    iSeq.ages = [None]
    iSeq.genders = [None]
    iSeq.racetweens = [None]
    iSeq.expressions = [None]
    iSeq.morphs = [None]
    iSeq.poses = [None]
    iSeq.lightings = [None]
    iSeq.slow_signal = 0 #real slow signal is the translation in the x axis (correlated to identity), added during image loading
    iSeq.step = 1
    iSeq.offset = 0
    iSeq.input_files = imageLoader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids, iSeq.ages, \
                                                iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, \
                                                iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset, len_ids=5, image_postfix=".jpg")
    ##WARNING! (comment this!)
#    dir_list = os.listdir(iSeq.data_base_dir)
#    iSeq.input_files = []
#    for filename in dir_list:
#        iSeq.input_files.append( os.path.join(iSeq.data_base_dir,filename) )

    print "number of input files=", len(iSeq.input_files)
    print "number of iSeq.ids=", len(iSeq.ids)

    iSeq.input_files = iSeq.input_files * repetition_factor # warning!!! 4, 8
    iSeq.num_images = len(iSeq.input_files)

    #To avoid grouping similar images next to one other, even though available images already shuffled
    numpy.random.shuffle(iSeq.input_files)  

    if iSeq.pre_mirroring == "none":
        iSeq.pre_mirror_flags = [False] * iSeq.num_images
    elif iSeq.pre_mirroring == "all":
        iSeq.pre_mirror_flags = [True] * iSeq.num_images
    elif iSeq.pre_mirroring == "random":
        iSeq.pre_mirror_flags = more_nodes.random_boolean_array(iSeq.num_images)
    elif iSeq.pre_mirroring == "duplicate":
        input_files_duplicated = list(iSeq.input_files)
        iSeq.pre_mirror_flags = more_nodes.random_boolean_array(iSeq.num_images)

        shuffling = numpy.arange(0, iSeq.num_images)
        numpy.random.shuffle(shuffling)
        
        input_files_duplicated = [input_files_duplicated[i] for i in shuffling]
        pre_mirror_flags_duplicated = iSeq.pre_mirror_flags[shuffling]^True

        iSeq.input_files.extend(input_files_duplicated)
        iSeq.pre_mirror_flags = numpy.concatenate((iSeq.pre_mirror_flags, pre_mirror_flags_duplicated))  
        iSeq.num_images *= 2
    else:
        er = "Erroneous parameter iSeq.pre_mirroring=",iSeq.pre_mirroring
        raise Exception(er)
    
    #iSeq.params = [ids, expressions, morphs, poses, lightings]
    iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                      iSeq.morphs, iSeq.poses, iSeq.lightings]

    iSeq.block_size = iSeq.num_images / num_steps

    if continuous == False:
        print "before len(iSeq.trans=", len(iSeq.trans)
        if slow_var == "X" or slow_var == "Y": 
            iSeq.trans = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
        elif slow_var == "Scale":
            iSeq.scales = sfa_libs.wider_1Darray(iSeq.scales, iSeq.block_size)
        else:
            er = "Wrong slow_variable: ", slow_var
            raise Exception(er)  
        print "after len(iSeq.trans=", len(iSeq.trans)
        
#    if continuous == False:
#        iSeq.train_mode = "serial" # = "serial" "mixed", None
#    else:
#        iSeq.train_mode = "mirror_window64" # "mirror_window32" # None, "regular", "window32", "fwindow16", "fwindow32", "fwindow64", "fwindow128", 

#        quit()
    iSeq.train_mode = "serial" # = "regular", "fwindow16" "serial" "mixed", None
#        iSeq.train_mode = None 

    print "BLOCK SIZE =", iSeq.block_size
    iSeq.correct_classes = numpy.arange(num_steps*iSeq.block_size)/iSeq.block_size
#    sfa_libs.wider_1Darray(numpy.arange(len(iSeq.trans)), iSeq.block_size)
#    if continuous == False:
#        iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
#    else:
    iSeq.correct_labels = iSeq.trans + 0.0
    SystemParameters.test_object_contents(iSeq)
    return iSeq

def sSeqCreateRTransXYScale(iSeq, seed=-1):
    if seed >= 0 or seed == None: #also works for 
        numpy.random.seed(seed)
    #else seed <0 then, do not change seed
    
    print "******** Setting Training Data Parameters for Real TransX  ****************"
    sSeq = SystemParameters.ParamsDataLoading()
    sSeq.input_files = iSeq.input_files
    sSeq.num_images = iSeq.num_images
    sSeq.block_size = iSeq.block_size
    sSeq.train_mode = iSeq.train_mode
    sSeq.include_latest = iSeq.include_latest
    sSeq.pre_mirror_flags = iSeq.pre_mirror_flags

    sSeq.image_width = 256
    sSeq.image_height = 192
    sSeq.subimage_width = 128
    sSeq.subimage_height = 128 
    
    sSeq.trans_x_max = iSeq.dx
    sSeq.trans_x_min = -1 * iSeq.dx
    
    #WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    sSeq.trans_y_max = iSeq.dy
    sSeq.trans_y_min = -1 * iSeq.dy
    
    #iSeq.scales = numpy.linspace(0.5, 1.30, 16) # (-50, 50, 2)
    sSeq.min_sampling = iSeq.smin
    sSeq.max_sampling = iSeq.smax
    
    #sSeq.subimage_pixelsampling = 2
    #sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
    sSeq.add_noise_L0 = True
    sSeq.convert_format = "L"
    sSeq.background_type = None
    #random translation for th w coordinate
    #sSeq.translation = 20 # 25, Should be 20!!! Warning: 25
    #sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)                                                           
    if iSeq.slow_var == "X":
        sSeq.translations_x = iSeq.trans
        #sSeq.translations_x = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
        print "sSeq.translations_x=", sSeq.translations_x
        print "len( sSeq.translations_x)=",  len(sSeq.translations_x)
    else:
        sSeq.translations_x = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, sSeq.num_images)
        
    if iSeq.slow_var == "Y":
        sSeq.translations_y = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
    else:
        sSeq.translations_y = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, sSeq.num_images)

    if iSeq.slow_var == "Scale":
        sSeq.pixelsampling_x = sfa_libs.wider_1Darray(iSeq.scales,  iSeq.block_size)
        sSeq.pixelsampling_y =  sfa_libs.wider_1Darray(iSeq.scales,  iSeq.block_size)
    else:
        sSeq.pixelsampling_x = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
        sSeq.pixelsampling_y = sSeq.pixelsampling_x + 0.0

    #Warning, code below seems to have been deleted at some point!!!
    sSeq.subimage_first_row =  sSeq.image_height/2.0-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
    sSeq.subimage_first_column = sSeq.image_width/2.0-sSeq.subimage_width*sSeq.pixelsampling_x/2.0

    sSeq.trans_sampled = True #TODO: Why are translations specified according to the sampled images?
    
    sSeq.name = "RTans XYScale %s Dx in (%d, %d) Dy in (%d, %d), sampling in (%d, %d)"%(iSeq.slow_var, sSeq.trans_x_min, 
        sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, int(sSeq.min_sampling*100), int(sSeq.max_sampling*100))

    print "Var in trans X is:", sSeq.translations_x.var()
    sSeq.load_data = load_data_from_sSeq
    SystemParameters.test_object_contents(sSeq)
    return sSeq

print ":)***********************"

alldbnormalized_available_images = numpy.arange(0,64470)
numpy.random.shuffle(alldbnormalized_available_images)  

#iSeq = iSeqCreateRTransY(4800, av_fim, first_image=0, repetition_factor=2, seed=-1)
#sSeq = sSeqCreateRTransY(iSeq, seed=-1)
#print ";) 3"
#quit()
normalized_base_dir_INIBilder = "/local/escalafl/Alberto/INIBilder/INIBilderNormalized"

## CASE [[F]]iSeqCreateRTransXYScale(dx=45, dy=20, smin=0.55, smax=1.1, num_steps=20, slow_var = "X", continuous=False, num_images_used=10000, images_base_dir= alldbnormalized_base_dir, normalized_images = alldbnormalized_available_images, first_image_index=0, repetition_factor=1, seed=-1):
continuous = True #and False

iSeq_set = iTrainRTransXYScale = [[iSeqCreateRTransXYScale(dx=45, dy=20, smin=0.55, smax=1.1, num_steps=50, slow_var = "X", continuous=continuous, num_images_used=30000, #30000 
                                                      images_base_dir=alldbnormalized_base_dir, normalized_images = alldbnormalized_available_images, 
                                                      first_image_index=0, pre_mirroring="none", repetition_factor=1, seed=-1)]]
#Experiment below is just for display purposes!!! comment it out!
#iSeq_set = iTrainRTransXYScale = [[iSeqCreateRTransXYScale(dx=45, dy=2, smin=0.85, smax=0.95, num_steps=50, slow_var = "X", continuous=continuous, num_images_used=15000, #30000 
#                                                      images_base_dir=alldbnormalized_base_dir, normalized_images = alldbnormalized_available_images, 
#                                                      first_image_index=0, repetition_factor=1, seed=-1)]]

#For generating INI images
#iSeq_set = iTrainRTransXYScale = [[iSeqCreateRTransXYScale(dx=45, dy=20, smin=0.55, smax=1.1, num_steps=71, slow_var = "X", continuous=continuous, num_images_used=71, #30000 
#                                                      images_base_dir=normalized_base_dir_INIBilder, normalized_images = numpy.arange(0, 71), 
#                                                      first_image_index=0, repetition_factor=1, seed=143)]]
sSeq_set = sTrainRTransXYScale = [[sSeqCreateRTransXYScale(iSeq_set[0][0], seed=-1)]]

#iSeq_set = iTrainRTransY2 = [[copy.deepcopy(iSeq0), iSeqCreateRTransY(900, av_fim, first_image=4500, repetition_factor=1, seed=-1)], 
#                             [copy.deepcopy(iSeq0), iSeqCreateRTransY(900, av_fim, first_image=8100, repetition_factor=1, seed=-1)],                             
#                             ]
#sSeq_set = sTrainRTransY2 = [[copy.deepcopy(sSeq0), sSeqCreateRTransY(iSeq_set[0][1], seed=-1)], 
#                             [copy.deepcopy(sSeq0), sSeqCreateRTransY(iSeq_set[4][1], seed=-1)],
#                             ]

iSeq_set = iSeenidRTransXYScale = iSeqCreateRTransXYScale(dx=45, dy=20, smin=0.55, smax=1.1, num_steps=50, slow_var = "X", continuous=True, num_images_used=25000, #20000
                                                      images_base_dir=alldbnormalized_base_dir, normalized_images = alldbnormalized_available_images, 
                                                      first_image_index=30000, pre_mirroring="none", repetition_factor=1, seed=-1)
sSeq_set = sSeenidRTransXYScale = sSeqCreateRTransXYScale(iSeq_set, seed=-1)

#WARNING, here continuous=continuous was wrong!!! we should always use the same test data!!!
iSeq_set = iNewidRTransXYScale = [[iSeqCreateRTransXYScale(dx=45, dy=20, smin=0.55, smax=1.1, num_steps=50, slow_var = "X", continuous=True, num_images_used=9000, #9000 
                                                      images_base_dir=alldbnormalized_base_dir, normalized_images = alldbnormalized_available_images, 
                                                      first_image_index=55000, pre_mirroring="none", repetition_factor=1, seed=-1)]]      #repetition_factor=4
#WARNING, code below is for display purposes only, it should be commented out!
#iSeq_set = iNewidRTransXYScale = [[iSeqCreateRTransXYScale(dx=45, dy=2, smin=0.85, smax=0.95, num_steps=50, slow_var = "X", continuous=True, num_images_used=2000, #9000 
#                                                      images_base_dir=alldbnormalized_base_dir, normalized_images = alldbnormalized_available_images, 
#                                                      first_image_index=55000, repetition_factor=1, seed=-1)]]


sSeq_set = sNewidRTransXYScale = [[sSeqCreateRTransXYScale(iSeq_set[0][0], seed=-1)]]

#This fixes classes of Newid data

print "Orig iSeenidRTransXYScale.correct_labels=", iSeenidRTransXYScale.correct_labels
print "Orig len(iSeenidRTransXYScale.correct_labels)=", len(iSeenidRTransXYScale.correct_labels)
print "Orig len(iSeenidRTransXYScale.correct_classes)=", len(iSeenidRTransXYScale.correct_classes)
all_classes = numpy.unique(iSeenidRTransXYScale.correct_classes)
print "all_classes=", all_classes
avg_labels = more_nodes.compute_average_labels_for_each_class(iSeenidRTransXYScale.correct_classes, iSeenidRTransXYScale.correct_labels)
print "avg_labels=", avg_labels 
iNewidRTransXYScale[0][0].correct_classes = more_nodes.map_labels_to_class_number(all_classes, avg_labels, iNewidRTransXYScale[0][0].correct_labels)


#quit()

ParamsRTransXYScaleFunc = SystemParameters.ParamsSystem()
ParamsRTransXYScaleFunc.name = "Function Based Data Creation for RTransXYScale"
ParamsRTransXYScaleFunc.network = linearNetwork4L #Default Network, but ignored
ParamsRTransXYScaleFunc.iTrain =iTrainRTransXYScale
ParamsRTransXYScaleFunc.sTrain = sTrainRTransXYScale

ParamsRTransXYScaleFunc.iSeenid = iSeenidRTransXYScale

ParamsRTransXYScaleFunc.sSeenid = sSeenidRTransXYScale

ParamsRTransXYScaleFunc.iNewid = iNewidRTransXYScale
ParamsRTransXYScaleFunc.sNewid = sNewidRTransXYScale

ParamsRTransXYScaleFunc.block_size = iTrainRTransXYScale[0][0].block_size
#if continuous == False:
#    ParamsRTransXYScaleFunc.train_mode = 'serial' #clustered improves final performance! mixed
#else:
#    ParamsRTransXYScaleFunc.train_mode = "window32" #clustered improves final performance! mixed
    
ParamsRTransXYScaleFunc.analysis = None
ParamsRTransXYScaleFunc.enable_reduced_image_sizes = True
ParamsRTransXYScaleFunc.reduction_factor = 4.0 # WARNING 1.0, 2.0, 4.0, 8.0
ParamsRTransXYScaleFunc.hack_image_size = 32 # WARNING   128,  64,  32 , 16
ParamsRTransXYScaleFunc.enable_hack_image_size = True

#quit()






#######################################################################################################################
#################                  AGE Extraction Experiments                             #############################
#######################################################################################################################
def find_available_images(base_dir, from_subdirs=None, verbose=False):
    """Counts how many files are in each subdirectory.
    Returns a dictionary d, with entries: d[subdir] = (num_files_in_subfolder, label, [file names])
    where the file names have a relative path w.r.t. the base directory """    

    files_dict={}
    if os.path.lexists(base_dir):
        dirList=os.listdir(base_dir)
    else:
        dirList = []
    for subdir in dirList:
        if from_subdirs == None or subdir in from_subdirs:
            subdir_full = os.path.join(base_dir, subdir)
            subdirList = os.listdir(subdir_full)
            if subdir in ["Male","Female", "Unknown"]:
                if  subdir == "Male":
                    label=1
                elif subdir == "Female":
                    label=-1
                else:
                    label = 0 
            else:
                if verbose:
                    print "Subdir: ", subdir,
                label = float(subdir)
            files_dict[subdir] = (len(subdirList), label, subdirList)
    return files_dict      
    

def cluster_images(files_dict, smallest_number_images=1400):
    """ clusters/joins the entries of files_dict, so that
        each cluster has at least smallest_number_images images. 
        The set of clusters is represented as a list of clusters [cluster1, cluster2, ...],
        and each cluster is a list of tuples [(num_files, label, files_with_subdir_dict), ...]  
        """
    subdirs = files_dict.keys()
    subdirs = sorted(subdirs, key = lambda subdir: float(subdir))
    subdirs.reverse()

    clusters = []
    cluster = []
    cluster_size = 0
    for subdir in subdirs:
        num_files_subdir, label, files_subdir = files_dict[subdir]
        files_subdir_with_subdir = []
        for file_name in files_subdir:
            files_subdir_with_subdir.append(os.path.join(subdir,file_name))
        cluster.append( (num_files_subdir, label, files_subdir_with_subdir) )
        cluster_size += num_files_subdir
        if smallest_number_images==None or cluster_size >=  smallest_number_images:
            #Save cluster
            clusters.append(cluster)
            #Start a new cluster
            cluster = []
            cluster_size=0
    if cluster_size != 0: #Something has not reached proper size, add it to lastly created cluster
        if len(clusters)>0:
            clusters[-1].extend(cluster)      
        else:
            clusters.append(cluster)
    return clusters

def cluster_to_filenames(clusters, trim_number=None, shuffle_each_cluster=True, verbose=False):
    """ Reads a cluster structure in a nested list, generates representative labels, 
    joins filenames and trims them.
    If trim_number is None, no trimming is done. 
    Otherwise clusters are of size at most trim_number
    new_clusters[0]= a cluster
    cluster = (total_files_cluster, avg_label, files_subdirs, orig_labels))
    """
    new_clusters=[]
    #print "clusters[0] is", clusters[0]
    #print "len(clusters[0]) is", len(clusters[0])

    for cluster in clusters:
        total_files_cluster = 0
        files_subdirs = []
        orig_labels = []
        sum_labels = 0
        #print "Processing cluster:", cluster
        if verbose:
            print "ClustersAdded:"
        for (num_files_subdir, label, files_subdir) in cluster:
            if verbose:
                print " With label:", label, "(%d imgs)"%num_files_subdir
            total_files_cluster += num_files_subdir
            files_subdirs.extend(files_subdir)
            orig_labels += [label]*num_files_subdir
            #TODO: handle non-float labels
            sum_labels += label*num_files_subdir
        avg_label = sum_labels / total_files_cluster
        if verbose:
            print ""

        if shuffle_each_cluster:
            selected = list(range(total_files_cluster))
            numpy.random.shuffle(selected)
            files_subdirs = [files_subdirs[i] for i in selected]
            orig_labels = [orig_labels[i] for i in selected]

        if len(files_subdirs) != len(orig_labels):
            print "Wrong cluster files and orig labels lenghts"
            print len(files_subdirs)
            print len(orig_labels)

        if trim_number != None:
            files_subdirs = files_subdirs[0:trim_number]
            orig_labels = orig_labels[0:trim_number]
            total_files_cluster = min(total_files_cluster, trim_number)
        new_clusters.append((total_files_cluster, avg_label, files_subdirs, orig_labels))

    new_clusters.reverse()
    return new_clusters


def MORPH_leave_k_identities_out(available_images_dict, k=0):
    # available_images_dict[subdir] = (num_files_in_subfolder, label, [file names])
    if k==0:
        return available_images_dict, {}
    
    subdirs = available_images_dict.keys()
    all_filenames = []
    for subdir in subdirs:
        all_filenames.extend(available_images_dict[subdir][2])
    all_identities = set()
    for filename in all_filenames:
        all_identities.add(string.split(filename, "_")[0])
    all_identities_list = list(all_identities)
    numpy.random.shuffle(all_identities_list)

    separated_identities = all_identities_list[0:k]
    print "separated_identities=", separated_identities
    
    #now create two dictionaries
    available_images_dict_orig = {}
    available_images_dict_separated = {}
    for subdir in subdirs:
        num_files_in_subfolder, label, filenames = available_images_dict[subdir]
        filenames_orig = []
        filenames_separated = []
        for file_name in filenames:
            sid = string.split(file_name, "_")[0]
            if sid in separated_identities:
                filenames_separated.append(file_name)
            else:
                filenames_orig.append(file_name)
        if len(filenames_separated)+len(filenames_orig) != num_files_in_subfolder:
            er = "Unexpected error in the sizes of the filename arrays"
            raise Exception(er)
        available_images_dict_orig[subdir]=(len(filenames_orig), label, filenames_orig)
        available_images_dict_separated[subdir]=(len(filenames_separated), label, filenames_separated)
        print "Separating %d/%d"%(len(filenames_orig), len(filenames_separated)),
    return available_images_dict_orig, available_images_dict_separated

#numpy.random.seed(111222333)
#r_age_clusters = [[21,22,23],[24,25,26],[27,28,29],[30,31],[32,33],[34,35],[36,37],[38,39],
#                [40],[41],[42],[43],[44],[45],[46],[47],[48],[49],
#                [50],[51],[52],[53],[54],[55],[56],[57],[58],[59],
#                [60],[61],[62],[63],[64],[65],[66],[67],[68],[69],
#                [70]]

#####
####r_age_all_clustered_available_images = TODO: write this part. 

def iSeqCreateRAge(dx=2, dy=2, smin=0.95, smax=1.05, delta_rotation=0.0, pre_mirroring="none", 
                   contrast_enhance=False, obj_avg_std=0.0, obj_std_min=0.20, obj_std_max=0.20, clusters=None, num_images_per_cluster_used=10, 
                   images_base_dir="wrong dir", first_image_index=0, repetition_factor=1, seed=-1, use_orig_label_as_class=False, use_orig_label=False):
    if seed >= 0 or seed == None: #also works for 
        numpy.random.seed(seed)

    if len(clusters)>0:
        min_cluster_size = clusters[0][0]
    else:
        return None
        
    for cluster in clusters:
        min_cluster_size = min(min_cluster_size, cluster[0])

    if first_image_index + num_images_per_cluster_used > min_cluster_size:
        err = "Images to be used exceeds the number of available images of at least one cluster"
        raise Exception(err) 

    print "***** Setting Information Parameters for Real Translation XYScale ******"
    iSeq = SystemParameters.ParamsInput()
    #iSeq.name = "Real Translation " + slow_var + ": (%d, %d, %d, %d) numsteps %d"%(dx, dy, smin, smax, num_steps)
    iSeq.data_base_dir = images_base_dir
    #TODO: create clusters here. 
    iSeq.ids = []
    iSeq.input_files = []
    iSeq.orig_labels = []
    for cluster in clusters:
        cluster_id = [cluster[1]] * repetition_factor * num_images_per_cluster_used #avg_label. The average label of the current cluster being considered
        iSeq.ids.extend(cluster_id)

        cluster_labels = (cluster[3][first_image_index:first_image_index+num_images_per_cluster_used])*repetition_factor #orig_label
        if len(cluster_id) != len(cluster_labels):
            print "ERROR: Wrong number of cluster labels and original labels"
            print "len(cluster_id)=", len(cluster_id)
            print "len(cluster_labels)=", len(cluster_labels)
            quit()
        iSeq.orig_labels.extend(cluster_labels)

        selected_image_filenames = (cluster[2][first_image_index:first_image_index + num_images_per_cluster_used])*repetition_factor #filenames
        iSeq.input_files.extend(selected_image_filenames)

    iSeq.num_images = len(iSeq.input_files)
    iSeq.pre_mirroring = pre_mirroring
    
    if iSeq.pre_mirroring == "none":
        iSeq.pre_mirror_flags = [False] * iSeq.num_images
    elif iSeq.pre_mirroring == "all":
        iSeq.pre_mirror_flags = [True] * iSeq.num_images
    elif iSeq.pre_mirroring == "random":
        iSeq.pre_mirror_flags = more_nodes.random_boolean_array(iSeq.num_images)
    elif iSeq.pre_mirroring == "duplicate":
        iSeq.input_files = sfa_libs.repeat_list_elements(iSeq.input_files, rep=2)
        iSeq.ids = sfa_libs.repeat_list_elements(iSeq.ids)
        iSeq.orig_labels = sfa_libs.repeat_list_elements(iSeq.orig_labels)
        tmp_pre_mirror_flags = more_nodes.random_boolean_array(iSeq.num_images) #e.g., [T, T, F, F, T, F]
        iSeq.pre_mirror_flags = numpy.array([item^val for item in tmp_pre_mirror_flags for val in (False,True)]) #e.g., [T,F, T,F, F,T, F,T, T,F, F,T])
        iSeq.num_images *= 2
        repetition_factor *= 2
    else:
        er = "Erroneous parameter iSeq.pre_mirroring=",iSeq.pre_mirroring
        raise Exception(er)
        


    #iSeq.slow_var = slow_var
    iSeq.dx = dx
    iSeq.dy = dy
    iSeq.smin = smin
    iSeq.smax = smax
    iSeq.delta_rotation = delta_rotation
    if contrast_enhance == True:
        contrast_enhance = "AgeContrastEnhancement_Avg_Std" # "PostEqualizeHistogram" # "AgeContrastEnhancement"
    if contrast_enhance in ["AgeContrastEnhancement_Avg_Std", "AgeContrastEnhancement25", "AgeContrastEnhancement20", "AgeContrastEnhancement15", "AgeContrastEnhancement", "PostEqualizeHistogram", "SmartEqualizeHistogram", False]:
        iSeq.contrast_enhance = contrast_enhance
    else:
        ex = "Contrast method unknown"
        raise Exception(ex)

    iSeq.obj_avg_std=obj_avg_std
    iSeq.obj_std_min=obj_std_min
    iSeq.obj_std_max=obj_std_max
        
#    if len(iSeq.ids) % len(iSeq.trans) != 0 and continuous == False:
#        ex="Here the number of translations/scalings must be a divisor of the number of identities"
#        raise Exception(ex)
    iSeq.ages = [None]
    iSeq.genders = [None]
    iSeq.racetweens = [None]
    iSeq.expressions = [None]
    iSeq.morphs = [None]
    iSeq.poses = [None]
    iSeq.lightings = [None]
    iSeq.slow_signal = 0 #real slow signal is the translation in the x axis (correlated to identity), added during image loading
    iSeq.step = 1
    iSeq.offset = 0

    #iSeq.params = [ids, expressions, morphs, poses, lightings]
    iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                      iSeq.morphs, iSeq.poses, iSeq.lightings]

    iSeq.block_size = num_images_per_cluster_used * repetition_factor

    iSeq.train_mode = "serial" # = "serial" "mixed", None
# None, "regular", "fwindow16", "fwindow32", "fwindow64", "fwindow128"
#        quit()
#        iSeq.train_mode = None 

    print "BLOCK SIZE =", iSeq.block_size 
    if use_orig_label_as_class == False: #Use cluster (average) labels as classes, or the true original labels
        unique_ids, iSeq.correct_classes = numpy.unique(iSeq.ids, return_inverse=True)
#        iSeq.correct_classes_from_zero = iSeq.correct_classes
#        iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(clusters)),  iSeq.block_size)
    else:
        unique_labels, iSeq.correct_classes = numpy.unique(iSeq.orig_labels, return_inverse=True)
#        iSeq.correct_classes_from_zero = iSeq.correct_classes

        
    if use_orig_label == False: #Use cluster (average) labels as labels, or the true original labels
        iSeq.correct_labels = numpy.array(iSeq.ids)
    else:
        iSeq.correct_labels = numpy.array(iSeq.orig_labels)        


    if len(iSeq.ids) != len(iSeq.orig_labels) or len(iSeq.orig_labels) != len(iSeq.input_files):
        er = "Computation of orig_labels failed:"+str(iSeq.ids)+str(iSeq.orig_labels)
        er += "len(iSeq.ids)=%d"%len(iSeq.ids) + "len(iSeq.orig_labels)=%d"%len(iSeq.orig_labels)+"len(iSeq.input_files)=%d"%len(iSeq.input_files)
        raise Exception(er)

    SystemParameters.test_object_contents(iSeq)
    return iSeq

def sSeqCreateRAge(iSeq, seed=-1, use_RGB_images=False):
    if seed >= 0 or seed == None: #also works for 
        numpy.random.seed(seed)
    #else seed <0 then, do not change seed
    
    if iSeq==None:
        print "iSeq was None, this might be an indication that the data is not available"
        sSeq = SystemParameters.ParamsDataLoading()
        return sSeq
    
    print "******** Setting Training Data Parameters for Real Age  ****************"
    sSeq = SystemParameters.ParamsDataLoading()
    sSeq.input_files = [ os.path.join(iSeq.data_base_dir, file_name) for file_name in iSeq.input_files]
    sSeq.num_images = iSeq.num_images
    sSeq.block_size = iSeq.block_size
    sSeq.train_mode = iSeq.train_mode
    sSeq.include_latest = iSeq.include_latest
    sSeq.image_width = 256
    sSeq.image_height = 260 #192
    sSeq.subimage_width = 160 #192 # 160 #128 
    sSeq.subimage_height = 160 #192 # 160 #128 
    sSeq.pre_mirror_flags = iSeq.pre_mirror_flags
    
    sSeq.trans_x_max = iSeq.dx
    sSeq.trans_x_min = -1 * iSeq.dx
    sSeq.trans_y_max = iSeq.dy
    sSeq.trans_y_min = -1 * iSeq.dy
    sSeq.min_sampling = iSeq.smin
    sSeq.max_sampling = iSeq.smax
    sSeq.delta_rotation = iSeq.delta_rotation
    sSeq.contrast_enhance = iSeq.contrast_enhance
    sSeq.obj_avg_std = iSeq.obj_avg_std
    sSeq.obj_std_min = iSeq.obj_std_min
    sSeq.obj_std_max = iSeq.obj_std_max
            
    #sSeq.subimage_pixelsampling = 2
    #sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
    sSeq.add_noise_L0 = True
    if use_RGB_images:
        sSeq.convert_format = "RGB" # "RGB", "L"
    else:
        sSeq.convert_format = "L"
    sSeq.background_type = None
    #random translation for th w coordinate
    #sSeq.translation = 20 # 25, Should be 20!!! Warning: 25
    #sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)                                                           
#    print "=>", sSeq.trans_x_min, sSeq.trans_x_max, sSeq.num_images, sSeq.translations_x
#    quit()

#    Do integer displacements make more sense? depends if respect to original image or not. Translation logic needs urgent review!!!
#    sSeq.translations_x = numpy.random.uniform(low=sSeq.trans_x_min, high=sSeq.trans_x_max, size=sSeq.num_images)
#    sSeq.translations_y = numpy.random.uniform(low=sSeq.trans_y_min, high=sSeq.trans_y_max, size=sSeq.num_images)
    #BUG0: why is this an integer offset? also frationary offsets should be supported and give acceptable results.
    
    #Continuous translations: 
#    sSeq.translations_x = numpy.random.uniform(low=sSeq.trans_x_min, high=sSeq.trans_x_max, size=sSeq.num_images)
#    sSeq.translations_y = numpy.random.uniform(low=sSeq.trans_y_min, high=sSeq.trans_y_max, size=sSeq.num_images) 
    #Or alternatively, discrete ofsets:  
    sSeq.offset_translation_x = -15.0 
    sSeq.offset_translation_y = -10.0
    sSeq.translations_x = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, sSeq.num_images) + sSeq.offset_translation_x
    sSeq.translations_y = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, sSeq.num_images) + sSeq.offset_translation_y

    print "GSP: sSeq.translations_x=", sSeq.translations_x

    sSeq.pixelsampling_x = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
    sSeq.pixelsampling_y = sSeq.pixelsampling_x + 0.0

    sSeq.rotation = numpy.random.uniform(-sSeq.delta_rotation, sSeq.delta_rotation, sSeq.num_images)
    if iSeq.obj_avg_std > 0:
        sSeq.obj_avgs = numpy.random.normal(0.0, iSeq.obj_avg_std, size=sSeq.num_images)
    else:
        sSeq.obj_avgs = numpy.zeros(sSeq.num_images)
    sSeq.obj_stds = numpy.random.uniform(sSeq.obj_std_min, sSeq.obj_std_max, sSeq.num_images)

    #BUG1: image center is not computed that way!!! also half(width-1) computation is wrong!!!
    sSeq.subimage_first_row =  sSeq.image_height/2.0-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
    sSeq.subimage_first_column = sSeq.image_width/2.0-sSeq.subimage_width*sSeq.pixelsampling_x/2.0

    sSeq.trans_sampled = True #TODO:check semantics, when is sampling/translation done? why does this value matter?
    sSeq.name = "RAge Dx in (%d, %d) Dy in (%d, %d), sampling in (%d perc, %d perc)"%(sSeq.trans_x_min, 
        sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, int(sSeq.min_sampling*100), int(sSeq.max_sampling*100))

    print "Mean in correct_labels is:", iSeq.correct_labels.mean()
    print "Var in correct_labels is:", iSeq.correct_labels.var()
    sSeq.load_data = load_data_from_sSeq
    SystemParameters.test_object_contents(sSeq)
    return sSeq


experiment_seed = os.getenv("CUICUILCO_EXPERIMENT_SEED") #1112223339 #1112223339
if experiment_seed:
    experiment_seed = int(experiment_seed)
else:
    experiment_seed = 1112223334 #111222333
    ex = "CUICUILCO_EXPERIMENT_SEED unset"
    raise Exception(ex)
print "Age estimation. experiment_seed=", experiment_seed
numpy.random.seed(experiment_seed) #seed|-5789
print "experiment_seed=", experiment_seed

age_use_RGB_images = LRec_use_RGB_images
#TODO: Repeat computation of blind levels for MORPH, FG_Net and MORPH+FGNet. (orig labels, no rep, all images)
min_cluster_size_MORPH = 60000 # 1400 # 60000
max_cluster_size_MORPH = None  # 1400 # None
age_trim_number_MORPH = 1400 # 1400
leave_k_out_MORPH = 0 #1000 #1000 #

if age_use_RGB_images:
    age_eyes_normalized_base_dir_MORPH = "/local/escalafl/Alberto/MORPH_normalizedEyesZ2_horiz_RGB_ByAge" #RGB: "/local/escalafl/Alberto/MORPH_normalizedEyesZ2_horiz_RGB_ByAge"
else:
    age_eyes_normalized_base_dir_MORPH = "/local/escalafl/Alberto/MORPH_normalizedEyesZ2_horiz_ByAge"
age_files_dict_MORPH = find_available_images(age_eyes_normalized_base_dir_MORPH, from_subdirs=None) #change from_subdirs to select a subset of all ages!

if leave_k_out_MORPH:
    age_files_dict_MORPH, age_files_dict_MORPH_out = MORPH_leave_k_identities_out(age_files_dict_MORPH, k=leave_k_out_MORPH)
    age_trim_number_MORPH = 1270 #age_trim_number_MORPH = 1270
    print "age_files_dict_MORPH_out=", age_files_dict_MORPH_out
else:
    age_files_dict_MORPH_out = {}

counter = 0
for subdir in age_files_dict_MORPH.keys():
    counter += age_files_dict_MORPH[subdir][0]
print "age_files_dict_MORPH contains %d images", counter
counter = 0
for subdir in age_files_dict_MORPH_out.keys():
    counter += age_files_dict_MORPH_out[subdir][0]
print "age_files_dict_MORPH_out contains %d images", counter
    
age_clusters_MORPH = cluster_images(age_files_dict_MORPH, smallest_number_images=age_trim_number_MORPH) #Cluster so that all clusters have size at least 1400 
age_clusters_MORPH = cluster_to_filenames(age_clusters_MORPH, trim_number=age_trim_number_MORPH)

age_clusters_MORPH_out = cluster_images(age_files_dict_MORPH_out, smallest_number_images=80000) #A single cluster for all images
age_clusters_MORPH_out = cluster_to_filenames(age_clusters_MORPH_out, trim_number=None, shuffle_each_cluster=False)

if len(age_clusters_MORPH_out) > 0:
    num_images_per_cluster_used_MORPH_out =  age_clusters_MORPH_out[0][0]
else:
    num_images_per_cluster_used_MORPH_out = 0
print "num_images_per_cluster_used_MORPH_out=", num_images_per_cluster_used_MORPH_out




min_cluster_size_FGNet = 5000 # 32 #5000
max_cluster_size_FGNet = None # 32 #None
if age_use_RGB_images:
    age_eyes_normalized_base_dir_FGNet = "/local/escalafl/Alberto/FGNet/FGNet_normalizedEyesZ2_horiz_RGB_ByAge"
else:
    age_eyes_normalized_base_dir_FGNet = "/local/escalafl/Alberto/FGNet/FGNet_normalizedEyesZ2_horiz_ByAge"
subdirs_FGNet=None
subdirs_FGNet=["%d"%i for i in range(16, 77)] #70 77
age_files_dict_FGNet = find_available_images(age_eyes_normalized_base_dir_FGNet, from_subdirs=subdirs_FGNet) #change from_subdirs to select a subset of all ages!
age_clusters_FGNet = cluster_images(age_files_dict_FGNet, smallest_number_images=min_cluster_size_FGNet) #Cluster so that all clusters have size at least 1400 
#print "******************"
#print len(age_clusters_FGNet), age_clusters_FGNet[0], ":)"
#print "******************"
age_clusters_FGNet = cluster_to_filenames(age_clusters_FGNet, trim_number=max_cluster_size_FGNet)

if len(age_clusters_FGNet) > 0:
    num_images_per_cluster_used_FGNet =  age_clusters_FGNet[0][0]
else:
    num_images_per_cluster_used_FGNet = 0
#print "******************"
#print len(age_clusters_FGNet), age_clusters_FGNet[0], ":)"
#print "******************"
print "num_images_per_cluster_used_FGNet=", num_images_per_cluster_used_FGNet
#quit()

age_trim_number_INIBilder = None
if age_use_RGB_images:
    age_eyes_normalized_base_dir_INIBilder = "/local/escalafl/Alberto/INIBilder/INIBilder_normalizedEyesZ2_horiz_RGB"
else:
    age_eyes_normalized_base_dir_INIBilder = "/local/escalafl/Alberto/INIBilder/INIBilder_normalizedEyesZ2_horiz"
age_files_dict_INIBilder = find_available_images(age_eyes_normalized_base_dir_INIBilder, from_subdirs=None) #change from_subdirs to select a subset of all ages!
age_clusters_INIBilder = cluster_images(age_files_dict_INIBilder, smallest_number_images=age_trim_number_INIBilder) #Cluster so that all clusters have size at least 1400 
age_clusters_INIBilder = cluster_to_filenames(age_clusters_INIBilder, trim_number=age_trim_number_INIBilder, shuffle_each_cluster=False) 
if len(age_clusters_INIBilder) > 0:
    num_images_per_cluster_used_INIBilder =  age_clusters_INIBilder[0][0]
else:
    num_images_per_cluster_used_INIBilder = 0
print "num_images_per_cluster_used_INIBilder=", num_images_per_cluster_used_INIBilder

age_trim_number_MORPH_FGNet = 1400
if age_use_RGB_images:
    age_eyes_normalized_base_dir_MORPH_FGNet = "/local/escalafl/Alberto/MORPH_FGNet_normalizedEyesZ2_horiz_ByAge"
else:
    age_eyes_normalized_base_dir_MORPH_FGNet = "/local/escalafl/Alberto/MORPH_FGNet_normalizedEyesZ2_horiz"
age_files_dict_MORPH_FGNet = find_available_images(age_eyes_normalized_base_dir_MORPH_FGNet, from_subdirs=None) #change from_subdirs to select a subset of all ages!
age_clusters_MORPH_FGNet = cluster_images(age_files_dict_MORPH_FGNet, smallest_number_images=age_trim_number_MORPH_FGNet) #Cluster so that all clusters have size at least 1400 
age_clusters_MORPH_FGNet = cluster_to_filenames(age_clusters_MORPH_FGNet, trim_number=age_trim_number_MORPH_FGNet)
if len(age_clusters_MORPH_FGNet) > 0:
    num_images_per_cluster_used_MORPH_FGNet =  age_clusters_MORPH_FGNet[0][0]
else:
    num_images_per_cluster_used_MORPH_FGNet = 0
print "num_images_per_cluster_used_MORPH_FGNet=", num_images_per_cluster_used_MORPH_FGNet


age_clusters = age_clusters_MORPH  #_MORPH #_FGNet
age_eyes_normalized_base_dir = age_eyes_normalized_base_dir_MORPH #_MORPH #_FGNet

verbose=False or True
if verbose:
    print "num clusters =", len(age_clusters)
    for age_cluster in age_clusters:
        print "avg_label=%f, num_images=%d"%(age_cluster[1], age_cluster[0]), 
        print "filenames[0]=", age_cluster[2][0], 
        print "orig_labels[0]=", age_cluster[3][0],
        print "filenames[-1]=", age_cluster[2][-1],
        print "orig_labels[0]=", age_cluster[3][-1]
#quit()
if leave_k_out_MORPH == 1000 and len(age_clusters) != 30:
    er = "leave_k_out_MORPH is changing the number of clusters (%d clusters instead of 30)"%len(age_clusters)
    raise Exception(er)

use_seenid_classes_to_generate_knownid_and_seenid_classes = True

#0.825 #0.55 # 1.1
#smin=0.575, smax=0.625 (orig images) 
# (zoomed images)   
# smin=1.25, smax=1.40, 1.325
base_scale = 1.14
factor_training = 1.03573 # 1.032157 #1.0393 # 1.03573
factor_seenid = 1.01989 #1.021879 #1.018943  #1.020885         # 1.01989  
scale_offset = 0.00 #0.08 0.04  
#DEFINITIVE:
#128x128: iSeq_set = iTrainRAge = [[iSeqCreateRAge(dx=0.0, dy=0.0, smin=1.275, smax=1.375, delta_rotation=3.0, pre_mirroring="none", contrast_enhance=True, 
#160x160:
iSeq_set = iTrainRAge = [[iSeqCreateRAge(dx=0.0, dy=0.0, smin=(base_scale+scale_offset) / factor_training, smax=(base_scale+scale_offset) * factor_training, delta_rotation=3.0, pre_mirroring="none", contrast_enhance=True, 
#-0.05
#192x192: iSeq_set = iTrainRAge = [[iSeqCreateRAge(dx=0.0, dy=0.0, smin=0.85+scale_offset, smax=0.91666+scale_offset, delta_rotation=3.0, pre_mirroring="none", contrast_enhance=True, 
                                         obj_avg_std=0.00, obj_std_min=0.20, obj_std_max=0.20, clusters=age_clusters, num_images_per_cluster_used=1000,  #1000 #1000, 900=>27000
                                         images_base_dir=age_eyes_normalized_base_dir, first_image_index=0, repetition_factor=8, seed=-1, use_orig_label_as_class=False, use_orig_label=True)]] #repetition_factor=6 or at least 4 #T: dx=2, dy=2, smin=1.25, smax=1.40, repetition_factor=5
#Experimental:
#iSeq_set = iTrainRAge = [[iSeqCreateRAge(dx=0, dy=0, smin=1.0, smax=1.0, delta_rotation=0.0, pre_mirroring="none", contrast_enhance=True, 
#                                         obj_avg_std=0.00, obj_std_min=0.20, obj_std_max=0.20, clusters=age_clusters, num_images_per_cluster_used=1000,  #1000 #1000, 900=>27000
#                                         images_base_dir=age_eyes_normalized_base_dir, first_image_index=0, repetition_factor=1, seed=-1, use_orig_label=True)]] #repetition_factor=6 or at least 4 #T: dx=2, dy=2, smin=1.25, smax=1.40, repetition_factor=5
sSeq_set = sTrainRAge = [[sSeqCreateRAge(iSeq_set[0][0], seed=-1, use_RGB_images=age_use_RGB_images)]]



#MORPH+FGNet
#iSeq_set = iTrainRAge = [[iSeqCreateRAge(dx=2, dy=2, smin=1.25, smax=1.40, clusters=age_clusters, num_images_per_cluster_used=num_images_per_cluster_used_MORPH_FGNet,  #1000 =>30000, 900=>27000
#                                         images_base_dir=age_eyes_normalized_base_dir, first_image_index=0, repetition_factor=5, seed=-1, use_orig_label=False)]] #rep=5
#sSeq_set = sTrainRAge = [[sSeqCreateRAge(iSeq_set[0][0], seed=-1)]] 
#smin=0.595, smax=0.605 (orig images)
#128x128: iSeq_set = iSeenidRAge = iSeqCreateRAge(dx=0.0, dy=0.0, smin=1.3, smax=1.35, delta_rotation=1.5, pre_mirroring="none", contrast_enhance=True, 
#160x160: 
iSeq_set = iSeenidRAge = iSeqCreateRAge(dx=0.0, dy=0.0, smin=(base_scale+scale_offset) / factor_seenid, smax=(base_scale+scale_offset) * factor_seenid, delta_rotation=0.65, pre_mirroring="none", contrast_enhance=True, 
#192x192:iSeq_set = iSeenidRAge = iSeqCreateRAge(dx=0.0, dy=0.0, smin=0.86667+scale_offset, smax=0.9+scale_offset, delta_rotation=1.5, pre_mirroring="none", contrast_enhance=True, 
                                        obj_avg_std=0.00, obj_std_min=0.20, obj_std_max=0.20,clusters=age_clusters, num_images_per_cluster_used=200,   #200 #300=>9000
                                        images_base_dir=age_eyes_normalized_base_dir, first_image_index=1000, repetition_factor=8, seed=-1, 
                                        use_orig_label_as_class=use_seenid_classes_to_generate_knownid_and_seenid_classes, use_orig_label=True) #repetition_factor=4 #T=repetition_factor=3
sSeq_set = sSeenidRAge = sSeqCreateRAge(iSeq_set, seed=-1, use_RGB_images=age_use_RGB_images)
###Testing Original MORPH:
#128x128: iSeq_set = iNewidRAge = [[iSeqCreateRAge(dx=0, dy=0, smin=1.325, smax=1.326, delta_rotation=0.0, pre_mirroring="none", contrast_enhance=True, 
#192x192: 
#iSeq_set = iNewidRAge = [[iSeqCreateRAge(dx=0, dy=0, smin=0.8833+scale_offset, smax=0.8833+scale_offset, delta_rotation=0.0, pre_mirroring="none", contrast_enhance=True, 
#160x160: 
#TODO:get rid of this conditional. Add Leave-k-out for MORPH+FGNet

if leave_k_out_MORPH==0:
    iSeq_set = iNewidRAge = [[iSeqCreateRAge(dx=0, dy=0, smin=(base_scale+scale_offset), smax=(base_scale+scale_offset), delta_rotation=0.0, pre_mirroring="none", contrast_enhance=True, 
                                             obj_avg_std=0.0, obj_std_min=0.20, obj_std_max=0.20, clusters=age_clusters, num_images_per_cluster_used=200,   #200=>6000
                                             images_base_dir=age_eyes_normalized_base_dir, first_image_index=1200, repetition_factor=1, seed=-1, use_orig_label_as_class=False, use_orig_label=True)]]
else:
    iSeq_set = iNewidRAge = [[iSeqCreateRAge(dx=0, dy=0, smin=(base_scale+scale_offset), smax=(base_scale+scale_offset), delta_rotation=0.0, pre_mirroring="none", contrast_enhance=True, 
                                             obj_avg_std=0.0, obj_std_min=0.20, obj_std_max=0.20, clusters=age_clusters_MORPH_out, num_images_per_cluster_used=num_images_per_cluster_used_MORPH_out,   #200=>6000
                                             images_base_dir=age_eyes_normalized_base_dir, first_image_index=0, repetition_factor=1, seed=-1, use_orig_label_as_class=False, use_orig_label=True)]]
sSeq_set = sNewidRAge = [[sSeqCreateRAge(iSeq_set[0][0], seed=-1, use_RGB_images=age_use_RGB_images)]]


#iSeenidRAge = iTrainRAge[0][0]
#sSeenidRAge = sTrainRAge[0][0]
#iNewidRAge = iTrainRAge
#sNewidRAge = sTrainRAge
#Testing with INI Bilder:
#iSeq_set = iNewidRAge = [[iSeqCreateRAge(dx=0, dy=0, smin=(base_scale+scale_offset), smax=(base_scale+scale_offset), delta_rotation=0.0, pre_mirroring="duplicate", contrast_enhance=True,
#                                         obj_avg_std=0.0, obj_std_min=0.20, obj_std_max=0.20, clusters=age_clusters_INIBilder, num_images_per_cluster_used=num_images_per_cluster_used_INIBilder,   
#                                         images_base_dir=age_eyes_normalized_base_dir_INIBilder, first_image_index=0, repetition_factor=1, seed=-1, use_orig_label_as_class=False, use_orig_label=True)]]
#sSeq_set = sNewidRAge = [[sSeqCreateRAge(iSeq_set[0][0], seed=-1, use_RGB_images=age_use_RGB_images)]]
#Testing with FGNet:
#iSeq_set = iNewidRAge = [[iSeqCreateRAge(dx=0, dy=0, smin=(base_scale+scale_offset), smax=(base_scale+scale_offset), delta_rotation=0.0, pre_mirroring="none", contrast_enhance=True,
#                                         obj_avg_std=0.0, obj_std_min=0.20, obj_std_max=0.20, clusters=age_clusters_FGNet, num_images_per_cluster_used=num_images_per_cluster_used_FGNet,  
#                                         images_base_dir=age_eyes_normalized_base_dir_FGNet, first_image_index=0, repetition_factor=1, seed=-1, use_orig_label_as_class=False, use_orig_label=True)]]
#sSeq_set = sNewidRAge = [[sSeqCreateRAge(iSeq_set[0][0], seed=-1, use_RGB_images=age_use_RGB_images)]]
#quit()

ParamsRAgeFunc = SystemParameters.ParamsSystem()
ParamsRAgeFunc.name = "Function Based Data Creation for RAge"
ParamsRAgeFunc.network = linearNetwork4L #Default Network, but ignored
ParamsRAgeFunc.iTrain =iTrainRAge
ParamsRAgeFunc.sTrain = sTrainRAge

ParamsRAgeFunc.iSeenid = iSeenidRAge
ParamsRAgeFunc.sSeenid = sSeenidRAge

ParamsRAgeFunc.iNewid = iNewidRAge
ParamsRAgeFunc.sNewid = sNewidRAge

if iTrainRAge != None and iTrainRAge[0][0]!=None:
    ParamsRAgeFunc.block_size = iTrainRAge[0][0].block_size
ParamsRAgeFunc.train_mode = "Weird Mode" #Ignored for the moment 
ParamsRAgeFunc.analysis = None
ParamsRAgeFunc.enable_reduced_image_sizes = True
ParamsRAgeFunc.reduction_factor = 2.0 # T=2.0 WARNING 1.0, 2.0, 4.0, 8.0
ParamsRAgeFunc.hack_image_size = 80 # T=64 WARNING  96, 80, 128,  64,  32 , 16
ParamsRAgeFunc.enable_hack_image_size = True
ParamsRAgeFunc.patch_network_for_RGB = False #

#print sTrainRAge[0][0].translations_x
#print sSeenidRAge.translations_x
#print sNewidRAge[0][0].translations_x
#quit()

all_classes = numpy.unique(iSeenidRAge.correct_classes)
smallest_number_of_samples_per_class = 30 #14
current_count = 0
current_class = 0
class_list = []
for classnr in all_classes: #[::-1]:
    class_list.append(current_class)
    current_count += (iSeenidRAge.correct_classes==classnr).sum()    
    if current_count >= smallest_number_of_samples_per_class:
        current_count = 0
        current_class += 1
if current_count >= smallest_number_of_samples_per_class:
    print "fine"
elif current_count > 0:
    print "fixing"
    class_list = numpy.array(class_list)
    class_list[class_list == current_class] = current_class-1
#class_list.reverse()
#class_list = numpy.array(class_list)
#class_list = class_list.max()-class_list

print "class_list=", class_list
for i, classnr in enumerate(all_classes):
    iSeenidRAge.correct_classes[iSeenidRAge.correct_classes==classnr] = class_list[i]
        

if use_seenid_classes_to_generate_knownid_and_seenid_classes:
    print "Orig iSeenidRTransXYScale.correct_labels=", iSeenidRAge.correct_labels
    print "Orig len(iSeenidRAge.correct_labels)=", len(iSeenidRAge.correct_labels)
    print "Orig len(iSeenidRAge.correct_classes)=", len(iSeenidRAge.correct_classes)
    all_classes = numpy.unique(iSeenidRAge.correct_classes)
    print "all_classes=", all_classes
    avg_labels = more_nodes.compute_average_labels_for_each_class(iSeenidRAge.correct_classes, iSeenidRAge.correct_labels)
    print "avg_labels=", avg_labels 
    iTrainRAge[0][0].correct_classes = more_nodes.map_labels_to_class_number(all_classes, avg_labels, iTrainRAge[0][0].correct_labels)
    iNewidRAge[0][0].correct_classes = more_nodes.map_labels_to_class_number(all_classes, avg_labels, iNewidRAge[0][0].correct_labels)

    for classnr in all_classes:
        print "class %d appears %d times"%(classnr, (iSeenidRAge.correct_classes==classnr).sum())
    #quit()




import os
import time

minutes_sleep=0
time.sleep(minutes_sleep*60)

#nums_feats = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
#nums_feats = [24, 25, 26, 27, 28, 29, 30, 31, 32]
#nums_feats = [4, 5, 10, 15, 20, 25, 30, 31, 32]
nums_feats = [3,]
#enable_ncc = 0
#enable_svm = 0
seeds = [31337311, 31337312, 31337313, 31337314, 31337315, 31337316, 31337318,31337319, 31337320, 31337321,]
#seeds = [31337311,]
#net_numbers = [31,]
#net_numbers = [32,33,34,35,36,37,38,39,40]
#seeds = [31337311, 31337312, 31337313, 31337314, 31337315, 31337316, 31337318,31337319, 31337320, 31337321,]
#net_numbers = [31,32,33,34,35,36,37,38,39,40] #compact+5
#net_numbers = [21,22,23,24,25,26,27,28,29,30] #compact+31 
#net_numbers = [13,14,15,16,17,18,19,20] #clustered, seeds 31337313 to 31337321
net_numbers = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1] 
#net_numbers = [13,14,15,16,17,18,19,20]
#seeds = [31337314, 31337315, 31337316, 31337318, 31337319]
#seeds = [31337318,31337319, 31337320, 31337321,]
parameters = [30,]
#seeds = [123,]
#parameters = [300]



#svm_cs = [2.0]
#svm_gammas = [0.125,]
#enable_lr=1
#network_number=170
network_desc = "clustered_r" #compact+3 clustered # warning, update GenerateSystemParameters.py and GSFA_node.py

command =  """time nice -n 19 python -u cuicuilco_run_GTSRB.py --EnableDisplay=0 --CacheAvailable=1 --NetworkCacheReadDir=/local/tmp/escalafl/Alberto/SavedNetworks --NetworkCacheWriteDir=None --NodeCacheReadDir=/local/tmp/escalafl/Alberto/SavedNodes --NodeCacheWriteDir=/local/tmp/escalafl/Alberto/SavedNodes --SaveSubimagesTraining=0 --SaveAverageSubimageTraining=0 --NumFeaturesSup=%d --SaveSorted_AE_GaussNewid=0 --SaveSortedIncorrectClassGaussNewid=0 --ComputeSlowFeaturesNewidAcrossNet=0 --UseFilter=0 --EnableGC=1 --EnableKNN=0 --kNN_k=5 --EnableNCC=1 --EnableSVM=0 --SVM_C=0.125 --SVM_gamma=1.0 --EnableLR=0 --AskNetworkLoading=0 --LoadNetworkNumber=%d --NParallel=2 --EnableScheduler=0 --EstimateExplainedVarWithInverse=0 --EstimateExplainedVarWithKNN_k=0 --EstimateExplainedVarWithKNNLinApp=0 --EstimateExplainedVarLinGlobal_N=0 --AddNormalizationNode=0 --MakeLastPCANodeWhithening=0 --FeatureCutOffLevel=0.0 --ExportDataToLibsvm=0 --IntegerLabelEstimation=0 --CumulativeScores=0 --SleepM=0 2>&1 > ELLArticle_GTSRB32C/GTSRB_48x48_32signs_PCA120_QE_%s_S%d_shuffle_feats%d.txt"""

#command = """time python cuicuilco_run.py --EnableDisplay=0 --CacheAvailable=0 --NetworkCacheReadDir=/local/tmp/escalafl/Alberto/SavedNetworks --NetworkCacheWriteDir=None --NodeCacheReadDir=/local/tmp/escalafl/Alberto/SavedNodes --NodeCacheWriteDir=/local/tmp/escalafl/Alberto/SavedNodes --SaveSubimagesTraining=0 --SaveAverageSubimageTraining=0 --NumFeaturesSup=%d --SaveSorted_AE_GaussNewid=0 --SaveSortedIncorrectClassGaussNewid=0 --ComputeSlowFeaturesNewidAcrossNet=0 --UseFilter=0 --EnableKNN=0 --kNN_k=5 --EnableNCC=0 --EnableSVM=0 --SVM_C=0.125 --SVM_gamma=1.0 --EnableLR=0 --AskNetworkLoading=0 --LoadNetworkNumber=-1 --NParallel=2 --EnableScheduler=0 --EstimateExplainedVarWithInverse=0 --EstimateExplainedVarWithKNN_k=0 --EstimateExplainedVarWithKNNLinApp=0 --EstimateExplainedVarLinGlobal_N=-1 --AddNormalizationNode=0 --MakeLastPCANodeWhithening=0 --FeatureCutOffLevel=0.0 --ExportDataToLibsvm=0 --IntegerLabelEstimation=0 --CumulativeScores=0 --SleepM=0 > MNIST/MNISTExp_24x24_%s_Feats%d_P%d_S%d_.txt"""

#command = """time python cuicuilco_run.py --EnableDisplay=0 --CacheAvailable=1 --NetworkCacheReadDir=/local/tmp/escalafl/Alberto/SavedNetworks --NetworkCacheWriteDir=None --NodeCacheReadDir=/local/tmp/escalafl/Alberto/SavedNodes --NodeCacheWriteDir=/local/tmp/escalafl/Alberto/SavedNodes --SaveSubimagesTraining=0 --SaveAverageSubimageTraining=0 --NumFeaturesSup=%d --SaveSorted_AE_GaussNewid=0 --SaveSortedIncorrectClassGaussNewid=0 --ComputeSlowFeaturesNewidAcrossNet=0 --UseFilter=0 --EnableKNN=0 --kNN_k=5 --EnableNCC=%d --EnableSVM=%d --SVM_C=%f --SVM_gamma=%f --EnableLR=%d --AskNetworkLoading=0 --LoadNetworkNumber=%d --NParallel=2 --EnableScheduler=0 --EstimateExplainedVarWithInverse=0 --EstimateExplainedVarWithKNN_k=0 --EstimateExplainedVarWithKNNLinApp=0 --EstimateExplainedVarLinGlobal_N=0 --AddNormalizationNode=0 --MakeLastPCANodeWhithening=0 --FeatureCutOffLevel=0.0 --ExportDataToLibsvm=0 > tune_%s/PosX_ncc%d_svm%d_svmC%f_svmG%f_lr%d_net%d_%03dF.txt"""

#feats_cs_gammas = [(20, 0.5, 0.5), (21, 0.5, 0.5), (22, 0.5, 0.5), (19, 1.0, 0.5),(19, 0.5, 0.25),(19, 0.25, 0.5)]
#feats_cs_gammas = [(15,4,0.125), (15,2,0.25), (15,1,0.125), (15,2,0.0625), ]
#feats_cs_gammas = [(15,8,0.125), (15,4,0.25), (15,2,0.125), (15,16,0.125)]
#feats_cs_gammas = [(15,8,0.5), (15,2,0.5)]
#feats_cs_gammas = [(15,2,1),(15,4,1),(15,4,0.5),(15,8,0.25)]
#feats_cs_gammas = [(15,1,2),(15,2,2)]
#feats_cs_gammas = [(15,0.5,2), (15,0.5,1), (15,1,0.5)]
#feats_cs_gammas = [(14,1,1),(16,1,1)]
#feats_cs_gammas = [(4,0,0), (6,0,0), (8,0,0)]
#feats_cs_gammas = [(3,0,0), (5,0,0), (10,0,0)]
#feats_cs_gammas = [(50,0,0), (59,0,0), (60,0,0)]

for parameter in parameters:
    os.putenv("CUICUILCO_TUNING_PARAMETER",str(parameter))
    print "Setting CUICUILCO_TUNING_PARAMETER=", str(parameter)
    for net, seed in enumerate(seeds):
        os.putenv("CUICUILCO_EXPERIMENT_SEED",str(seed))
        network = net_numbers[net]
        print "Setting CUICUILCO_EXPERIMENT_SEED=", str(seed), "using trained network", network
        for num_feats in nums_feats:
            cmd = command%(num_feats, network, network_desc, seed, num_feats)
            print "excecuting: ", cmd
            os.system(cmd)


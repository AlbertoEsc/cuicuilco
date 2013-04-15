import os

seeds = [1112223335]

gc_num_feats = 60
 
network_numbers = [-1]
network_desc = "mixed"

for i, seed in enumerate(seeds):
    network_number = network_numbers[i]
    os.putenv("POSX_SEED",str(seed))
   
    enable_ncc=0
    enable_svm=0 
    svm_c = 0
    svm_gamma = 0
    enable_lr=1
    num_feats = gc_num_feats
    command2 = """time python cuicuilco_run.py --EnableDisplay=0 --CacheAvailable=1 --NetworkCacheReadDir=/local/tmp/escalafl/Alberto/SavedNetworks --NetworkCacheWriteDir=None --NodeCacheReadDir=/local/tmp/escalafl/Alberto/SavedNodes --NodeCacheWriteDir=/local/tmp/escalafl/Alberto/SavedNodes --SaveSubimagesTraining=0 --SaveAverageSubimageTraining=0 --NumFeaturesSup=%d --SaveSorted_AE_GaussNewid=0 --SaveSortedIncorrectClassGaussNewid=0 --ComputeSlowFeaturesNewidAcrossNet=0 --UseFilter=0 --EnableKNN=0 --kNN_k=5 --EnableNCC=%d --EnableSVM=%d --SVM_C=%f --SVM_gamma=%f --EnableLR=%d --AskNetworkLoading=0 --LoadNetworkNumber=%d --NParallel=2 --EnableScheduler=0 --EstimateExplainedVarWithInverse=0 --EstimateExplainedVarWithKNN_k=0 --EstimateExplainedVarWithKNNLinApp=0 --EstimateExplainedVarLinGlobal_N=0 --AddNormalizationNode=0 --MakeLastPCANodeWhithening=0 --FeatureCutOffLevel=0.0 --ExportDataToLibsvm=0 > tune_%s/PosX_ncc%d_svm%d_svmC%f_svmG%f_lr%d_net%d_%03dF_%dS_ver22.txt"""
    cmd = command2%(num_feats, enable_ncc, enable_svm, svm_c, svm_gamma, enable_lr, network_number, network_desc, enable_ncc, enable_svm, svm_c, svm_gamma, enable_lr, network_number, num_feats, seed)
    print "excecuting: ", cmd
    os.system(cmd)


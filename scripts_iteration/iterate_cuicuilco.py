import os

nums_feats = [20]
enable_ncc = 0
enable_svm = 1
svm_cs = [2.0]
svm_gammas = [0.125,]
enable_lr=0
network_number=175
network_desc = "fw16"


command = """time python cuicuilco_run.py --EnableDisplay=0 --CacheAvailable=1 --NetworkCacheReadDir=/local/tmp/escalafl/Alberto/SavedNetworks --NetworkCacheWriteDir=None --NodeCacheReadDir=/local/tmp/escalafl/Alberto/SavedNodes --NodeCacheWriteDir=/local/tmp/escalafl/Alberto/SavedNodes --SaveSubimagesTraining=0 --SaveAverageSubimageTraining=0 --NumFeaturesSup=%d --SaveSorted_AE_GaussNewid=0 --SaveSortedIncorrectClassGaussNewid=0 --ComputeSlowFeaturesNewidAcrossNet=0 --UseFilter=0 --EnableKNN=0 --kNN_k=5 --EnableNCC=%d --EnableSVM=%d --SVM_C=%f --SVM_gamma=%f --EnableLR=%d --AskNetworkLoading=0 --LoadNetworkNumber=%d --NParallel=2 --EnableScheduler=0 --EstimateExplainedVarWithInverse=0 --EstimateExplainedVarWithKNN_k=0 --EstimateExplainedVarWithKNNLinApp=0 --EstimateExplainedVarLinGlobal_N=0 --AddNormalizationNode=0 --MakeLastPCANodeWhithening=0 --FeatureCutOffLevel=0.0 --ExportDataToLibsvm=0 > tune_%s/PosX_ncc%d_svm%d_svmC%f_svmG%f_lr%d_net%d_%03dF.txt"""

feats_cs_gammas = [(17, 0.5, 0.25), (17, 0.5, 1.0), (17, 0.25, 1.0)]

for num_feats,svm_c,svm_gamma in feats_cs_gammas:
    cmd = command%(num_feats, enable_ncc, enable_svm, svm_c, svm_gamma, enable_lr, network_number, network_desc, enable_ncc, enable_svm, svm_c, svm_gamma, enable_lr, network_number, num_feats)
    print "excecuting: ", cmd
    os.system(cmd)

#for num_feats in nums_feats:
#    for svm_c in svm_cs:
#        for svm_gamma in svm_gammas:
#            cmd = command%(num_feats, enable_ncc, enable_svm, svm_c, svm_gamma, enable_lr, network_number, network_desc, enable_ncc, enable_svm, svm_c, svm_gamma, enable_lr, network_number, num_feats)
#            print "excecuting: ", cmd
#            os.system(cmd)

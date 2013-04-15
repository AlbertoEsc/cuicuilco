import os

nums_feats = [20]
enable_ncc = 1
enable_svm = 0
svm_cs = [2.0]
svm_gammas = [0.125,]
enable_lr=1
network_number=170
network_desc = "serial"


command = """time python cuicuilco_run.py --EnableDisplay=0 --CacheAvailable=1 --NetworkCacheReadDir=/local/tmp/escalafl/Alberto/SavedNetworks --NetworkCacheWriteDir=None --NodeCacheReadDir=/local/tmp/escalafl/Alberto/SavedNodes --NodeCacheWriteDir=/local/tmp/escalafl/Alberto/SavedNodes --SaveSubimagesTraining=0 --SaveAverageSubimageTraining=0 --NumFeaturesSup=%d --SaveSorted_AE_GaussNewid=0 --SaveSortedIncorrectClassGaussNewid=0 --ComputeSlowFeaturesNewidAcrossNet=0 --UseFilter=0 --EnableKNN=0 --kNN_k=5 --EnableNCC=%d --EnableSVM=%d --SVM_C=%f --SVM_gamma=%f --EnableLR=%d --AskNetworkLoading=0 --LoadNetworkNumber=%d --NParallel=2 --EnableScheduler=0 --EstimateExplainedVarWithInverse=0 --EstimateExplainedVarWithKNN_k=0 --EstimateExplainedVarWithKNNLinApp=0 --EstimateExplainedVarLinGlobal_N=0 --AddNormalizationNode=0 --MakeLastPCANodeWhithening=0 --FeatureCutOffLevel=0.0 --ExportDataToLibsvm=0 > tune_%s/PosX_ncc%d_svm%d_svmC%f_svmG%f_lr%d_net%d_%03dF.txt"""

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
feats_cs_gammas = [(50,0,0), (59,0,0), (60,0,0)]

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

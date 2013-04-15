import os

nums_feats = [20]
enable_ncc = 1
enable_svm = 0
svm_cs = [2.0]
svm_gammas = [0.125,]
enable_lr=1
network_number=171
network_desc = "pca"


command = """time python cuicuilco_run.py --EnableDisplay=0 --CacheAvailable=1 --NetworkCacheReadDir=/local/tmp/escalafl/Alberto/SavedNetworks --NetworkCacheWriteDir=None --NodeCacheReadDir=/local/tmp/escalafl/Alberto/SavedNodes --NodeCacheWriteDir=/local/tmp/escalafl/Alberto/SavedNodes --SaveSubimagesTraining=0 --SaveAverageSubimageTraining=0 --NumFeaturesSup=%d --SaveSorted_AE_GaussNewid=0 --SaveSortedIncorrectClassGaussNewid=0 --ComputeSlowFeaturesNewidAcrossNet=0 --UseFilter=0 --EnableKNN=0 --kNN_k=5 --EnableNCC=%d --EnableSVM=%d --SVM_C=%f --SVM_gamma=%f --EnableLR=%d --AskNetworkLoading=0 --LoadNetworkNumber=%d --NParallel=2 --EnableScheduler=0 --EstimateExplainedVarWithInverse=0 --EstimateExplainedVarWithKNN_k=0 --EstimateExplainedVarWithKNNLinApp=0 --EstimateExplainedVarLinGlobal_N=0 --AddNormalizationNode=0 --MakeLastPCANodeWhithening=0 --FeatureCutOffLevel=0.0 --ExportDataToLibsvm=0 > tune_%s/PosX_ncc%d_svm%d_svmC%f_svmG%f_lr%d_net%d_%03dF.txt"""

#feats_cs_gammas = [(80,32,0.125),(90,32,0.125),(100,5.6568,0.17677),(110,5.6568,0.17677),(120,5.6568,0.17677)]  #From classif
#feats_cs_gammas = [(50,16,0.125), (50,32,0.25), (50, 64, 0.125), (50, 64, 0.0625)] 
#feats_cs_gammas = [(50, 64.0, 0.25), (48,32,0.25), (52,32,0.25)]
#feats_cs_gammas = [(50, 64.0, 0.5), (50, 32.0, 0.5)]
#feats_cs_gammas = [(50,32,1),(50,64,1),(50,128,1),(50,128,0.5)]
#feats_cs_gammas = [(50,64,2), (50,128,2), (50,64,4)]
#feats_cs_gammas = [(17, 0.5, 0.5), (16, 0.5, 0.5), (18, 0.5, 0.5), (19, 0.5, 0.5),   (15, 2.0, 0.125), (14, 2.0, 0.125), (16, 2.0, 0.125), (17, 2.0, 0.125)] #from fw16 and serial
#feats_cs_gammas = [(19, 2.0, 0.125), (18, 2.0, 0.125), (17, 2.0, 0.125), (20, 2.0, 0.125),   (24,0.5,0.5), (23,0.5,0.5), (26,0.5,0.5)] #from mixed and refine
#feats_cs_gammas = [(20, 0.5, 0.5), (21, 0.5, 0.5), (22, 0.5, 0.5), (19, 1.0, 0.5),(19, 0.5, 0.25),(19, 0.25, 0.5)]
#feats_cs_gammas = [(50,64,2), (50,128,2), (50,64,4)]
#feats_cs_gammas = [(49,64,1),(51,64,1)]
#feats_cs_gammas = [(30,0,0), (40,0,0),(50,0,0),(60,0,0)]
#feats_cs_gammas = [(59,0,0), (60,0,0)]
#feats_cs_gammas = [(46,0,0), (48,0,0), (52,0,0), (54,0,0),]
#feats_cs_gammas = [(55,0,0), (53,0,0), (70,0,0), (80,0,0),]
#feats_cs_gammas = [(68,0,0), (72,0,0), (110,0,0), (119,0,0),(120,0,0)]
feats_cs_gammas = [(118,0,0), (117,0,0),(115,0,0)]

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

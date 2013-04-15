import numpy
import mdp
import mdp.parallel as parallel
from mdp.parallel import pp_support
import sys
sys.path.append("/home/escalafl/workspace/cuicuilco/src")
import patch_mdp
import time

nodes = (mdp.nodes.SFANode, mdp.nodes.PCANode)
#size_var_set = (10, 20, 40, 80, 160) 
input_dim_set = (500,) 
#num_samples_set = (1000, 2000, 4000, 8000, 16000, 32000, 64000) 
num_samples_set = (60000,) 
output_dim_set = (20,)
block_sizes_set = (100, )
#num_reps = 40
#num_reps_inv = 20
num_reps= 1
num_reps_inv= 1
n_parallel = 8
scheduler = pp_support.LocalPPScheduler(ncpus=n_parallel, max_queue_length=0, verbose=True)

#scheduler = mdp.parallel.ThreadScheduler(n_threads=n_parallel)

##FIRST EXPERIMENT... SFA
#print "Benchmarking regular/parallel sfa_node"
#for input_dim in input_dim_set:
#    for num_samples in num_samples_set:
#            x_train = numpy.random.random((num_samples,input_dim))
#            x_exec = numpy.random.random((num_samples,input_dim))
#            for output_dim in output_dim_set:
#                for block_size in block_sizes_set:
#                    for parallel in (True, False):
#                        print "Using Parallel SFA = ", parallel                         
#                        print "Size of x: ", input_dim, " # Samples: ", num_samples, " output_dim = ", output_dim 
#                #        x[:,2]=x[:,3]
#                #       sfa = mdp.nodes.SFANode(svd=True)
#                #    'sequence', 'complete', 'mixed'
#                        sfa = mdp.nodes.SFANode(output_dim=output_dim, block_size=block_size, train_mode='mixed')
#
#                        t1 = time.time()
#                        if parallel == False:
#                            sfa.train(x_train)
#                        else:
#                            print "requested chunk_size=", (x_train.shape[0]/block_size) / (ncpus-1)
#                            sfa.train(x_train, scheduler=scheduler, chunk_size=(x_train.shape[0]/block_size) / (n_parallel-1))
#                        sfa.stop_training()
#                        t2 = time.time()
#    #                    for i in range(num_reps):
#    #                        x = numpy.random.random((1,size_var))
#    #                        y = sfa.execute(x)
#                        t3 = time.time()
#
#                        y = sfa.execute(x_exec)
#                        print "y=", y
#                        t4 = time.time()
#        
#    #                    y = numpy.random.random((1,output_dim))
#    #                    x = sfa.inverse(y)
#        
#                        t5 = time.time()                
#    #                    for i in range(num_reps_inv):
#    #                        y = numpy.random.random((1,output_dim))
#    #                        x = sfa.inverse(y)
#                        t6 = time.time()
##                        y = numpy.random.random((num_reps_inv,output_dim))
##                        x = sfa.inverse(y)
#                        t7 = time.time()
#        
#                        print 'Training: %0.3f ms' % ((t2-t1)*1000.0),
#                        print 'Single Execution %0.3f ms, ' % ((t3-t2)*1000.0 / num_reps ),
#                        print 'Batch exec %0.3f ms, ' % ((t4-t3)*1000.0 / num_reps ),
#                        print 'PINV computation %0.3f ms, ' % ((t5-t4)*1000.0 / num_reps_inv ),
#                        print 'Sing. inverse %0.3f ms, ' % ((t6-t5)*1000.0 / num_reps_inv ),
#                        print 'Batch inv %0.3f ms, ' % ((t7-t6)*1000.0 / num_reps_inv )
#    print "Node Benchmarking Finished!"


##SECOND EXPERIMENT... PCA / Whitening
node_classes = (mdp.nodes.PCANode, mdp.nodes.WhiteningNode)

print "Benchmarking regular/parallel pca_node"
for input_dim in input_dim_set:
    for num_samples in num_samples_set:
        for node in node_classes:
            x_train = numpy.random.random((num_samples,input_dim))
            x_exec = numpy.random.random((num_samples,input_dim))
            for output_dim in output_dim_set:
                    for parallel in (True, False):
                        print "Using Parallel Node = ", parallel, " of class: ", node                         
                        print "Size of x: ", input_dim, " # Samples: ", num_samples, " output_dim = ", output_dim 
                #        x[:,2]=x[:,3]
                #       sfa = mdp.nodes.SFANode(svd=True)
                #    'sequence', 'complete', 'mixed'
                        pca = node(output_dim=output_dim)

                        t1 = time.time()
                        if parallel == False:
                            pca.train(x_train)
                        else:
                            print "requested chunk_size_samples=", x_train.shape[0] / (n_parallel-1)
                            pca.train(x_train, scheduler=scheduler, chunk_size_samples= x_train.shape[0] / (n_parallel-1))
                        pca.stop_training()
                        t2 = time.time()
    #                    for i in range(num_reps):
    #                        x = numpy.random.random((1,size_var))
    #                        y = sfa.execute(x)
                        t3 = time.time()

                        y = pca.execute(x_exec)
                        print "y=", y
                        t4 = time.time()
        
    #                    y = numpy.random.random((1,output_dim))
    #                    x = sfa.inverse(y)
        
                        t5 = time.time()                
    #                    for i in range(num_reps_inv):
    #                        y = numpy.random.random((1,output_dim))
    #                        x = sfa.inverse(y)
                        t6 = time.time()
#                        y = numpy.random.random((num_reps_inv,output_dim))
#                        x = sfa.inverse(y)
                        t7 = time.time()
        
                        print 'Training: %0.3f ms' % ((t2-t1)*1000.0),
                        print 'Single Execution %0.3f ms, ' % ((t3-t2)*1000.0 / num_reps ),
                        print 'Batch exec %0.3f ms, ' % ((t4-t3)*1000.0 / num_reps ),
                        print 'PINV computation %0.3f ms, ' % ((t5-t4)*1000.0 / num_reps_inv ),
                        print 'Sing. inverse %0.3f ms, ' % ((t6-t5)*1000.0 / num_reps_inv ),
                        print 'Batch inv %0.3f ms, ' % ((t7-t6)*1000.0 / num_reps_inv )
    print "Node Benchmarking Finished!"

scheduler.shutdown()
 
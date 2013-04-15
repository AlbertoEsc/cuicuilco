import time
import numpy
import mdp
import eban_SFA_libs



print "**********************************************"
print "BENCHMARKING APPEND METHOD"
print "**********************************************"
appends = (100000, 200000)
for num_appends in appends:
    w = []
    t1 = time.time()
    for i in range(num_appends):
        w.append([w, 0, 1, w, 0.1, 0.7, 3.5, 2.7, 100.2, w])
    t2 = time.time()
    print 'Appending %d times took %0.3f ms' % (num_appends, (t2-t1)*1000.0)


print "**********************************************"
print "BENCHMARKING SIMPLE PCA/SFA"
print "**********************************************"

nodes = (mdp.nodes.SFANode, mdp.nodes.PCANode)
#size_var_set = (10, 20, 40, 80, 160) 
size_var_set = (160,) 
#num_samples_set = (1000, 2000, 4000, 8000, 16000, 32000, 64000) 
num_samples_set = (1000,) 
output_dim_set = (20,)
num_reps = 40
num_reps_inv = 20

#time = numpy.zero((len(size_var_set), len(num_samples_set)))
for node in nodes:
    print "Benchmarking node " + str(node)
    for num_samples in num_samples_set:
        for num_samples in num_samples_set:
            for size_var in size_var_set:
                for output_dim in output_dim_set:
                    print "Size of x: ", size_var, " # Samples: ", num_samples, " # Repetitions: ", num_reps, "# Repetitions Inverse: ", num_reps_inv, " output_dim = ", output_dim 
            #        x[:,2]=x[:,3]
            #       sfa = mdp.nodes.SFANode(svd=True)
                    sfa = node(output_dim=output_dim)
                    x = numpy.random.random((num_samples,size_var))
                    t1 = time.time()
                    sfa.train(x)
                    sfa.stop_training()
                    t2 = time.time()
                    for i in range(num_reps):
                        x = numpy.random.random((1,size_var))
                        y = sfa.execute(x)
                    t3 = time.time()
                    x = numpy.random.random((num_reps,size_var))
                    y = sfa.execute(x)
                    t4 = time.time()
    
                    y = numpy.random.random((1,output_dim))
                    x = sfa.inverse(y)
    
                    t5 = time.time()                
                    for i in range(num_reps_inv):
                        y = numpy.random.random((1,output_dim))
                        x = sfa.inverse(y)
                    t6 = time.time()
                    y = numpy.random.random((num_reps_inv,output_dim))
                    x = sfa.inverse(y)
                    t7 = time.time()
    
                    print 'Training: %0.3f ms' % ((t2-t1)*1000.0),
                    print 'Single Execution %0.3f ms, ' % ((t3-t2)*1000.0 / num_reps ),
                    print 'Batch exec %0.3f ms, ' % ((t4-t3)*1000.0 / num_reps ),
                    print 'PINV computation %0.3f ms, ' % ((t5-t4)*1000.0 / num_reps_inv ),
                    print 'Sing. inverse %0.3f ms, ' % ((t6-t5)*1000.0 / num_reps_inv ),
                    print 'Batch inv %0.3f ms, ' % ((t7-t6)*1000.0 / num_reps_inv )
    print "Node Benchmarking Finished!"


#class PInvSwitchboard(mdp.hinet.Switchboard):
#    def __init__(self, input_dim, connections, slow_inv=False, type_inverse="average"):
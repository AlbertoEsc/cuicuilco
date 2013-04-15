import mdp
import numpy as np
import mdp.parallel
from mdp.parallel.makeparallel import make_flow_parallel
import sys

sys.path.append("/home/escalafl/workspace/cuicuilco/src")
import patch_mdp
#import mdp.parallel


node1 = mdp.nodes.PCANode(input_dim=1000, output_dim=10)
node2 = mdp.nodes.SFANode(input_dim=10, output_dim=10, train_mode ='mixed', block_size=10)
flow = node1 + node2
n_data_chunks = 1
n_processes = 2
data_iterables = [[mdp.numx_rand.random((20000, 1000))
                   for _ in range(n_data_chunks)]
                   for _ in range(n_processes)]
flags = [[(False, False ) 
          for _ in range(n_data_chunks)]
          for _ in range(n_processes)]

flags[n_processes-1][0] = (True, True)
          
#data_iterables = [mdp.numx_rand.random((20000, 2000))
#                   for _ in range(2)]

scheduler = mdp.parallel.ProcessScheduler(n_processes=n_processes)
print "***********1"
parallel_flow = make_flow_parallel(flow)
parallel_flow.verbose = True
print "***********2"
parallel_flow.train(data_iterables, scheduler=scheduler, flags=flags)
print "***********3"
scheduler.shutdown()
quit()


node1 = mdp.parallel.ParallelPCANode(input_dim=3000, output_dim=10)
node2 = mdp.parallel.ParallelSFANode(input_dim=10, output_dim=10)
parallel_flow = mdp.parallel.ParallelFlow([node1, node2])

n_data_chunks = 2

data_iterables = [[mdp.numx_rand.random((40000, 3000))
                    for _ in range(n_data_chunks)]
                    for _ in range(2)]

print "data_iterables =", data_iterables
#print "data_iterables.shape =", data_iterables
scheduler = mdp.parallel.ProcessScheduler(n_processes=2)
try:
    parallel_flow.train(data_iterables, scheduler=scheduler)
finally:
    scheduler.shutdown()

quit()
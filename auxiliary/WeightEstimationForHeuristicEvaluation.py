import numpy
import mdp

###################################################################################################
######### CODE FOR DETERMINATION OF WEIGHTS
num_experiments = 6
num_problems = 9
problem_OV_index = num_problems-2
problem_OA_index = num_problems-1 
data_dim = 60
num_samples = 10000

net_num_layers = 9 # height = width = ((4)*2*2*2*2) = 64
net_sfa_out_dim = 30
net_num_samples = 30000
correct_all_metrics_from_network = True

metric_results = numpy.zeros((num_experiments, num_problems))

#Results from "heuristic evaluation" paper
metric_results = numpy.array([[0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.0700, 1.0000],
                              [0.0000, 0.7810, 0.7071, 0.7933, 0.8209, 0.7137, 0.7163, 0.4810, 1.0520],
                              [0.3777, 0.7348, 0.6706, 0.7725, 0.8084, 0.6603, 0.6608, 0.4810, 0.9631],
                              [0.2648, 0.7431, 0.7225, 0.7766, 0.8202, 0.6673, 0.6607, 0.4990, 0.9793],
                              [0.0000, 0.7341, 1.0001, 0.7766, 1.0000, 1.0000, 1.0001, 0.1100, 0.9854],
                              [0.0000, 0.7810, 1.0001, 0.7933, 1.0000, 1.0000, 1.0000, 0.1090, 1.0872]
                              ])

print "metric_results=", metric_results

metric_mask_1D = numpy.array([True, True, False, False, False, False, False, True, True], dtype=bool)
metric_mask_2D = numpy.zeros((num_experiments, num_problems), dtype=bool)
metric_mask_2D[:] = metric_mask_1D
#print metric_mask_2D

approximated_results = metric_results + 0.0

if correct_all_metrics_from_network:
    #Theory: Score for P1, P2, P3, P4, P5, P6 and P7 is mostly insensitive to (data_dim, num_samples) and (net_sfa_out_dim, net_num_samples)
    #They might be made worse by net_num_layers. 
    #Px <- 1 - (1 - Px) ** net_num_layers
    approximated_results[:, 0:num_problems-2] = 1.0 - (1.0 - metric_results[:, 0:num_problems-2]) ** net_num_layers

    #Theory: Score for P_OV is sensitive to all!!!
    #dimension_increase = net_sfa_out_dim * 1.0 / data_dim
    #num_samples_increase = net_num_samples * 1.0 / num_samples
    #ov_increase = (net_num_layers / num_samples_increase) * dimension_increase
    #P_OV <- P_OV * ov_increase
    dimension_increase = net_sfa_out_dim * 1.0 / data_dim
    num_samples_increase = net_num_samples * 1.0 / num_samples
    ov_increase = (net_num_layers / num_samples_increase) * dimension_increase
    #P_OV <- P_OV * ov_increase
    approximated_results[:, problem_OV_index] = metric_results[:, problem_OV_index]* ov_increase

    #Theory: Score for P_OA is mostly insensitive to (data_dim, num_samples) and (net_sfa_out_dim, net_num_samples)
    #P_OA <- P_OA ** net_num_layers
    approximated_results[:, problem_OA_index] = metric_results[:, problem_OA_index] ** net_num_layers

tmp = approximated_results[:,metric_mask_1D]

approximated_results = tmp+0

print "metric_results converted to the corresponding network =", approximated_results
   
true_performance = numpy.zeros(num_experiments)
true_performance = numpy.array([17.96, 27.54, 5.85, 6.18, 6.16, 16.44])
true_performance = true_performance.reshape((-1,1))
print "true_performance.flatten()=", true_performance.flatten()

LR_node = mdp.nodes.LinearRegressionNode(use_pinv=True)
LR_node.train(approximated_results, y=true_performance)
estimated_performance = LR_node.execute(approximated_results)
print "estimated_performance.flatten()=", estimated_performance.flatten()

print "metric_mask_1D=", metric_mask_1D
print "LR_node.beta.flatten()=", LR_node.beta.flatten()
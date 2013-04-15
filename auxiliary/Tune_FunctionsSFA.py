import random
import os
import numpy

std_noise=0.75
num_steps=400
k_vals = numpy.linspace(0.01, 3.0, 1)
d_vals = numpy.linspace(0.2, 2.0, 20)
all_basis = ["identity","SExp","0.8Exp", "QExp", "TExp", "QE_AN_exp", "TE_AN_exp", "QE_N_exp", "TE_N_exp", 
             "QE_E_exp", "TE_E_exp","QE_AE_exp", "TE_AE_exp",]
all_basis = ["0.8Exp", ]
cmd = "python FunctionApproximationForSFA_experiments.py --std_noise=%f --num_steps=%d --EnableDisplay=0 --ShowFunctions=0 --ShowFunctions2D=0 --SelectFunction=%s --k=%f --d=%f --OnlySelectedFunction=1"

print "std_noise=%f --num_steps=%d"%(std_noise, num_steps)

best_performances = {}
for i, base in enumerate(all_basis):
    print "base:", base
    for k in k_vals:
        for d in d_vals:
            cmd2 = cmd%(std_noise, num_steps, base, k, d)
#            print "Executing:", cmd2
            fin, fout = os.popen4(cmd2)
            result = fout.readlines()
            last_line = result[-1]
            perf  = float(last_line)
            print " Perf(%2.3f,%2.3f)="%(k, d), perf, 
            if base in best_performances.keys(): 
                if best_performances[base][0] > perf:
                    best_performances[base] = (perf, k, d)
            else:
                best_performances[base] = (perf, k, d)

print ""
for i, base in enumerate(all_basis):
    print "best_performances[%s]="%base, best_performances[base]